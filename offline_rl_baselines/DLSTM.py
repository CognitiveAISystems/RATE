# Adopted from https://github.com/max7born/decision-lstm
import torch
import torch.nn as nn
import torch.nn.functional as F

from RATE.env_encoders import ObsEncoder, ActEncoder, RTGEncoder, ActDecoder

class DecisionLSTM(nn.Module):
    def __init__(
        self, 
        state_dim, 
        act_dim,
        d_model=64, 
        hidden_layers=[128, 128],
        dropout=0.1,
        env_name='mujoco',
        padding_idx=None,
        lstm_layers=1,
        bidirectional=False,
        max_ep_len=1000,
        backbone='lstm',
        **kwargs
    ):    
        super(DecisionLSTM, self).__init__()

        self.d_embed = d_model
        self.d_model = d_model
        self.env_name = env_name
        self.act_dim = act_dim
        self.padding_idx = padding_idx
        self.mem_tokens = None
        self.attn_map = None
        self.lstm_layers = lstm_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.embed_timestep = nn.Embedding(max_ep_len, self.d_embed)
        self.backbone = backbone.lower()
        
        # Энкодеры для разных типов входных данных
        self.state_encoder = ObsEncoder(self.env_name, state_dim, self.d_embed).obs_encoder
        self.action_embeddings = ActEncoder(self.env_name, act_dim, self.d_embed).act_encoder
        self.ret_emb = RTGEncoder(self.env_name, self.d_embed).rtg_encoder
        
        # RNN для обработки последовательности триплетов
        self.hidden_size = hidden_layers[-1]
        
        # Удаляем triplet_projector, вместо него будем использовать прямую конкатенацию
        self.dropout = nn.Dropout(dropout)
        
        # Выбор рекуррентной сети на основе параметра backbone
        if self.backbone == 'lstm':
            self.backbone_net = nn.LSTM(
                input_size=d_model,
                hidden_size=self.hidden_size,
                num_layers=lstm_layers,
                dropout=dropout if lstm_layers > 1 else 0,
                batch_first=True,
                bidirectional=bidirectional
            )
        elif self.backbone == 'gru':
            self.backbone_net = nn.GRU(
                input_size=d_model,
                hidden_size=self.hidden_size,
                num_layers=lstm_layers,
                dropout=dropout if lstm_layers > 1 else 0,
                batch_first=True,
                bidirectional=bidirectional
            )
        else:
            raise ValueError(f"Unsupported RNN type: {self.backbone}. Choose 'lstm' or 'gru'.")
        
        self.output_dim = self.hidden_size * self.num_directions
        
        # Декодер действий
        self.head = ActDecoder(self.env_name, act_dim, self.output_dim).act_decoder
        
        # Инициализация весов RNN
        for name, param in self.backbone_net.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def init_hidden(self, batch_size, device):
        """Инициализация скрытых состояний RNN"""
        if self.backbone == 'lstm':
            # LSTM требует два скрытых состояния (h_0 и c_0)
            h_0 = torch.zeros(
                self.lstm_layers * self.num_directions,
                batch_size,
                self.hidden_size,
                device=device
            )
            c_0 = torch.zeros(
                self.lstm_layers * self.num_directions,
                batch_size,
                self.hidden_size,
                device=device
            )
            
            nn.init.orthogonal_(h_0)
            nn.init.orthogonal_(c_0)
            
            return (h_0, c_0)
        else:  # GRU
            # GRU требует только одно скрытое состояние
            h_0 = torch.zeros(
                self.lstm_layers * self.num_directions,
                batch_size,
                self.hidden_size,
                device=device
            )
            
            nn.init.orthogonal_(h_0)
            
            return h_0

    def reshape_states(self, states):
        reshape_required = False

        if len(states.shape) == 5:
            reshape_required = True
            B, B1, C, H, W = states.shape
        elif len(states.shape) == 6:
            reshape_required = True
            B, B1, _, C, H, W = states.shape
        else:
            B, B1, _ = states.shape
        
        if reshape_required:
            states = states.reshape(-1, C, H, W).type(torch.float32).contiguous()

        return B, B1, states, reshape_required
    
    def encode_actions(self, actions):
        use_long = False
        for name, module in self.action_embeddings.named_children():
            if isinstance(module, nn.Embedding):
                use_long = True
        if use_long:
            actions = actions.to(dtype=torch.long, device=actions.device)
            if self.padding_idx is not None:
                actions = torch.where(
                    actions == self.padding_idx,
                    torch.tensor(self.act_dim),
                    actions,
                )
            action_embeddings = self.action_embeddings(actions).squeeze(2)
        else:
            action_embeddings = self.action_embeddings(actions)

        return action_embeddings

    def forward(self, states, actions=None, rtgs=None, target=None, timesteps=None, mem_tokens=None, masks=None, hidden=None, *args, **kwargs):
        B, B1, states, reshape_required = self.reshape_states(states)
        
        # Получаем эмбеддинги состояний
        state_embeddings = self.state_encoder(states)
        if reshape_required:
            state_embeddings = state_embeddings.reshape(B, B1, self.d_embed)
        
        # Получаем эмбеддинги ожидаемых наград
        rtg_embeddings = self.ret_emb(rtgs)
        
        # Подготовка последовательности токенов в формате RATE
        if actions is not None:
            action_embeddings = self.encode_actions(actions)
            
            # Формируем токены по аналогии с RATE - чередуем rtg, state, action
            token_embeddings = torch.zeros((B, B1*3 - int(target is None), self.d_embed), dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:, ::3, :] = rtg_embeddings
            token_embeddings[:, 1::3, :] = state_embeddings
            token_embeddings[:, 2::3, :] = action_embeddings[:, :B1, :]
            
        else:
            # Если действия отсутствуют, чередуем только rtg и state
            token_embeddings = torch.zeros((B, B1*2, self.d_embed), dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:, ::2, :] = rtg_embeddings
            token_embeddings[:, 1::2, :] = state_embeddings
        
        # Применяем дропаут
        token_embeddings = self.dropout(token_embeddings)
        
        # Если скрытые состояния не переданы, инициализируем их
        if hidden is None:
            hidden = self.init_hidden(B, states.device)
        
        # Обрабатываем последовательность с помощью выбранной RNN
        features, new_hidden = self.backbone_net(token_embeddings, hidden)
        
        # Получаем предсказания действий
        logits = self.head(features)

        if actions is not None:
            logits = logits[:, 1::3, :]
        else:
            logits = logits[:, 1:, :]    
        
        output = {
            'logits': logits,
            'new_mems': None,
            'mem_tokens': None,
            'hidden': new_hidden
        }
        
        return output

    def reset_hidden(self, batch_size=None, device=None):
        """Сброс скрытых состояний RNN"""
        if batch_size is None or device is None:
            return None
        return self.init_hidden(batch_size, device)
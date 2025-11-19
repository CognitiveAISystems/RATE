import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from decision_transformer.model import Block
from decision_transformer.utils import encode_return


class CNNEncoder(nn.Module):
    """CNN encoder for processing image observations."""
    
    def __init__(self, input_channels=3, output_dim=128):
        super().__init__()
        
        # Architecture inspired by common ViZDoom approaches
        # Input: (3, 64, 112) -> ViZDoom Two Colors resolution
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        
        # Calculate the flattened size after conv layers
        # For (3, 64, 112) input:
        # After conv1 (k=8, s=4, p=2): (32, 16, 28)
        # After conv2 (k=4, s=2, p=1): (64, 8, 14)
        # After conv3 (k=3, s=1, p=1): (64, 8, 14)
        # Flattened: 64 * 8 * 14 = 7168
        
        self.fc = nn.Linear(64 * 8 * 14, output_dim)
        
    def forward(self, x):
        """
        Args:
            x: Image tensor of shape (B, T, C, H, W) or (B, C, H, W)
        Returns:
            Encoded features of shape (B, T, output_dim) or (B, output_dim)
        """
        # Handle both batched sequences and single images
        if len(x.shape) == 5:  # (B, T, C, H, W)
            B, T, C, H, W = x.shape
            x = x.reshape(B * T, C, H, W)
            x = self.conv_layers(x)
            x = x.reshape(B * T, -1)
            x = self.fc(x)
            x = x.reshape(B, T, -1)
        else:  # (B, C, H, W)
            x = self.conv_layers(x)
            x = x.reshape(x.size(0), -1)
            x = self.fc(x)
        
        return x


class ElasticDecisionTransformerViZDoom(nn.Module):
    """Elastic Decision Transformer adapted for ViZDoom with image observations."""
    
    def __init__(
        self,
        img_channels=3,
        img_height=64,
        img_width=112,
        act_dim=5,  # ViZDoom Two Colors has 5 discrete actions
        n_blocks=4,
        h_dim=128,
        context_len=50,
        n_heads=4,
        drop_p=0.1,
        env_name="vizdoom",
        max_timestep=4096,
        num_bin=120,
        dt_mask=False,
        rtg_scale=1000,
        num_inputs=3,
        real_rtg=False,
    ):
        super().__init__()
        
        self.img_channels = img_channels
        self.img_height = img_height
        self.img_width = img_width
        self.act_dim = act_dim
        self.h_dim = h_dim
        self.num_bin = num_bin
        self.env_name = env_name
        self.rtg_scale = rtg_scale
        self.num_inputs = num_inputs
        
        # CNN encoder for image observations
        self.state_encoder = CNNEncoder(
            input_channels=img_channels,
            output_dim=h_dim
        )
        
        # Transformer blocks
        input_seq_len = num_inputs * context_len
        blocks = [
            Block(
                h_dim,
                input_seq_len,
                n_heads,
                drop_p,
                mgdt=True,
                dt_mask=dt_mask,
                num_inputs=num_inputs,
                real_rtg=real_rtg,
            )
            for _ in range(n_blocks)
        ]
        self.transformer = nn.Sequential(*blocks)
        
        # Embedding layers
        self.embed_ln = nn.LayerNorm(h_dim)
        self.embed_timestep = nn.Embedding(max_timestep, h_dim)
        self.embed_rtg = torch.nn.Linear(1, h_dim)
        
        # Discrete action embedding for ViZDoom
        self.embed_action = torch.nn.Embedding(act_dim, h_dim)
        
        # Prediction heads
        self.predict_rtg = torch.nn.Linear(h_dim, int(num_bin))
        self.predict_rtg2 = torch.nn.Linear(h_dim, 1)
        self.predict_action = nn.Linear(h_dim, act_dim)  # Logits for discrete actions
        self.predict_reward = torch.nn.Linear(h_dim, 1)
        
    def forward(self, timesteps, states, actions, returns_to_go, rewards=None):
        """
        Args:
            timesteps: (B, T) or (B, T, 1)
            states: (B, T, C, H, W) - image observations
            actions: (B, T) - discrete action indices
            returns_to_go: (B, T, 1)
            rewards: (B, T, 1) - optional
        """
        # Ensure timesteps are 2D (B, T)
        if len(timesteps.shape) == 3:
            timesteps = timesteps.squeeze(-1)
        
        B, T = timesteps.shape
        
        # Encode image observations
        state_features = self.state_encoder(states)  # (B, T, h_dim)
        
        # Process returns-to-go
        returns_to_go = returns_to_go.float()
        returns_to_go = (
            encode_return(
                self.env_name, returns_to_go, num_bin=self.num_bin, rtg_scale=self.rtg_scale
            )
            - self.num_bin / 2
        ) / (self.num_bin / 2)
        
        # Get embeddings
        time_embeddings = self.embed_timestep(timesteps)
        state_embeddings = state_features + time_embeddings
        action_embeddings = self.embed_action(actions.long()) + time_embeddings
        returns_embeddings = self.embed_rtg(returns_to_go) + time_embeddings
        
        # Stack embeddings: (state, rtg, action) for each timestep
        h = (
            torch.stack(
                (
                    state_embeddings,
                    returns_embeddings,
                    action_embeddings,
                ),
                dim=1,
            )
            .permute(0, 2, 1, 3)
            .reshape(B, self.num_inputs * T, self.h_dim)
        )
        
        h = self.embed_ln(h)
        
        # Transformer
        h = self.transformer(h)
        h = h.reshape(B, T, self.num_inputs, self.h_dim).permute(0, 2, 1, 3)
        
        # Predictions
        return_preds = self.predict_rtg(h[:, 0])  # predict next rtg given s
        return_preds2 = self.predict_rtg2(h[:, 0])  # implicit return prediction
        action_preds = self.predict_action(h[:, 1])  # predict action given s, R
        reward_preds = self.predict_reward(h[:, 2])  # predict reward given s, R, a
        
        return (
            None,  # state_preds (not used for ViZDoom)
            action_preds,
            return_preds,
            return_preds2,
            reward_preds,
        )


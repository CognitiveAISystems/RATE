import torch

# if not env.action_space.sample().shape == () -> action.unsqueeze(-1)
# if any(game in env_name for game in ['Battleship', 'MineSweeper']):
def coords_to_idx(action, board_size=8):
    # action shape: (batch_size, seq_length, 2) -> (batch_size, seq_length, 1)
    if len(action.shape) == 2:  # Single coordinate pair
        return action[0] * board_size + action[1]
    else:  # Batched input
        return (action[..., 0] * board_size + action[..., 1]).unsqueeze(-1)

def idx_to_coords(idx, board_size=8):
    # idx shape: (batch_size, seq_length, 1) or scalar
    if not torch.is_tensor(idx):
        idx = torch.tensor(idx)
    
    if idx.dim() == 0:  # Single index
        x = idx // board_size
        y = idx % board_size
        return torch.stack([x, y])
    else:  # Batched input
        idx = idx.squeeze(-1)  # Remove last dimension if present
        x = idx // board_size
        y = idx % board_size
        return torch.stack([x, y], dim=-1)  # Result: (batch_size, seq_length, 2)

# # * in act encoder:
# act = coords_to_idx(a)
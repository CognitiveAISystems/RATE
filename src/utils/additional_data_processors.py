import torch

# if any(game in env_name for game in ['Battleship', 'MineSweeper']):
def coords_to_idx(action: torch.Tensor, board_size: int = 8) -> torch.Tensor:
    """Convert 2D coordinates to 1D indices for grid-based environments.

    This function converts 2D grid coordinates (x, y) to 1D indices using the formula:
    idx = x * board_size + y. It handles both single coordinate pairs and batched inputs.

    Args:
        action: Input tensor containing 2D coordinates.
            - For single pair: shape (2,) containing [x, y]
            - For batched input: shape (batch_size, seq_length, 2) containing [x, y] pairs
        board_size: Size of the grid board (default: 8).
            Used for environments like:
            - Battleship: Easy (8x8), Medium (10x10), Hard (12x12)
            - Minesweeper: Easy (4x4), Medium (6x6), Hard (8x8)

    Returns:
        A tensor containing 1D indices:
            - For single pair: shape (1,) containing the index
            - For batched input: shape (batch_size, seq_length, 1) containing indices

    Examples:
        >>> # Single coordinate pair
        >>> coords = torch.tensor([2, 3])
        >>> idx = coords_to_idx(coords, board_size=8)  # Returns tensor([19])
        
        >>> # Batched coordinates
        >>> coords = torch.tensor([[[2, 3], [1, 4]], [[0, 0], [7, 7]]])
        >>> idx = coords_to_idx(coords, board_size=8)
        >>> # Returns tensor([[[19], [12]], [[0], [63]]])

    Note:
        This function is primarily used for converting actions in grid-based
        environments like Battleship and Minesweeper from 2D coordinates to
        1D indices for model input.
    """
    # action shape: (batch_size, seq_length, 2) -> (batch_size, seq_length, 1)
    if len(action.shape) == 2:  # Single coordinate pair
        return action[0] * board_size + action[1]
    else:  # Batched input
        return (action[..., 0] * board_size + action[..., 1]).unsqueeze(-1)

def idx_to_coords(idx: torch.Tensor, board_size: int = 8) -> torch.Tensor:
    """Convert 1D indices to 2D coordinates for grid-based environments.

    This function converts 1D indices to 2D grid coordinates (x, y) using the formulas:
    x = idx // board_size
    y = idx % board_size
    It handles both single indices and batched inputs.

    Args:
        idx: Input tensor containing 1D indices.
            - For single index: scalar or tensor of shape (1,)
            - For batched input: tensor of shape (batch_size, seq_length, 1)
        board_size: Size of the grid board (default: 8).
            Used for environments like:
            - Battleship: Easy (8x8), Medium (10x10), Hard (12x12)
            - Minesweeper: Easy (4x4), Medium (6x6), Hard (8x8)

    Returns:
        A tensor containing 2D coordinates:
            - For single index: shape (2,) containing [x, y]
            - For batched input: shape (batch_size, seq_length, 2) containing [x, y] pairs

    Examples:
        >>> # Single index
        >>> idx = torch.tensor(19)
        >>> coords = idx_to_coords(idx, board_size=8)  # Returns tensor([2, 3])
        
        >>> # Batched indices
        >>> idx = torch.tensor([[[19], [12]], [[0], [63]]])
        >>> coords = idx_to_coords(idx, board_size=8)
        >>> # Returns tensor([[[2, 3], [1, 4]], [[0, 0], [7, 7]]])

    Note:
        This function is primarily used for converting model outputs back to
        2D coordinates in grid-based environments like Battleship and Minesweeper.
        It is the inverse operation of coords_to_idx.
    """
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
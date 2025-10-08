import torch, torch.nn as nn


class RelativeBias(nn.Module):
    """
    Learned relative position bias D(Δ) per head, similar in spirit to Transformer‑XL.

    Core idea:
    - Attention scores are augmented with a learnable term that depends on the
      relative distance Δ = q_pos − k_pos between query and key positions.
    - For each head h, we learn a scalar bias D_h(Δ). This bias is added to the
      raw dot‑product scores before softmax, encouraging or discouraging specific
      relative offsets.

    Indexing scheme:
    - We clamp Δ to the range [-(max_dist−1), +(max_dist−1)] to bound the table.
    - We map Δ -> index = Δ + (max_dist − 1) \in [0, 2*max_dist − 2].

    Shapes:
    - q_pos: [T]        absolute integer positions for queries (tokens)
    - k_pos: [B, M]     absolute integer positions for keys (e.g., memory slots)
    - output: [B, T, M, H] where H = num_heads. This is broadcast‑compatible with
      attention score tensors of shape [B, H, T, M] after a simple permute.

    Usage patterns in this codebase:
    - Token reads from memory: rel_bias = RelativeBias(...)(tok_pos, mem_pos)
      gives [B, T, M, H] and is permuted downstream when needed.
    - Memory writes from tokens: distances invert the sign (k_pos − q_pos),
      which corresponds to -(q_pos − k_pos). We implement this by transposing
      and negating the previously computed tensor (see comments at call sites).
    """

    def __init__(self, num_heads: int, max_dist: int = 2048):
        super().__init__()
        self.max_dist = max_dist
        # Embedding table over clamped relative offsets; returns a vector per head
        self.bias = nn.Embedding(2 * max_dist - 1, num_heads)

    def forward(self, q_pos: torch.LongTensor, k_pos: torch.LongTensor):
        """
        Compute relative bias for all query/key pairs.

        Args:
            q_pos: [T] absolute positions for queries
            k_pos: [B, M] absolute positions for keys

        Returns:
            rel_bias: [B, T, M, H] bias values per head, broadcast‑friendly
        """
        # Pairwise relative offsets Δ = q − k for all (T, M) pairs per batch
        rel = q_pos[None, :, None] - k_pos[:, None, :]      # [B, T, M]
        # Clamp to the supported range and shift to [0, 2*max_dist-2]
        rel = rel.clamp(-self.max_dist + 1, self.max_dist - 1) + self.max_dist - 1
        # Lookup per‑head biases: [B, T, M, H]
        
        return self.bias(rel)

    def mem_to_tok(self, mem_pos: torch.LongTensor, tok_pos: torch.LongTensor):
        """
        Compute bias for memory (queries) attending to tokens (keys).

        Args:
            mem_pos: [B, M] absolute positions for memory slots (queries)
            tok_pos: [T]    absolute positions for tokens (keys)

        Returns:
            rel_bias: [B, H, M, T]
        """
        rel = mem_pos[:, :, None] - tok_pos[None, None, :]   # [B, M, T]
        rel = rel.clamp(-self.max_dist + 1, self.max_dist - 1) + self.max_dist - 1
        # bias: [B, M, T, H] -> permute to [B, H, M, T]
        return self.bias(rel).permute(0, 3, 1, 2)
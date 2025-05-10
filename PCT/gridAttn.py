import torch
import torch.nn as nn

# === Grid Cell Positional Encoding ===
class GridCellPositionalEncoding(nn.Module):
    def __init__(self, dimension, qk_dim, num_heads):
        """
        Initialize the Grid Cell Positional Encoding using PyTorch.

        Parameters:
        - dimension: int, spatial dimensionality (e.g., 2 for 2D space)
        - embedding_dim: int, the total embedding dimension (must be divisible by 2*(dimension+1))
        """
        super().__init__()
        self.dimension = dimension
        self.embedding_dim = qk_dim
        self.num_heads = num_heads
        self.num_head_dim = self.embedding_dim // num_heads

        self.n = self.dimension + 1 if self.dimension > 1 else 1
        
        assert self.num_head_dim % (2 * self.n) == 0, "per num head embedding_dim must be divisible by 2*(dimension+1)"
        self.num_scales = self.num_head_dim // (2 * self.n)
        
        # fixed omega for per block
        omegas = []
        for _ in range(self.num_heads):
            omega = self._generate_simplex_vectors_with_projection(self.dimension)
            omegas.append(omega)
        self.register_buffer("omegas", torch.stack(omegas, dim=0))

    def _generate_simplex_vectors_with_projection(self, dimension, apply_random_rotation=True):
        """
        Generate n wave vectors (omega) in an n-dimensional space using a regular simplex projection.
        """
        if dimension == 1:
            return torch.tensor([[1.0]], dtype=torch.float32)  # 1D case

        points = torch.eye(dimension + 1, dtype=torch.float32)
        points -= points.mean(dim=0)
        U, _, _ = torch.linalg.svd(points.T, full_matrices=False)
        reduced_vectors = U[:, :-1]
        reduced_vectors /= torch.norm(reduced_vectors, dim=1, keepdim=True)
        if apply_random_rotation:
            Q, _ = torch.linalg.qr(torch.randn(dimension, dimension))
            reduced_vectors = reduced_vectors @ Q.T
        return reduced_vectors

    def _generate_batch_encoding(self, positions, base = 100):
        """
        Generate multi-scale positional encoding for batch input positions.

        Parameters:
        - positions: (batch_size, seq_length, d) tensor, input position vectors.

        Returns:
        - theta: (batch_size, num_head, seq_length, n, num_scales) real-valued tensor.
        """
        theta_heads = []
        mag = 1 / (base ** (2 * self.n * torch.arange(self.num_scales, dtype=torch.float32, device=positions.device)[:, None] / self.num_head_dim))
        for i in range(self.num_heads):
            omega = self.omegas[i]
            theta = torch.einsum('bsd,nd->bsn', positions, omega)[:, :, :, None] * mag.T
            theta_heads.append(theta)
        theta_heads = torch.stack(theta_heads, dim=0).permute(1, 0, 2, 3, 4)  # (B, H, N, n, S)
        return theta_heads

    def _compute_rotation_vectors(self, theta):
        """
        Convert positional encoding angles directly into cosine and sine components.

        Parameters:
        - theta: (batch_size, num_head, seq_length, n, S) tensor of angles.

        Returns:
        - cos_vec: (batch_size, num_head, seq_length, n, S) cosine components.
        - sin_vec: (batch_size, num_head, seq_length, n, S) sine components.
        """
        return torch.cos(theta), torch.sin(theta)

    def _apply_rotation(self, q, cos_vec, sin_vec):
        """
        Apply the rotation transformation similar to RoPE in a memory-efficient manner.

        Parameters:
        - q: (batch_size, num_heads, seq_length, 2 * n * S) Input tensor representing token embeddings.
        - cos_vec: (batch_size, num_heads, seq_length, n, S) Cosine components.
        - sin_vec: (batch_size, num_heads, seq_length, n, S) Sine components.

        Returns:
        - q_rotated: (batch_size, num_heads, seq_length, 2 * n * S) Output tensor after transformation.
        """
        B, H, N, D = q.shape
        n, S = cos_vec.shape[-2:]

        assert D == 2 * n * S, f"Expected last dim {D} == 2 * n * S = {2 * n * S}"

        # Reshape q to (B, H, N, n, S, 2)
        q = q.view(B, H, N, n, S, 2)
        q_even = q[..., 0]  # (B, H, N, n, S)
        q_odd = q[..., 1]   # (B, H, N, n, S)

        # Apply complex rotation
        q_rotated_even = q_even * cos_vec - q_odd * sin_vec
        q_rotated_odd = q_even * sin_vec + q_odd * cos_vec

        q_rot = torch.stack([q_rotated_even, q_rotated_odd], dim=-1)  # (B, H, N, n, S, 2)
        q_rot = q_rot.view(B, H, N, 2 * n * S)  # Flatten back
        return q_rot

    def forward(self, positions, q, k):
        """
        Compute the full grid cell positional encoding and apply rotation transformation.

        Parameters:
        - positions: (batch_size, seq_length, d) tensor, input position vectors.
        - q: (batch_size,  num_heads, seq_length, 2nS) tensor, input token embeddings.

        Returns:
        - q(k)_rotated: (batch_size, num_heads, seq_length, 2nS) tensor, output after transformation.
        """
        theta = self._generate_batch_encoding(positions)
        cos_vec, sin_vec = self._compute_rotation_vectors(theta)
        q_rotated = self._apply_rotation(q, cos_vec, sin_vec)
        k_rotated = self._apply_rotation(k, cos_vec, sin_vec)
        return q_rotated, k_rotated

# === Attention Layers ===
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class GridPEAttention(Attention):
    """Multi-head Attention block with GridCell Positional Encoding (RoPE style)."""
    def __init__(self, qk_dim, dimension=2, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__(qk_dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop)

        self.grid_pe = GridCellPositionalEncoding(
            dimension=dimension,
            qk_dim=qk_dim,
            num_heads=num_heads
        )

    def forward(self, x, positions):
        """
        Parameters:
        - x: (B, N + 1, C) input tokens (include cls token)
        - positions: (B, N, d) spatial/temporal positions
        """
        B, N, C = x.shape
        # [3, B, H, N, C_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, N, C_head)

        # ==== Rotary Positional Encoding ====
        q_rot, k_rot = q.clone(), k.clone()
        q_rot[:,:,1:], k_rot[:,:,1:] = self.grid_pe(positions, q[:,:,1:], k[:,:,1:]) # (B, H, N, C_head) 
  
        # ==== Attention ====
        q_rot = q_rot * self.scale  # scaled dot-product
        attn = (q_rot @ k_rot.transpose(-2, -1))  # (B, H, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, torch.distributed as dist
from util import sample_and_group 
from typing import Tuple
from functools import partial
import os

def compute_axial_cis(xyz, dim: int, base: float = 100.0):
    K = dim // 6
    scale_factor = 10
    freqs = scale_factor / (base ** (torch.arange(0, dim, 6, device=xyz.device)[:K].float() / dim))
    
    t_x = xyz[:, :, 0]
    t_y = xyz[:, :, 1]
    t_z = xyz[:, :, 2]
    
    encoded_x = t_x.unsqueeze(-1) * freqs
    encoded_y = t_y.unsqueeze(-1) * freqs
    encoded_z = t_z.unsqueeze(-1) * freqs
    
    cis_x = torch.polar(torch.ones_like(encoded_x), encoded_x)
    cis_y = torch.polar(torch.ones_like(encoded_y), encoded_y)
    cis_z = torch.polar(torch.ones_like(encoded_z), encoded_z)
    
    # The final shape is (B, N, 3*K), which equals (B, N, 3 * (dim//6)).
    return torch.cat([cis_x, cis_y, cis_z], dim=-1)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    B, H, N, C_head = xq.shape
    freqs_cis = freqs_cis.unsqueeze(1)  # Shape becomes (B, 1, N, C_head)
    freqs_cis = freqs_cis.expand(B, H, N, C_head//2)
    
    xq = xq.contiguous()
    xk = xk.contiguous()
    
    xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    xq_emb = xq_complex * freqs_cis
    xk_emb = xk_complex * freqs_cis

    xq_out = torch.view_as_real(xq_emb)
    xk_out = torch.view_as_real(xk_emb)

    xq_out = xq_out.flatten(3)
    xk_out = xk_out.flatten(3)
    
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)

class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Local_op, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6]) 
        x = x.permute(0, 1, 3, 2)   
        x = x.reshape(-1, d, s) 
        batch_size, _, N = x.size()
        x = F.relu(self.bn1(self.conv1(x))) # B, D, N
        x = F.relu(self.bn2(self.conv2(x))) # B, D, N
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x

class SA_Layer(nn.Module):
    def __init__(self, channels, emb_dim=252, num_heads=1):
        super(SA_Layer, self).__init__()

        head_dim = emb_dim // num_heads
        if head_dim % 6 != 0:
            # Adjust qk_dim so that head_dim becomes divisible by 6 (3 dimension * 2)
            emb_dim = (head_dim // 6) * 6 * num_heads

        self.num_heads = num_heads
        self.emb_dim = emb_dim
        self.head_dim = emb_dim // num_heads  # Each head's dimensionality

        # Convolution layers for Q, K, V
        self.q_conv = nn.Conv1d(channels, emb_dim, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, emb_dim, 1, bias=False)
        self.v_conv = nn.Conv1d(channels, emb_dim, 1, bias=False)

        # Transformation and normalization
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

        # For computing the rotary encoding
        self.compute_cis = partial(compute_axial_cis, dim=emb_dim // self.num_heads)

    def split_heads(self, x, num_heads):
        """ Split the tensor into multiple heads, reshaping (B, N, C) -> (B, H, N, C_head) """
        B, N, C = x.shape
        C_head = C // num_heads
        return x.view(B, N, num_heads, C_head).permute(0, 2, 1, 3)  # (B, H, N, C_head)

    def forward(self, x, xyz):
        # Input x: (B, C, N)
        B, C, N = x.shape

        # Q, K, V projection
        x_q = self.q_conv(x).permute(0, 2, 1)  # (B, N, C)
        x_k = self.k_conv(x).permute(0, 2, 1)
        x_v = self.v_conv(x).permute(0, 2, 1)

        # Split heads
        x_q = self.split_heads(x_q, self.num_heads)  # (B, H, N, C_head)
        x_k = self.split_heads(x_k, self.num_heads)
        x_v = self.split_heads(x_v, self.num_heads)

        # Rotary positional encoding
        freqs_cis = self.compute_cis(xyz)  # whatever shape your rotary uses
        x_q, x_k = apply_rotary_emb(x_q, x_k, freqs_cis)

        # Scaled dot-product attention
        C_head = x_q.shape[-1]
        x_q = x_q * (C_head ** -0.5)
        attn = torch.matmul(x_q, x_k.transpose(-2, -1))  # (B, H, N, N)
        attn = self.softmax(attn)
        
        # ---- compute distance & entropy ----
        with torch.no_grad():
            # attention distance: average weighted euclidean between each query-key pair
            # compute pairwise distances among xyz
            D = torch.cdist(xyz, xyz, p=2)             # (B, N, N)
            D = D.unsqueeze(1).expand(B, self.num_heads, N, N)
            attn_distance = (attn * D).sum(dim=-1).mean()

            # attention entropy
            eps = 1e-12
            ent = -(attn * torch.log(attn + eps)).sum(dim=-1)  # (B, H, N)
            attn_entropy = ent.mean()
            
        attn = attn / (1e-9 + attn.sum(dim=-1, keepdim=True))
        x_r = torch.matmul(attn, x_v)  # (B, H, N, C_head)
        x_r = x_r.permute(0, 2, 1, 3).reshape(B, N, self.num_heads * C_head).permute(0, 2, 1)  # (B, D, N)

        # Residual + FFN
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x, attn_distance, attn_entropy
    
class Point_Transformer_Last(nn.Module):
    def __init__(self, args, channels=252):
        super(Point_Transformer_Last, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)

        self.sa1 = SA_Layer(channels)
        self.sa2 = SA_Layer(channels)
        self.sa3 = SA_Layer(channels)
        self.sa4 = SA_Layer(channels)

    def forward(self, x, xyz):
        x = F.relu(self.bn1(self.conv1(x)))  # (B, C, N)

        attn_distances = []
        attn_entropies = []

        x1, d1, e1 = self.sa1(x, xyz)
        attn_distances.append(d1)
        attn_entropies.append(e1)

        x2, d2, e2 = self.sa2(x1, xyz)
        attn_distances.append(d2)
        attn_entropies.append(e2)

        x3, d3, e3 = self.sa3(x2, xyz)
        attn_distances.append(d3)
        attn_entropies.append(e3)

        x4, d4, e4 = self.sa4(x3, xyz)
        attn_distances.append(d4)
        attn_entropies.append(e4)

        x_cat = torch.cat((x1, x2, x3, x4), dim=1)  # (B, 4*C, N)

        return x_cat, attn_distances, attn_entropies

class PctRope(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PctRope, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=252)

        self.pt_last = Point_Transformer_Last(args)

        self.conv_fuse = nn.Sequential(nn.Conv1d(1260, 1024, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        xyz = x.permute(0, 2, 1)
        batch_size, C, N = x.size()
        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        # B, D, N
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=N, radius=0.15, nsample=32, xyz=xyz, points=x)         
        feature_0 = self.gather_local_0(new_feature)
        feature = feature_0.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=N//2, radius=0.2, nsample=32, xyz=new_xyz, points=feature) 
        feature_1 = self.gather_local_1(new_feature)

        x, attn_distances, attn_entropies = self.pt_last(feature_1, new_xyz)
        # output_dir = f"./attn_output/"
        # if output_dir is not None and (not dist.is_initialized() or dist.get_rank()==0):
        #     os.makedirs(output_dir, exist_ok=True)
        #     dist_path = os.path.join(output_dir, "pctrope_attn_distance.txt")
        #     ent_path  = os.path.join(output_dir, "pctrope_attn_entropy.txt")
        #     with open(dist_path, "a") as f_dist, open(ent_path, "a") as f_ent:
        #         for i, d in enumerate(attn_distances):
        #             f_dist.write(f"Block {i}: {d:.6f}\n")
        #         for i, e in enumerate(attn_entropies):
        #             f_ent.write(f"Block {i}: {e:.6f}\n")
        
        x = torch.cat([x, feature_1], dim=1)
        x = self.conv_fuse(x)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)

        return x
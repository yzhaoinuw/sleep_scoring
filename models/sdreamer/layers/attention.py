from inspect import isfunction

import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange
from timm.layers import DropPath


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_zero_(layer):
    nn.init.constant_(layer.weight, 0.0)
    if exists(layer.bias):
        nn.init.constant_(layer.bias, 0.0)


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


class ReluSquared(nn.Module):
    def forward(self, x):
        return F.relu(x) ** 2


class GLU(nn.Module):
    def __init__(self, dim_in, dim_out, activation):
        super().__init__()
        self.act = activation
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * self.act(gate)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        dim_out=None,
        mult=4,
        glu=False,
        swish=False,
        relu_squared=False,
        relu=False,
        post_act_ln=False,
        dropout=0.0,
        no_bias=False,
        zero_init_output=False,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)

        if relu_squared:
            activation = ReluSquared()
        if relu:
            activation = nn.ReLU()
        elif swish:
            activation = nn.SiLU()
        else:
            activation = nn.GELU()

        project_in = (
            nn.Sequential(nn.Linear(dim, inner_dim, bias=not no_bias), activation)
            if not glu
            else GLU(dim, inner_dim, activation)
        )

        self.ff = nn.Sequential(
            project_in,
            nn.LayerNorm(inner_dim) if post_act_ln else nn.Identity(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out, bias=not no_bias),
        )

        # init last linear layer to 0
        if zero_init_output:
            init_zero_(self.ff[-1])

    def forward(self, x):
        return self.ff(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(
        num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
    )


class CrossAttention(nn.Module):
    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        output_attentions=False,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim, bias=True), nn.Dropout(dropout)
        )
        self.att_fn = F.softmax
        self.output_attentions = output_attentions

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(
            lambda t: rearrange(t, "b ... n (h d) -> b h ... n d", h=h), (q, k, v)
        )

        dots = einsum("b h ... i d, b h ... j d -> b h ... i j", q, k) * self.scale
        mask_value = max_neg_value(dots)

        if exists(mask):
            assert (
                2 <= mask.ndim <= 4
            ), "attention mask must have greater than 2 dimensions but less than or equal to 4"
            if mask.ndim == 2:
                mask = rearrange(mask, "i j -> 1 1 i j")
            elif mask.ndim == 3:
                mask = rearrange(mask, "h i j -> 1 h i j")
            dots = dots.masked_fill(~mask, mask_value)

        attn = self.att_fn(dots, dim=-1)

        out = einsum("b h ... i j, b h ... j d -> b h ... i d", attn, v)
        out = rearrange(out, "b h ... n d -> b ... n (h d)", h=h)

        if self.output_attentions:
            return self.to_out(out), attn
        else:
            return self.to_out(out), None


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        path_drop=0.0,
        activation="glu",
        norm="layernorm",
        mult=4,
        layer_scale_init_values=0.1,
        output_attentions=False,
    ):
        super().__init__()
        glu, relu, relu_squared = False, False, False
        if activation == "glu":
            glu = True
        elif activation == "relu":
            relu = True
        else:
            relu_squared = True
        self.output_attentions = output_attentions
        self.attn1 = CrossAttention(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            output_attentions=output_attentions,
        )
        self.ff = FeedForward(
            dim,
            mult=mult,
            dropout=dropout,
            glu=glu,
            relu=relu,
            relu_squared=relu_squared,
        )
        self.norm1 = (
            nn.LayerNorm(dim)
            if norm == "layernorm"
            else nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(dim), Transpose(1, 2))
        )
        self.norm2 = (
            nn.LayerNorm(dim)
            if norm == "layernorm"
            else nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(dim), Transpose(1, 2))
        )
        self.drop_path = DropPath(path_drop) if path_drop > 0.0 else nn.Identity()

        self.gamma1 = (
            nn.Parameter(
                layer_scale_init_values * torch.ones((dim)), requires_grad=True
            )
            if layer_scale_init_values is not None
            else 1.0
        )

        self.gamma2 = (
            nn.Parameter(
                layer_scale_init_values * torch.ones((dim)), requires_grad=True
            )
            if layer_scale_init_values is not None
            else 1.0
        )

    def forward(self, x, context=None):
        new_x, attn = self.attn1(self.norm1(x), context)
        x = x + self.drop_path(self.gamma1 * new_x)
        x = x + self.drop_path(self.gamma2 * self.ff(self.norm2(x)))
        if self.output_attentions:
            return x, attn
        else:
            return x, None


class MoEBlock(nn.Module):
    def __init__(
        self,
        n_patches,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        path_drop=0.0,
        activation="glu",
        norm="layernorm",
        mult=4,
        with_mixffn=False,
        layer_scale_init_values=0.1,
        output_attentions=False,
    ):
        super().__init__()
        self.output_attentions = output_attentions
        glu, relu, relu_squared = False, False, False
        if activation == "glu":
            glu = True
        elif activation == "relu":
            relu = True
        else:
            relu_squared = True
        self.n_patches = n_patches
        self.drop_path = DropPath(path_drop) if path_drop > 0.0 else nn.Identity()
        self.norm1 = (
            nn.LayerNorm(dim)
            if norm == "layernorm"
            else nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(dim), Transpose(1, 2))
        )
        self.attn = Attention(
            dim,
            num_heads=n_heads,
            proj_drop=dropout,
            output_attentions=output_attentions,
        )
        self.norm2_eeg = (
            nn.LayerNorm(dim)
            if norm == "layernorm"
            else nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(dim), Transpose(1, 2))
        )
        self.norm2_emg = (
            nn.LayerNorm(dim)
            if norm == "layernorm"
            else nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(dim), Transpose(1, 2))
        )
        self.mlp_eeg = MLP(
            dim,
            mult=mult,
            dropout=dropout,
            glu=glu,
            relu=relu,
            relu_squared=relu_squared,
        )
        self.mlp_emg = MLP(
            dim,
            mult=mult,
            dropout=dropout,
            glu=glu,
            relu=relu,
            relu_squared=relu_squared,
        )
        self.mlp_mix = None
        if with_mixffn:
            self.mlp_mix = MLP(
                dim,
                mult=mult,
                dropout=dropout,
                glu=glu,
                relu=relu,
                relu_squared=relu_squared,
            )
            self.norm2_mix = (
                nn.LayerNorm(dim)
                if norm == "layernorm"
                else nn.Sequential(
                    Transpose(1, 2), nn.BatchNorm1d(dim), Transpose(1, 2)
                )
            )

        self.gamma_1 = (
            nn.Parameter(
                layer_scale_init_values * torch.ones((dim)), requires_grad=True
            )
            if layer_scale_init_values is not None
            else 1.0
        )
        self.gamma_2 = (
            nn.Parameter(
                layer_scale_init_values * torch.ones((dim)), requires_grad=True
            )
            if layer_scale_init_values is not None
            else 1.0
        )

    def forward(self, x, mask=None, modality_type=None, relative_position_bias=None):
        x = x + self.drop_path(
            self.gamma_1 * self.attn(self.norm1(x), mask, relative_position_bias)
        )

        if modality_type == "eeg":
            x = x + self.drop_path(self.gamma_2 * self.mlp_eeg(self.norm2_eeg(x)))
        elif modality_type == "emg":
            x = x + self.drop_path(self.gamma_2 * self.mlp_emg(self.norm2_emg(x)))
        else:
            if self.mlp_mix is None:
                x_eeg = x[:, : self.n_patches]
                x_emg = x[:, self.n_patches :]
                x_eeg = x_eeg + self.drop_path(
                    self.gamma_2 * self.mlp_eeg(self.norm2_eeg(x_eeg))
                )
                x_emg = x_emg + self.drop_path(
                    self.gamma_2 * self.mlp_emg(self.norm2_emg(x_emg))
                )
                x = torch.cat([x_eeg, x_emg], dim=-2)
            else:
                x = x + self.drop_path(self.gamma_2 * self.mlp_mix(self.norm2_mix(x)))

        return x


class MLP(nn.Module):
    def __init__(
        self,
        dim,
        dim_out=None,
        mult=4,
        glu=False,
        swish=False,
        relu_squared=False,
        relu=False,
        post_act_ln=False,
        dropout=0.0,
        no_bias=False,
        zero_init_output=False,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)

        if relu_squared:
            activation = ReluSquared()
        if relu:
            activation = nn.ReLU()
        elif swish:
            activation = nn.SiLU()
        else:
            activation = nn.GELU()

        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim, bias=not no_bias),
            activation,
        )
        self.ff = nn.Sequential(
            project_in,
            nn.LayerNorm(inner_dim) if post_act_ln else nn.Identity(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out, bias=not no_bias),
            nn.Dropout(dropout),
        )

        # init last linear layer to 0
        if zero_init_output:
            init_zero_(self.ff[-1])

    def forward(self, x):
        return self.ff(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        output_attentions=False,
    ):
        super().__init__()
        self.output_attentions = output_attentions
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None, relative_position_bias=None):
        B, N, C = x.shape

        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (
                    self.q_bias,
                    torch.zeros_like(self.v_bias, requires_grad=False),
                    self.v_bias,
                )
            )
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)

        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = q.float() @ k.float().transpose(-2, -1)

        if relative_position_bias is not None:
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            mask = mask.bool()
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
        attn = attn.softmax(dim=-1).type_as(x)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        # if self.output_attentions:
        #     return x, attn
        return x

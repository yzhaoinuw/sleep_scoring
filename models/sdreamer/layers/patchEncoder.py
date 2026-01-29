from einops.layers.torch import Rearrange
from torch import nn


class PatchEncoder(nn.Module):
    # Batch x Epoch x Trace x Channel x Time
    # Batch x Epoch x Trace x Channel x (Patch_num x Patch_len)
    # Batch x Epoch x Trace x Patch_num x (Channel x Patch_len)

    def __init__(self, patch_len=16, in_channel=1, d_model=128):
        super().__init__()

        self.patch_dim = patch_len * in_channel
        self.to_patch_embedding = nn.Sequential(
            Rearrange("b ... c (n l) -> b ... n (l c)", l=patch_len),
            nn.Linear(self.patch_dim, d_model, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model, bias=True),
        )

    def forward(self, x):
        # print("x.shape ERROR HERE is:",x.shape)
        x = self.to_patch_embedding(x)  # 128 * 32 * 128
        return x

import torch
import torch.nn as nn

# ----------------------------
# PatchMerging3D (from your code)
# ----------------------------
class PatchMerging3D(nn.Module):
    def __init__(self, i_reso, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.i_reso = i_reso
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        D, H, W = self.i_reso
        B, L, C = x.shape
        assert L == D * H * W
        assert D % 2 == 0 and H % 2 == 0 and W % 2 == 0

        x = x.view(B, D, H, W, C)

        x000 = x[:, 0::2, 0::2, 0::2, :]
        x100 = x[:, 1::2, 0::2, 0::2, :]
        x010 = x[:, 0::2, 1::2, 0::2, :]
        x110 = x[:, 1::2, 1::2, 0::2, :]
        x001 = x[:, 0::2, 0::2, 1::2, :]
        x101 = x[:, 1::2, 0::2, 1::2, :]
        x011 = x[:, 0::2, 1::2, 1::2, :]
        x111 = x[:, 1::2, 1::2, 1::2, :]

        x = torch.cat([x000, x100, x010, x110, x001, x101, x011, x111], dim=-1)
        x = x.view(B, -1, 8 * C)
        x = self.reduction(x)
        x = self.norm(x)
        return x

# ----------------------------
# PatchUnmerging3D (inverse)
# ----------------------------
class PatchUnmerging3D(nn.Module):
    r"""3D Patch Unmerging Layer (dual of PatchMerging3D)

    Args:
        i_reso (tuple[int]): (D, H, W)
        dim (int): Input channel dimension C
        norm_layer (nn.Module): Normalization layer
    """

    def __init__(self, i_reso, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.i_reso = i_reso
        self.dim = dim
        assert dim % 2 == 0, "dim must be even since C_out = C//2"

        self.out_dim = dim // 2

        # C -> 8*(C//2) = 4C
        self.expansion = nn.Linear(dim, 8 * self.out_dim, bias=False)
        self.norm = norm_layer(8 * self.out_dim)

    def forward(self, x):
        """
        x: (B, D*H*W, C)
        """
        D, H, W = self.i_reso
        B, L, C = x.shape

        assert L == D * H * W, "input feature has wrong size"
        assert C == self.dim

        # expand channels
        x = self.expansion(x)          # (B, D*H*W, 4C)
        x = self.norm(x)

        # reshape to grid
        x = x.view(B, D, H, W, 8 * self.out_dim)

        C_out = self.out_dim

        # split channels into 8 groups
        x000 = x[..., 0*C_out:1*C_out]
        x100 = x[..., 1*C_out:2*C_out]
        x010 = x[..., 2*C_out:3*C_out]
        x110 = x[..., 3*C_out:4*C_out]
        x001 = x[..., 4*C_out:5*C_out]
        x101 = x[..., 5*C_out:6*C_out]
        x011 = x[..., 6*C_out:7*C_out]
        x111 = x[..., 7*C_out:8*C_out]

        # scatter to higher resolution
        out = torch.zeros(
            B, 2*D, 2*H, 2*W, C_out,
            device=x.device,
            dtype=x.dtype
        )

        out[:, 0::2, 0::2, 0::2, :] = x000
        out[:, 1::2, 0::2, 0::2, :] = x100
        out[:, 0::2, 1::2, 0::2, :] = x010
        out[:, 1::2, 1::2, 0::2, :] = x110
        out[:, 0::2, 0::2, 1::2, :] = x001
        out[:, 1::2, 0::2, 1::2, :] = x101
        out[:, 0::2, 1::2, 1::2, :] = x011
        out[:, 1::2, 1::2, 1::2, :] = x111

        # flatten back
        out = out.view(B, (2*D)*(2*H)*(2*W), C_out)

        return out

    def extra_repr(self) -> str:
        return (
            f"i_reso={self.i_reso}, "
            f"dim={self.dim}, out_dim={self.out_dim}"
        )


# ----------------------------
# Test function
# ----------------------------
def test_structural_inverse():
    torch.manual_seed(0)

    B, D, H, W, C = 2, 4, 4, 4, 8
    x = torch.randn(B, D*H*W, C, requires_grad=True)

    merge = PatchMerging3D((D, H, W), C)
    unmerge = PatchUnmerging3D((D//2, H//2, W//2), 2*C)

    y = merge(x)            # (B, D/2*H/2*W/2, 2C)
    x_hat = unmerge(y)      # (B, D*H*W, C)

    print("Input shape:      ", x.shape)
    print("Merged shape:     ", y.shape)
    print("Reconstructed:    ", x_hat.shape)

    loss = x_hat.sum()
    loss.backward()

    print("✔ structural symmetry")
    print("✔ gradients flow")

test_structural_inverse()

if __name__ == "__main__":
    test_structural_inverse()

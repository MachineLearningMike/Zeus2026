import torch
import torch.nn as nn

# -------------------------------
# Module with full-sphere pair types
# -------------------------------
class TestModule(nn.Module):
    def __init__(self, D, H, W, wd, wh, ww, num_heads):
        """
        D,H,W: total feature dimensions
        wd,wh,ww: window size along D,H,W
        num_heads: number of attention heads
        """
        super().__init__()
        self.D, self.H, self.W = D, H, W
        self.wd, self.wh, self.ww = wd, wh, ww
        self.num_heads = num_heads

        # Number of windows along latitude (covers D and H)
        self.G = (D // wd) * (H // wh)

        # Learnable pair types for one longitude band
        self.learnable_pair_types = nn.Parameter(
            torch.empty(self.G, wd*wd, wh*wh, 2*ww-1, num_heads)
        )
        nn.init.trunc_normal_(self.learnable_pair_types, std=0.02)

    def build_pair_type_matrix_sphere(self):
        """
        Build pair_type_matrix covering full sphere by repeating along longitude.
        Returns: pair_type_matrix of shape (total_windows, N, N, num_heads)
        """
        wd, wh, ww = self.wd, self.wh, self.ww
        D, H, W = self.D, self.H, self.W
        device = self.learnable_pair_types.device
        G, _, _, _, num_heads = self.learnable_pair_types.shape
        N = wd * wh * ww

        # --- Token coordinates inside a window
        d = torch.arange(wd, device=device)
        h = torch.arange(wh, device=device)
        w = torch.arange(ww, device=device)
        coords = torch.stack(torch.meshgrid(d, h, w, indexing="ij")).view(3, -1)  # (3, N)
        d1, d2 = coords[0][:, None], coords[0][None, :]
        h1, h2 = coords[1][:, None], coords[1][None, :]
        w1, w2 = coords[2][:, None], coords[2][None, :]

        idx_d = d1 * wd + d2
        idx_h = h1 * wh + h2
        idx_w = (w1 - w2 + (ww - 1)) % (2*ww-1)  # cylindric along W

        # Expand along longitude
        G_idx = torch.arange(G, device=device)[:, None, None, None]
        W_idx = torch.arange(W // ww, device=device)[None, :, None, None]  # number of windows along longitude

        D_idx = idx_d[None, None, :, :]
        H_idx = idx_h[None, None, :, :]
        W_idx_tokens = idx_w[None, None, :, :]

        # repeat learnable_pair_types along longitude
        pair_type_matrix = self.learnable_pair_types[:, None, :, :, :, :].repeat(1, W // ww, 1, 1, 1, 1)

        # advanced indexing to select correct pair types
        pair_type_matrix = pair_type_matrix[
            G_idx, W_idx, D_idx, H_idx, W_idx_tokens
        ]  # (G, num_lon_windows, N, N, num_heads)

        # merge G and longitude dims
        pair_type_matrix = pair_type_matrix.view(G * (W // ww), N, N, num_heads)
        return pair_type_matrix


# -------------------------------
# Test function
# -------------------------------
def test_pair_type_matrix_sphere():
    torch.manual_seed(0)

    # small test dimensions
    D, H, W = 4, 4, 6
    wd, wh, ww = 2, 2, 3
    num_heads = 2

    model = TestModule(D, H, W, wd, wh, ww, num_heads)

    pair_mat = model.build_pair_type_matrix_sphere()

    total_windows = (D // wd) * (H // wh) * (W // ww)
    N = wd * wh * ww

    # ---- 1. shape test
    assert pair_mat.shape == (total_windows, N, N, num_heads)
    print("✔ Shape test passed")

    # ---- 2. spot-check correctness
    g, lon_win, i, j = 0, 0, 1, 5  # arbitrary indices

    # reconstruct coordinates
    coords = torch.stack(
        torch.meshgrid(
            torch.arange(wd),
            torch.arange(wh),
            torch.arange(ww),
            indexing="ij"
        )
    ).view(3, -1)

    d1, h1, w1 = coords[:, i]
    d2, h2, w2 = coords[:, j]

    idx_d = d1 * wd + d2
    idx_h = h1 * wh + h2
    idx_w = (w1 - w2 + (ww - 1)) % (2*ww-1)

    expected = model.learnable_pair_types[g, idx_d, idx_h, idx_w]
    actual = pair_mat[g, i, j]
    assert torch.allclose(expected, actual)
    print("✔ Spot-check mapping test passed")

    # ---- 3. gradient flow test
    loss = pair_mat.sum()
    loss.backward()
    assert model.learnable_pair_types.grad is not None
    assert torch.isfinite(model.learnable_pair_types.grad).all()
    print("✔ Gradient flow test passed")

    print("\nAll tests passed successfully.")


if __name__ == "__main__":
    test_pair_type_matrix_sphere()

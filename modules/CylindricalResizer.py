import torch
import torch.nn.functional as F

class CylindricalResizer:
    def resize_cylindrical(self, x, size_hw):
        """
        Resize cylindrical images along width with circular interpolation.
        x: (..., frames, C, H, W)
        size_hw: (H1, W1)
        """
        *leading, Fm, C, H, W = x.shape
        outs = []

        # flatten leading dims
        x_flat = x.reshape(-1, Fm, C, H, W)

        for f in range(Fm):
            frame = x_flat[:, f]  # (N, C, H, W)

            # circular padding along width
            frame = F.pad(frame, (1, 1, 0, 0), mode="circular")  # (N, C, H, W+2)

            # interpolate each frame
            frame_resized = F.interpolate(
                frame, size=(size_hw[0], size_hw[1] + 2),
                mode="bicubic", align_corners=False
            )

            # remove circular padding
            frame_resized = frame_resized[:, :, :, 1:-1]  # (N, C, H1, W1)
            outs.append(frame_resized)

        # stack frames back
        output = torch.stack(outs, dim=1)  # (N, Fm, C, H1, W1)

        # reshape back to original leading dims
        output = output.reshape(*leading, Fm, C, size_hw[0], size_hw[1])
        return output
    
# --- Demo ---
if __name__ == "__main__":
    resizer = CylindricalResizer()

    # Example 1: no leading dims
    x1 = torch.randn(2, 3, 4, 8, 16)  # (B=2, frames=3, C=4, H=8, W=16)
    y1 = resizer.resize_cylindrical(x1, (12, 24))
    print("Example 1:", x1.shape, "->", y1.shape)

    # Example 2: one leading dim
    x2 = torch.randn(5, 2, 3, 4, 8, 16)  # (T=5, B=2, frames=3, C=4, H=8, W=16)
    y2 = resizer.resize_cylindrical(x2, (12, 24))
    print("Example 2:", x2.shape, "->", y2.shape)

    # Example 3: multiple leading dims
    x3 = torch.randn(2, 2, 2, 3, 4, 8, 16)  # (..., frames=3, C=4, H=8, W=16)
    y3 = resizer.resize_cylindrical(x3, (10, 20))
    print("Example 3:", x3.shape, "->", y3.shape)
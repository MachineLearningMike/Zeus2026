import torch

class ShiftManager:
    """
    Manages shifting along level, latitude, and longitude dimensions with appropriate padding.
    Supports forward shift and inverse shift.
    
    Tensor shape convention: (C, L, H, W)
        L: vertical levels (0 = topmost)
        H: latitude (0 = north pole)
        W: longitude
    """
    def __init__(self, pad_mode='copy'):
        """
        pad_mode: 'copy' or 'zero'
        """
        assert pad_mode in ['copy', 'zero']
        self.pad_mode = pad_mode

    def shift_levels(self, x, shift=1, surface_more_important=True):
        # surface level = max, uppermost level = 0
        C, L, H, W = x.shape
        assert 0 < shift < L
        if surface_more_important:
            remaining = x[:, shift:, :, :]                  # discard upper levels.
            if self.pad_mode == 'copy':
                pad = x[:, -shift:, :, :]                   # pad with lower levels.
            else:
                pad = torch.zeros(C, shift, H, W, device=x.device, dtype=x.dtype)
            shifted_x = torch.cat([remaining, pad], dim=1)  # pad to lower side
        else:
            remaining = x[:, :-shift, :, :]                 # discard lower levels.
            if self.pad_mode == 'copy':
                pad = x[:, :shift, :, :]                    # pad with upper levels. 
            else:
                pad = torch.zeros(C, shift, H, W, device=x.device, dtype=x.dtype)
            shifted_x = torch.cat([pad, remaining], dim=1)  # pad to upper side.
        return shifted_x
    
    def shift_levels_inverse(self, x, shift=1, surface_more_important=True):
        # surface level = max, uppermost level = 0
        C, L, H, W = x.shape
        assert 0 < shift < L
        if surface_more_important:       
            remaining = x[:, :-shift, :, :]                 # discard lower levels.
            if self.pad_mode == 'copy':
                pad = x[:, :shift, :, :]                    # pad with upper levels.
            else:
                pad = torch.zeros(C, shift, H, W, device=x.device, dtype=x.dtype)
            x_restored = torch.cat([pad, remaining], dim=1) # pad to upper side.
        else:
            remaining = x[:, shift:, :, :]                 # discard upper levels.
            if self.pad_mode == 'copy':
                pad = x[:, -shift:, :, :]                    # pad with lower levels.
            else:
                pad = torch.zeros(C, shift, H, W, device=x.device, dtype=x.dtype)
            x_restored = torch.cat([remaining, pad], dim=1) # pad to lower side.
        return x_restored


    def shift_latitude(self, x, shift=1, north_more_important=True):
        C, L, H, W = x.shape
        assert 0 < shift < H
        if north_more_important:
            remaining = x[:, :, shift:, :]                  # discard southern part
            pad = x[:, :, -shift:, :] if self.pad_mode == 'copy' else torch.zeros(C, L, shift, W, device=x.device, dtype=x.dtype)
            return torch.cat([remaining, pad], dim=2)       # pad northern side with norther part
        else:
            remaining = x[:, :, :-shift, :]                 # discard northern part
            pad = x[:, :, :shift, :].expand(C, L, shift, W) if self.pad_mode == 'copy' else torch.zeros(C, L, shift, W, device=x.device, dtype=x.dtype)
            return torch.cat([pad, remaining], dim=2)       # pad southern side with southern part

    def shift_latitude_inverse(self, x, shift=1, north_more_important=True):
        # simply reverse the above
        C, L, H, W = x.shape
        assert 0 < shift < H
        if north_more_important:
            remaining = x[:, :, :-shift, :]                 # discard northern part
            pad = x[:, :, :shift, :] if self.pad_mode == 'copy' else torch.zeros(C, L, shift, W, device=x.device, dtype=x.dtype)
            return torch.cat([pad, remaining], dim=2)       # pad southern side with southern part
        else:
            remaining = x[:, :, shift:, :]                  # discard southern part
            pad = x[:, :, -shift:, :] if self.pad_mode == 'copy' else torch.zeros(C, L, shift, W, device=x.device, dtype=x.dtype)
            return torch.cat([remaining, pad], dim=2)       # pad northern side with northern part

    def shift_longitude(self, x, shift=1, east_more_important=True):
        C, L, H, W = x.shape
        assert 0 < shift < W
        if east_more_important:
            return torch.roll(x, shifts=-shift, dims=3)
        else:
            return torch.roll(x, shifts=shift, dims=3)

    def shift_longitude_inverse(self, x, shift=1, east_less_important=True):
        # inverse is just rolling in opposite direction
        if east_less_important:
            return torch.roll(x, shifts=shift, dims=3)
        else:
            return torch.roll(x, shifts=-shift, dims=3)


import torch

class ShiftManagerBatch:
    """
    ShiftManager supporting batch dimension.
    Tensor shape: (B, C, L, H, W)
    """
    def __init__(self, pad_mode='copy'):
        assert pad_mode in ['copy', 'zero']
        self.pad_mode = pad_mode

    def shift_levels_up(self, x, shift=1):
        B, C, L, H, W = x.shape
        assert 0 < shift < L
        remaining = x[:, :, shift:, :, :]
        if self.pad_mode == 'copy':
            pad = x[:, :, -1:, :, :].expand(B, C, shift, H, W)
        else:
            pad = torch.zeros(B, C, shift, H, W, device=x.device, dtype=x.dtype)
        return torch.cat([remaining, pad], dim=2)

    def shift_levels_inverse(self, x, shift=1):
        B, C, L, H, W = x.shape
        remaining = x[:, :, :-shift, :, :]
        if self.pad_mode == 'copy':
            pad = x[:, :, 0:1, :, :].expand(B, C, shift, H, W)
        else:
            pad = torch.zeros(B, C, shift, H, W, device=x.device, dtype=x.dtype)
        return torch.cat([pad, remaining], dim=2)

    def shift_latitude(self, x, shift=1, north_less_important=True):
        B, C, L, H, W = x.shape
        assert 0 < shift < H
        if north_less_important:
            remaining = x[:, :, :, shift:, :]
            pad = x[:, :, :, -1:, :].expand(B, C, L, shift, W) if self.pad_mode == 'copy' else torch.zeros(B, C, L, shift, W, device=x.device, dtype=x.dtype)
            return torch.cat([remaining, pad], dim=3)
        else:
            remaining = x[:, :, :, :-shift, :]
            pad = x[:, :, :, 0:1, :].expand(B, C, L, shift, W) if self.pad_mode == 'copy' else torch.zeros(B, C, L, shift, W, device=x.device, dtype=x.dtype)
            return torch.cat([pad, remaining], dim=3)

    def shift_latitude_inverse(self, x, shift=1, north_less_important=True):
        B, C, L, H, W = x.shape
        if north_less_important:
            remaining = x[:, :, :, :-shift, :]
            pad = x[:, :, :, 0:1, :].expand(B, C, L, shift, W) if self.pad_mode == 'copy' else torch.zeros(B, C, L, shift, W, device=x.device, dtype=x.dtype)
            return torch.cat([pad, remaining], dim=3)
        else:
            remaining = x[:, :, :, shift:, :]
            pad = x[:, :, :, -1:, :].expand(B, C, L, shift, W) if self.pad_mode == 'copy' else torch.zeros(B, C, L, shift, W, device=x.device, dtype=x.dtype)
            return torch.cat([remaining, pad], dim=3)

    def shift_longitude(self, x, shift=1, east_less_important=True):
        if east_less_important:
            return torch.roll(x, shifts=-shift, dims=4)
        else:
            return torch.roll(x, shifts=shift, dims=4)

    def shift_longitude_inverse(self, x, shift=1, east_less_important=True):
        if east_less_important:
            return torch.roll(x, shifts=shift, dims=4)
        else:
            return torch.roll(x, shifts=-shift, dims=4)

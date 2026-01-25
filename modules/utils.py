import os
import torch
import random
import numpy as np
from pathlib import Path
import torch.backends.cudnn as cudnn

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from datetime import timedelta, datetime
import time
import matplotlib.ticker as mticker


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True

def output_namespace(namespace):
    configs = namespace.__dict__
    message = ''
    for k, v in configs.items():
        message += '\n' + k + ': \t' + str(v) + '\t'
    return message

def check_dir(path):
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True)

def check_gradient(model):
    
    for name, param in model.named_parameters():
        if (param.requires_grad is True) and (param.grad is None):
            print(f"Parameter '{name}' did NOT receive a gradient --------------------------------------------")
        else:
            pass
            # print(f"Parameter '{name}' received gradient, shape: {param.grad.shape}, device: {param.grad.device}")

def get_gpu_names():
    gpu_names = []

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {gpu_name}")
            
            if "A100" in gpu_name:
                print("  -> This is an A100")
                gpu_names.append('A100')
                
            elif "B200" in gpu_name:
                print("  -> This is a B200")
                gpu_names.append('B200')
                
    return gpu_names

            # Note: B200 identification string may vary when released
            
def best_segments_increasing(N, S):
    """
    Return actual segments of numbers 0..N-1
    based on non-decreasing lengths maximizing trailing uniformity.
    Mirrors the logic of best_segments_decreasing.
    """
    # Step 1: compute segment lengths
    def best_segment_lengths_increasing(N, S):
        best_seq = None
        best_trailing = 0

        def dfs(seq, remaining, remaining_count):
            nonlocal best_seq, best_trailing
            if remaining_count == 0:
                if remaining != 0:
                    return
                # trailing uniform length
                last_val = seq[-1]
                count = 0
                for x in reversed(seq):
                    if x == last_val:
                        count += 1
                    else:
                        break
                if count > best_trailing:
                    best_trailing = count
                    best_seq = seq[:]
                return

            # Mirror logic: next value ≥ previous (or floor of even distribution if seq empty)
            start = seq[-1] if seq else max(1, remaining // remaining_count)
            max_val = remaining - (remaining_count - 1)  # leave at least 1 for remaining segments
            for val in range(start, max_val + 1):
                dfs(seq + [val], remaining - val, remaining_count - 1)

        dfs([], N, S)
        return best_seq

    lengths = best_segment_lengths_increasing(N, S)

    # Step 2: build actual segments
    segments = []
    start = 0
    for length in lengths:
        segments.append(list(range(start, start + length)))
        start += length
    return segments

import math

def best_segments_decreasing(N, S):
    """
    Return actual segments of numbers 0..N-1
    based on non-increasing lengths maximizing trailing uniformity.
    """
    # Step 1: compute segment lengths
    def best_segment_lengths_decreasing(N, S):
        best_seq = None
        best_trailing = 0

        def dfs(seq, remaining, remaining_count):
            nonlocal best_seq, best_trailing
            if remaining_count == 0:
                if remaining != 0:
                    return
                # trailing uniform length
                last_val = seq[-1]
                count = 0
                for x in reversed(seq):
                    if x == last_val:
                        count += 1
                    else:
                        break
                if count > best_trailing:
                    best_trailing = count
                    best_seq = seq[:]
                return
            # next value ≤ previous (or ceiling of even distribution if seq empty)
            start = seq[-1] if seq else math.ceil(remaining / remaining_count)
            min_val = 1
            for val in range(start, min_val-1, -1):
                if val * remaining_count < remaining:
                    continue
                dfs(seq + [val], remaining - val, remaining_count - 1)

        dfs([], N, S)
        return best_seq

    lengths = best_segment_lengths_decreasing(N, S)

    # Step 2: build actual segments
    segments = []
    start = 0
    for length in lengths:
        segments.append(list(range(start, start + length)))
        start += length
    return segments

def check_consistency(n, s):
    inc = best_segments_increasing(n, s)
    dec = best_segments_decreasing(n, s)
    def assert_full(A):
        sum = []
        for a in A:
            sum = sum + a
        assert len(A) == s
        assert len(sum) == n
        assert set(sum) == set(range(n))
    assert_full(inc)
    assert_full(dec)
    inc_l = [len(a) for a in inc]
    dec_l = [len(a) for a in dec]
    print(inc_l)
    print(dec_l)
    assert inc_l == dec_l[::-1]
    print(inc)
    print(dec)

#---------------------------- show array -------------------------

def initialize_for_show_array():
    # Turn on interactive mode ONCE
    plt.ion()    

# Define global variables at the beginning of the function
_fig = None
_ax = None
_im = None
_cbar = None

def finalize_for_show_array():
    plt.ioff()
    if _fig is not None:
        plt.close(_fig)

def show_array(arr, variable="", dt=datetime(2000,1,1,0), modulo=6):
    global _fig, _ax, _im, _cbar
    
    # Create figure if it doesn't exist
    if _fig is None:
        print("Creating new figure...")
        _fig = plt.figure(figsize=(10, 7))
        _ax = plt.axes(projection=ccrs.PlateCarree())
        
        # Calculate the correct extent for your data
        # Your data: (721, 1440) = (lat, lon)
        # Latitude: -90 to +90 with 721 points
        # Longitude: -180 to +180 with 1440 points
        
        # Calculate step sizes
        lat_step = 180.0 / (arr.shape[0] - 1)  # From -90 to +90
        lon_step = 360.0 / (arr.shape[1] - 1)  # From -180 to +180
        
        # Calculate extent
        # imshow expects: [left, right, bottom, top]
        left = -180.0
        right = 180.0
        bottom = -90.0
        top = 90.0
        
        # Plot with correct extent
        _im = _ax.imshow(arr,
                        transform=ccrs.PlateCarree(),
                        extent=[left, right, bottom, top],
                        cmap='viridis',
                        origin='lower')
        
        # Add boundaries
        _ax.coastlines(resolution='110m', linewidth=0.5)
        gl = _ax.gridlines(draw_labels=True, linewidth=0.7, color='gray', alpha=0.5, linestyle='--')
        gl.xlocator = mticker.MultipleLocator(45)
        gl.ylocator = mticker.MultipleLocator(45)
        
        # Add colorbar
        _cbar = plt.colorbar(_im, ax=_ax)
        _cbar.set_label(variable)
        
        # Set title
        # _ax.set_title(f'{variable} | {dt.strftime("%Y-%m-%d %H")}, mod: {modulo}')
        _ax.set_title(
            f'{variable}, {dt.strftime("%Y-%m-%d %H")}, mod: {modulo}',
            fontsize=16,
            fontweight='bold',
            color='#00008B',
            fontfamily='DejaVu Sans'  # or 'Arial', 'Helvetica', etc.
        )
        
        # Show window (non-blocking)
        plt.show(block=False)
        plt.pause(0.5)
    
    else:
        # Update existing plot
        _im.set_data(arr)
        _im.set_clim(vmin=np.nanmin(arr), vmax=np.nanmax(arr))
        
        # Update labels
        # _ax.set_title(f'{variable} | {dt.strftime("%Y-%m-%d %H")}, mod: {modulo}')
        _ax.set_title(
            f'{variable}, {dt.strftime("%Y-%m-%d %H")}, mod: {modulo}',
            fontsize=16,
            fontweight='bold',
            color='#00008B',
            fontfamily='DejaVu Sans'  # or 'Arial', 'Helvetica', etc.
        )
        
        _cbar.set_label(variable)
        
        # Force redraw
        _fig.canvas.draw_idle()
        _fig.canvas.flush_events()
    
    # Small pause
    plt.pause(0.001)
    
def show_array_single(arr, variable="", dt=datetime(2000,1,1,0), modulo=6):
    _fig = plt.figure(figsize=(10, 7))
    _ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Calculate the correct extent for your data
    # Your data: (721, 1440) = (lat, lon)
    # Latitude: -90 to +90 with 721 points
    # Longitude: -180 to +180 with 1440 points
    
    # Calculate step sizes
    lat_step = 180.0 / (arr.shape[0] - 1)  # From -90 to +90
    lon_step = 360.0 / (arr.shape[1] - 1)  # From -180 to +180
    
    # Calculate extent
    # imshow expects: [left, right, bottom, top]
    left = -180.0
    right = 180.0
    bottom = -90.0
    top = 90.0
    
    # Plot with correct extent
    _im = _ax.imshow(arr,
                    transform=ccrs.PlateCarree(),
                    extent=[left, right, bottom, top],
                    cmap='viridis',
                    origin='lower')
    
    # Add boundaries
    _ax.coastlines(resolution='110m', linewidth=0.5)
    gl = _ax.gridlines(draw_labels=True, linewidth=0.7, color='gray', alpha=0.5, linestyle='--')
    gl.xlocator = mticker.MultipleLocator(45)
    gl.ylocator = mticker.MultipleLocator(45)
    
    # Add colorbar
    _cbar = plt.colorbar(_im, ax=_ax)
    _cbar.set_label(variable)
    
    # Set title
    # _ax.set_title(f'{variable} | {dt.strftime("%Y-%m-%d %H")}, mod: {modulo}')
    _ax.set_title(
        f'{variable}, {dt.strftime("%Y-%m-%d %H")}, mod: {modulo}',
        fontsize=16,
        fontweight='bold',
        color='#00008B',
        fontfamily='DejaVu Sans'  # or 'Arial', 'Helvetica', etc.
    )
    
    # Show window (non-blocking)
    plt.show()

    if _fig is not None:
        plt.close(_fig)
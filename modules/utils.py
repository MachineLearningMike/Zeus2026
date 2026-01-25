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

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
import numpy as np
from datetime import datetime

def initialize_for_show_array():
    """Turn on interactive mode and create figure with two subplots."""
    plt.ion()
    
    # Define global variables
    global _fig, _ax1, _ax2, _im1, _im2, _cbar1, _cbar2
    _fig = None
    _ax1 = None
    _ax2 = None
    _im1 = None
    _im2 = None
    _cbar1 = None
    _cbar2 = None

def finalize_for_show_array():
    """Turn off interactive mode and close figure."""
    plt.ioff()
    if _fig is not None:
        plt.close(_fig)

def show_array(arr1, arr2=None, variable1="", variable2="", dt=datetime(2000,1,1,0), modulo=6):
    """
    Display one or two arrays side-by-side with shared colorbar for two-panel plots.
    Automatically handles string-to-float conversion.
    """
    global _fig, _ax1, _ax2, _im1, _im2, _cbar_shared
    
    arr1_plot = arr1
    arr2_plot = arr2
    
    # Determine if showing one or two arrays
    show_two = arr2_plot is not None
    
    # Calculate color ranges
    if show_two:
        # Shared color scale for both arrays
        vmin = min(np.nanmin(arr1_plot), np.nanmin(arr2_plot))
        vmax = max(np.nanmax(arr1_plot), np.nanmax(arr2_plot))
    else:
        # Single array color scale
        vmin, vmax = np.nanmin(arr1_plot), np.nanmax(arr1_plot)
    
    # Create figure if it doesn't exist
    if _fig is None:
        # Create figure with appropriate layout
        if show_two:
            # Create figure with GridSpec for shared colorbar
            _fig = plt.figure(figsize=(16, 7))
            gs = plt.GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.1)
            _ax1 = _fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
            _ax2 = _fig.add_subplot(gs[1], projection=ccrs.PlateCarree())
            cbar_ax = _fig.add_subplot(gs[2])
        else:
            _fig, _ax1 = plt.subplots(1, 1, figsize=(10, 7),
                                     subplot_kw={'projection': ccrs.PlateCarree()})
            _ax2 = None
            cbar_ax = None
        
        # Set extent for your data (ERA5 grid)
        left, right, bottom, top = -180.0, 180.0, -90.0, 90.0
        
        # --- Plot First Array ---
        if show_two:
            _im1 = _ax1.imshow(arr1_plot,
                              transform=ccrs.PlateCarree(),
                              extent=[left, right, bottom, top],
                              cmap='viridis',
                              origin='lower',
                              vmin=vmin, vmax=vmax)
        else:
            _im1 = _ax1.imshow(arr1_plot,
                              transform=ccrs.PlateCarree(),
                              extent=[left, right, bottom, top],
                              cmap='viridis',
                              origin='lower')
        
        # Configure first subplot
        _ax1.coastlines(resolution='110m', linewidth=0.5, color='black', alpha=0.7)
        gl1 = _ax1.gridlines(draw_labels=True, linewidth=0.7, color='gray', 
                           alpha=0.5, linestyle='--')
        gl1.xlocator = mticker.MultipleLocator(45)
        gl1.ylocator = mticker.MultipleLocator(45)
        gl1.top_labels = False
        gl1.right_labels = False
        
        # --- Plot Second Array if provided ---
        if show_two:
            _im2 = _ax2.imshow(arr2_plot,
                              transform=ccrs.PlateCarree(),
                              extent=[left, right, bottom, top],
                              cmap='viridis',
                              origin='lower',
                              vmin=vmin, vmax=vmax)
            
            # Configure second subplot
            _ax2.coastlines(resolution='110m', linewidth=0.5, color='black', alpha=0.7)
            gl2 = _ax2.gridlines(draw_labels=True, linewidth=0.7, color='gray',
                               alpha=0.5, linestyle='--')
            gl2.xlocator = mticker.MultipleLocator(45)
            gl2.ylocator = mticker.MultipleLocator(45)
            gl2.top_labels = False
            gl2.right_labels = False
            gl2.left_labels = False  # Only show labels on left plot
            
            # Add shared colorbar for two-panel plot
            _cbar_shared = plt.colorbar(_im1, cax=cbar_ax, orientation='vertical')
            _cbar_shared.set_label(f'{variable1} / {variable2}', fontsize=12)
            
            # Set title for two-panel figure
            _fig.suptitle(f'{variable1} vs {variable2} | {dt.strftime("%Y-%m-%d %H:%M")} UTC | Mod: {modulo}',
                         fontsize=14, fontweight='bold', color='#00008B', y=0.98)
        else:
            # Add colorbar for single plot
            _cbar_shared = plt.colorbar(_im1, ax=_ax1, orientation='vertical', pad=0.03, shrink=0.9)
            _cbar_shared.set_label(variable1, fontsize=12)
            
            # Set title for single plot
            _ax1.set_title(f'{variable1} | {dt.strftime("%Y-%m-%d %H:%M")} UTC | Mod: {modulo}',
                          fontsize=14, fontweight='bold', color='#00008B', pad=20)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 0.95, 0.96] if show_two else None)
        
        # Show window (non-blocking)
        plt.show(block=False)
        plt.pause(0.001)
    
    else:
        # --- Update Existing Plot ---
        
        # Check if plot structure needs to change (single vs dual)
        current_show_two = _ax2 is not None and _im2 is not None
        if show_two != current_show_two:
            # Structure changed, recreate figure
            plt.close(_fig)
            _fig = None
            _ax1 = _ax2 = None
            _im1 = _im2 = None
            _cbar_shared = None
            # Recursive call with new figure
            show_array(arr1_plot, arr2_plot, variable1, variable2, dt, modulo)
            return
        
        # Update first array
        _im1.set_data(arr1_plot)
        
        # Update color limits
        if show_two:
            # Update shared color range
            vmin = min(np.nanmin(arr1_plot), np.nanmin(arr2_plot))
            vmax = max(np.nanmax(arr1_plot), np.nanmax(arr2_plot))
            _im1.set_clim(vmin, vmax)
            
            # Update second array
            if _im2 is not None:
                _im2.set_data(arr2_plot)
                _im2.set_clim(vmin, vmax)
            
            # Update shared colorbar
            if _cbar_shared is not None:
                _cbar_shared.set_label(f'{variable1} / {variable2}', fontsize=12)
                _cbar_shared.update_normal(_im1)
            
            # Update suptitle for two-panel figure
            if hasattr(_fig, 'suptitle'):
                _fig.suptitle(f'{variable1} vs {variable2} | {dt.strftime("%Y-%m-%d %H:%M")} UTC | Mod: {modulo}',
                             fontsize=14, fontweight='bold', color='#00008B', y=0.98)
        else:
            # Update single array color range
            vmin, vmax = np.nanmin(arr1_plot), np.nanmax(arr1_plot)
            _im1.set_clim(vmin, vmax)
            
            # Update colorbar
            if _cbar_shared is not None:
                _cbar_shared.set_label(variable1, fontsize=12)
                _cbar_shared.update_normal(_im1)
            
            # Update title for single plot
            if _ax1 is not None:
                _ax1.set_title(f'{variable1} | {dt.strftime("%Y-%m-%d %H:%M")} UTC | Mod: {modulo}',
                              fontsize=14, fontweight='bold', color='#00008B', pad=20)
        
        # Force redraw
        _fig.canvas.draw_idle()
        _fig.canvas.flush_events()
    
    # Small pause for UI responsiveness
    plt.pause(0.001)


def show_array_single(arr1, arr2=None, variable1="", variable2="", dt=datetime(2000,1,1,0), modulo=6):
    """
    Show one or two arrays in a standalone figure.
    """
    # Prepare arrays (using the same function as above)
    def prepare_array(arr):
        if arr is None:
            return None
        if arr.dtype.kind in ['i', 'f', 'c', 'b']:
            return arr.astype(np.float32)
        if arr.dtype.kind in ['U', 'O', 'S']:
            try:
                return arr.astype(np.float32)
            except:
                arr_float = np.zeros_like(arr, dtype=np.float32)
                for i in range(arr.shape[0]):
                    for j in range(arr.shape[1]):
                        try:
                            arr_float[i, j] = float(arr[i, j])
                        except:
                            arr_float[i, j] = np.nan
                return np.nan_to_num(arr_float, nan=0.0)
        return np.array(arr, dtype=np.float32)
    
    arr1_plot = prepare_array(arr1)
    arr2_plot = prepare_array(arr2) if arr2 is not None else None
    
    show_two = arr2_plot is not None
    
    # Create figure
    if show_two:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7),
                                      subplot_kw={'projection': ccrs.PlateCarree()})
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 7),
                               subplot_kw={'projection': ccrs.PlateCarree()})
        ax2 = None
    
    # Set extent
    left, right, bottom, top = -180.0, 180.0, -90.0, 90.0
    
    # Plot first array
    im1 = ax1.imshow(arr1_plot,
                    transform=ccrs.PlateCarree(),
                    extent=[left, right, bottom, top],
                    cmap='viridis',
                    origin='lower')
    
    # Configure first plot
    ax1.coastlines(resolution='110m', linewidth=0.5, color='black', alpha=0.7)
    gl1 = ax1.gridlines(draw_labels=True, linewidth=0.7, color='gray',
                       alpha=0.5, linestyle='--')
    gl1.xlocator = mticker.MultipleLocator(45)
    gl1.ylocator = mticker.MultipleLocator(45)
    gl1.top_labels = False
    gl1.right_labels = False
    
    # Colorbar for first plot
    cbar1 = plt.colorbar(im1, ax=ax1, orientation='vertical', pad=0.03, shrink=0.9)
    cbar1.set_label(variable1, fontsize=12)
    
    # Plot second array if provided
    if show_two:
        im2 = ax2.imshow(arr2_plot,
                        transform=ccrs.PlateCarree(),
                        extent=[left, right, bottom, top],
                        cmap='viridis',
                        origin='lower')
        
        # Configure second plot
        ax2.coastlines(resolution='110m', linewidth=0.5, color='black', alpha=0.7)
        gl2 = ax2.gridlines(draw_labels=True, linewidth=0.7, color='gray',
                           alpha=0.5, linestyle='--')
        gl2.xlocator = mticker.MultipleLocator(45)
        gl2.ylocator = mticker.MultipleLocator(45)
        gl2.top_labels = False
        gl2.right_labels = False
        gl2.left_labels = False
        
        # Colorbar for second plot
        cbar2 = plt.colorbar(im2, ax=ax2, orientation='vertical', pad=0.03, shrink=0.9)
        cbar2.set_label(variable2, fontsize=12)
        
        # Title for two-panel figure
        fig.suptitle(f'{variable1} vs {variable2} | {dt.strftime("%Y-%m-%d %H:%M")} UTC | Mod: {modulo}',
                    fontsize=14, fontweight='bold', color='#00008B', y=0.98)
    else:
        # Title for single plot
        ax1.set_title(f'{variable1} | {dt.strftime("%Y-%m-%d %H:%M")} UTC | Mod: {modulo}',
                     fontsize=14, fontweight='bold', color='#00008B', pad=20)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96] if show_two else None)
    plt.show()
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
    global _fig, _ax1, _ax2, _ax3, _im1, _im2, _im3, _cbar_shared
    _fig = None
    _ax1 = None
    _ax2 = None
    _ax3 = None
    _im1 = None
    _im2 = None
    _im3 = None
    _cbar_shared = None

def finalize_for_show_array():
    """Turn off interactive mode and close figure."""
    plt.ioff()
    if _fig is not None:
        plt.close(_fig)

def show_two_arrays(arr1, arr2=None, variable1="", variable2="", dt=datetime(2000,1,1,0), modulo=6):
    """
    Display one or two arrays side-by-side with shared colorbar for two-panel plots.
    If two arrays are provided, also show their difference in a third panel on the lower row.
    Top panels share a colorbar, bottom difference panel has its own colorbar with same colormap.
    Each panel title includes array number, mean and standard deviation on the same line.
    Mean and std values in the 3rd panel are displayed in dark blue color.
    """
    global _fig, _ax1, _ax2, _ax3, _im1, _im2, _im3, _cbar_top, _cbar_bottom
    
    # Determine if showing one or two arrays
    show_two = arr2 is not None
    show_three = show_two  # Show difference panel if we have two arrays
    
    # Resize arrays if they have different shapes
    if show_two and arr1.shape != arr2.shape:
        # print(f"Resizing arrays: {arr1.shape} -> common shape")
        arr1_resized, arr2_resized = resize_one_to_other_cylindric(arr1[None, :, :], arr2[None, :, :], larger_to_smaller=True)
        arr1_resized, arr2_resized = arr1_resized[0], arr2_resized[0]
        # print(f"Resized to: {arr1_resized.shape}")
        
        # Calculate difference with resized arrays (Array 2 - Array 1)
        diff_plot = arr2_resized - arr1_resized
        
        # Use resized arrays for display
        arr1_plot = arr1_resized
        arr2_plot = arr2_resized
    elif show_two:
        # Arrays already have same shape
        # Calculate Array 2 - Array 1 for consistency with label
        diff_plot = arr2 - arr1
        arr1_plot = arr1
        arr2_plot = arr2
    else:
        # Single array
        arr1_plot = arr1
        arr2_plot = None
        diff_plot = None
    
    # Helper function to create colored titles
    def create_titles(variable1, variable2, arr1_plot, arr2_plot, diff_plot, show_three):
        """Create titles with appropriate formatting for each panel."""
        # Calculate statistics
        mean1 = np.nanmean(arr1_plot)
        std1 = np.nanstd(arr1_plot)
        
        if show_three or show_two:
            mean2 = np.nanmean(arr2_plot)
            std2 = np.nanstd(arr2_plot)
        
        if show_three:
            mean_diff = np.nanmean(diff_plot)
            std_diff = np.nanstd(diff_plot)
        
        # Panel 1 title (normal black)
        title1 = f'{variable1} (μ={mean1:.5f}, σ={std1:.5f})'
        
        # Panel 2 title (normal black)
        title2 = None
        if show_three or show_two:
            title2 = f'{variable2} (μ={mean2:.5f}, σ={std2:.5f})'
        
        # Panel 3 title with colored mean/std values
        title3 = None
        if show_three:
            # Create a title with dark blue mean and std values
            # We'll use matplotlib's Text object with different colors for different parts
            title3_text = f'{variable2} - {variable1} (μ={mean_diff:.5f}, σ={std_diff:.5f})'
            title3 = {
                'text': title3_text,
                'mean_value': f'{mean_diff:.5f}',
                'std_value': f'{std_diff:.5f}',
                'mean_start': title3_text.find(f'{mean_diff:.5f}'),
                'mean_end': title3_text.find(f'{mean_diff:.5f}') + len(f'{mean_diff:.5f}'),
                'std_start': title3_text.find(f'{std_diff:.5f}'),
                'std_end': title3_text.find(f'{std_diff:.5f}') + len(f'{std_diff:.5f}')
            }
        
        return title1, title2, title3
    
    # Create titles
    title1, title2, title3_dict = create_titles(variable1, variable2, arr1_plot, arr2_plot, diff_plot, show_three)
    
    # Calculate color ranges
    if show_three:
        # Top panels: shared color range for arr1 and arr2
        vmin_top = min(np.nanmin(arr1_plot), np.nanmin(arr2_plot))
        vmax_top = max(np.nanmax(arr1_plot), np.nanmax(arr2_plot))
        
        # Bottom panel: separate color range for difference
        # Use the SAME colormap but with its own range
        vmin_diff = np.nanmin(diff_plot)
        vmax_diff = np.nanmax(diff_plot)
        
    elif show_two:
        # Two panels: shared color range
        vmin_top = min(np.nanmin(arr1_plot), np.nanmin(arr2_plot))
        vmax_top = max(np.nanmax(arr1_plot), np.nanmax(arr2_plot))
    else:
        # Single array
        vmin_top, vmax_top = np.nanmin(arr1_plot), np.nanmax(arr1_plot)
    
    # Create figure if it doesn't exist
    if _fig is None:
        # Create figure with appropriate layout
        if show_three:
            # Three panels: arr1 and arr2 on top row, difference on bottom row
            # Separate colorbars: one for top panels, one for bottom (same colormap)
            _fig = plt.figure(figsize=(14, 10))
            # Grid spec: 
            # Row 0: arr1 | arr2 | top colorbar
            # Row 1: difference | (empty) | bottom colorbar
            gs = plt.GridSpec(2, 3, 
                            height_ratios=[1, 1],
                            width_ratios=[1, 1, 0.05],
                            hspace=0.15, wspace=0.08,
                            left=0.04, right=0.96, top=0.92, bottom=0.04)  # Reduced top margin
            
            # Top row: arr1 and arr2
            _ax1 = _fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
            _ax2 = _fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
            
            # Top colorbar axis (right of top row)
            cbar_top_ax = _fig.add_subplot(gs[0, 2])
            
            # Bottom row: difference
            _ax3 = _fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
            
            # Bottom colorbar axis (right of bottom row, spans the empty cell)
            cbar_bottom_ax = _fig.add_subplot(gs[1, 2])
            
        elif show_two:
            # Two panels side by side with single colorbar
            _fig = plt.figure(figsize=(13, 5.5))
            gs = plt.GridSpec(1, 4, width_ratios=[1, 1, 0.05, 0.01], 
                            wspace=0.05, hspace=0, 
                            left=0.04, right=0.96, top=0.92, bottom=0.02)  # Reduced top margin
            _ax1 = _fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
            _ax2 = _fig.add_subplot(gs[1], projection=ccrs.PlateCarree())
            cbar_top_ax = _fig.add_subplot(gs[2])
            _ax3 = None
            cbar_bottom_ax = None
        else:
            # Single plot
            _fig, _ax1 = plt.subplots(1, 1, figsize=(10, 7),
                                     subplot_kw={'projection': ccrs.PlateCarree()})
            _fig.subplots_adjust(left=0.04, right=0.96, top=0.92, bottom=0.02)  # Reduced top margin
            _ax2 = None
            _ax3 = None
            cbar_top_ax = None
            cbar_bottom_ax = None
        
        # Set extent for your data (ERA5 grid)
        left, right, bottom, top = -180.0, 180.0, -90.0, 90.0
        
        # --- Plot First Array ---
        if show_two or show_three:
            _im1 = _ax1.imshow(arr1_plot,
                              transform=ccrs.PlateCarree(),
                              extent=[left, right, bottom, top],
                              cmap='viridis',
                              origin='lower',
                              vmin=vmin_top, vmax=vmax_top)
        else:
            _im1 = _ax1.imshow(arr1_plot,
                              transform=ccrs.PlateCarree(),
                              extent=[left, right, bottom, top],
                              cmap='viridis',
                              origin='lower')
        
        # Configure first subplot with stats in title
        _ax1.coastlines(resolution='110m', linewidth=0.5, color='black', alpha=0.7)
        gl1 = _ax1.gridlines(draw_labels=True, linewidth=0.7, color='gray', 
                           alpha=0.5, linestyle='--')
        gl1.xlocator = mticker.MultipleLocator(45)
        gl1.ylocator = mticker.MultipleLocator(45)
        gl1.top_labels = False
        gl1.right_labels = False
        _ax1.set_title(title1, fontsize=10, fontweight='bold', pad=6)  # Reduced pad
        
        # --- Plot Second Array if provided ---
        if show_two or show_three:
            _im2 = _ax2.imshow(arr2_plot,
                              transform=ccrs.PlateCarree(),
                              extent=[left, right, bottom, top],
                              cmap='viridis',
                              origin='lower',
                              vmin=vmin_top, vmax=vmax_top)
            
            # Configure second subplot with stats in title
            _ax2.coastlines(resolution='110m', linewidth=0.5, color='black', alpha=0.7)
            gl2 = _ax2.gridlines(draw_labels=True, linewidth=0.7, color='gray',
                               alpha=0.5, linestyle='--')
            gl2.xlocator = mticker.MultipleLocator(45)
            gl2.ylocator = mticker.MultipleLocator(45)
            gl2.top_labels = False
            gl2.right_labels = False
            gl2.left_labels = False  # Only show labels on left plot
            _ax2.set_title(title2, fontsize=10, fontweight='bold', pad=6)  # Reduced pad
        
        # --- Plot Difference Array if provided ---
        if show_three:
            # Use SAME colormap as top panels
            _im3 = _ax3.imshow(diff_plot,
                              transform=ccrs.PlateCarree(),
                              extent=[left, right, bottom, top],
                              cmap='viridis',  # Same colormap as top panels
                              origin='lower',
                              vmin=vmin_diff, vmax=vmax_diff)
            
            # Configure third subplot (difference) with stats in title
            _ax3.coastlines(resolution='110m', linewidth=0.5, color='black', alpha=0.7)
            gl3 = _ax3.gridlines(draw_labels=True, linewidth=0.7, color='gray',
                               alpha=0.5, linestyle='--')
            gl3.xlocator = mticker.MultipleLocator(45)
            gl3.ylocator = mticker.MultipleLocator(45)
            gl3.top_labels = False
            gl3.right_labels = False
            gl3.left_labels = True  # Show labels on left for bottom plot
            
            # Set the title for difference panel with colored mean/std values
            title_obj = _ax3.set_title(title3_dict['text'], fontsize=10, fontweight='bold', pad=6)
            
            # We'll store the title data for later updates
            _ax3._title_data = title3_dict
        
        # --- Add Color Bars ---
        if cbar_top_ax is not None:
            # Top colorbar for arr1 and arr2
            _cbar_top = plt.colorbar(_im1, cax=cbar_top_ax, orientation='vertical')
            if show_three:
                _cbar_top.set_label(f'{variable1} / {variable2}', fontsize=10)
            elif show_two:
                _cbar_top.set_label(f'{variable1} / {variable2}', fontsize=11)
        
        if cbar_bottom_ax is not None and show_three:
            # Bottom colorbar for difference (same colormap)
            _cbar_bottom = plt.colorbar(_im3, cax=cbar_bottom_ax, orientation='vertical')
            _cbar_bottom.set_label('Difference', fontsize=10)
        
        if not show_two and not show_three and cbar_top_ax is None:
            # Colorbar for single plot
            _cbar_top = plt.colorbar(_im1, ax=_ax1, orientation='vertical', 
                                   pad=0.02, shrink=0.95, aspect=20)
            _cbar_top.set_label(variable1, fontsize=12)
        
        # Set figure title with date/time info
        time_info = f'{dt.strftime("%Y-%m-%d %H:%M")} UTC | Delta Hours: {modulo}'
        if show_three:
            _fig.suptitle(f'{variable1} vs {variable2} | {time_info}',
                         fontsize=12, fontweight='bold', color='#00008B', 
                         x=0.01, y=0.97, ha='left', va='top')  # Adjusted y position
        elif show_two:
            _fig.suptitle(f'{variable1} vs {variable2} | {time_info}',
                         fontsize=12, fontweight='bold', color='#00008B', 
                         x=0.01, y=0.97, ha='left', va='top')  # Adjusted y position
        else:
            _fig.suptitle(f'{variable1} | {time_info}',
                         fontsize=12, fontweight='bold', color='#00008B', 
                         x=0.01, y=0.97, ha='left', va='top')  # Use suptitle for single plot
        
        # Show window (non-blocking)
        plt.show(block=False)
        plt.pause(0.001)
    
    else:
        # --- Update Existing Plot ---
        
        # Check if plot structure needs to change
        current_show_three = _ax3 is not None and _im3 is not None
        current_show_two = _ax2 is not None and _im2 is not None
        if (show_three != current_show_three) or (show_two != current_show_two):
            # Structure changed, recreate figure
            plt.close(_fig)
            _fig = None
            _ax1 = _ax2 = _ax3 = None
            _im1 = _im2 = _im3 = None
            _cbar_top = _cbar_bottom = None
            # Recursive call with new figure
            show_two_arrays(arr1, arr2, variable1, variable2, dt, modulo)
            return
        
        # Update first array
        _im1.set_data(arr1_plot)
        _im1.set_clim(vmin_top, vmax_top)
        
        # Update second array if exists
        if show_two and _im2 is not None:
            _im2.set_data(arr2_plot)
            _im2.set_clim(vmin_top, vmax_top)
        
        # Update difference panel if exists
        if show_three and _im3 is not None:
            _im3.set_data(diff_plot)
            _im3.set_clim(vmin_diff, vmax_diff)
        
        # Recreate titles with current statistics
        title1, title2, title3_dict = create_titles(variable1, variable2, arr1_plot, arr2_plot, diff_plot, show_three)
        
        _ax1.set_title(title1, fontsize=10, fontweight='bold', pad=6)
        if show_two or show_three:
            _ax2.set_title(title2, fontsize=10, fontweight='bold', pad=6)
        if show_three:
            # Update the title for difference panel
            _ax3.set_title(title3_dict['text'], fontsize=10, fontweight='bold', pad=6)
            # Store the updated title data
            _ax3._title_data = title3_dict
        
        # Update colorbars
        if _cbar_top is not None:
            if show_three:
                _cbar_top.set_label(f'{variable1} / {variable2}', fontsize=10)
            elif show_two:
                _cbar_top.set_label(f'{variable1} / {variable2}', fontsize=11)
            else:
                _cbar_top.set_label(variable1, fontsize=12)
            _cbar_top.update_normal(_im1)
        
        if show_three and _cbar_bottom is not None:
            _cbar_bottom.update_normal(_im3)
        
        # Update figure title
        time_info = f'{dt.strftime("%Y-%m-%d %H:%M")} UTC | Delta Hours: {modulo}'
        if show_three:
            if hasattr(_fig, 'suptitle'):
                _fig.suptitle(f'{variable1} vs {variable2} | {time_info}',
                             fontsize=12, fontweight='bold', color='#00008B', 
                             x=0.01, y=0.97, ha='left', va='top')
        
        elif show_two:
            if hasattr(_fig, 'suptitle'):
                _fig.suptitle(f'{variable1} vs {variable2} | {time_info}',
                             fontsize=12, fontweight='bold', color='#00008B', 
                             x=0.01, y=0.97, ha='left', va='top')
        
        else:
            if hasattr(_fig, 'suptitle'):
                _fig.suptitle(f'{variable1} | {time_info}',
                             fontsize=12, fontweight='bold', color='#00008B', 
                             x=0.01, y=0.97, ha='left', va='top')
        
        # Force redraw
        _fig.canvas.draw_idle()
        _fig.canvas.flush_events()
    
    # Small pause for UI responsiveness
    plt.pause(0.001)

def show_array_single(arr1, arr2=None, variable1="", variable2="", dt=datetime(2000,1,1,0), modulo=6):
    """
    Show one or two arrays in a standalone figure with top-left alignment.
    """
    # Prepare arrays
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
    
    arr1_plot = arr1 #prepare_array(arr1)
    arr2_plot = arr2 #prepare_array(arr2) if arr2 is not None else None
    
    show_two = arr2_plot is not None
    
    # Create figure with minimal top/bottom margins for top-left feel
    if show_two:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5),
                                      subplot_kw={'projection': ccrs.PlateCarree()})
        fig.subplots_adjust(left=0.04, right=0.96, top=0.98, bottom=0.02, wspace=0.05)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 7),
                               subplot_kw={'projection': ccrs.PlateCarree()})
        fig.subplots_adjust(left=0.04, right=0.96, top=0.98, bottom=0.02)
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
    cbar1 = plt.colorbar(im1, ax=ax1, orientation='vertical', 
                        pad=0.02, shrink=0.95, aspect=20)
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
        cbar2 = plt.colorbar(im2, ax=ax2, orientation='vertical', 
                            pad=0.02, shrink=0.95, aspect=20)
        cbar2.set_label(variable2, fontsize=12)
        
        # Title for two-panel figure - top-left aligned
        fig.suptitle(f'{variable1} vs. {variable2} | {dt.strftime("%Y-%m-%d %H:%M")} UTC | Delta Hours: {modulo}',
                    fontsize=12, fontweight='bold', color='#00008B', 
                    x=0.01, y=0.99, ha='left', va='top')
    else:
        # Title for single plot - left-aligned
        ax1.set_title(f'{variable1} | {dt.strftime("%Y-%m-%d %H:%M")} UTC | Delta Hours: {modulo}',
                     fontsize=13, fontweight='bold', color='#00008B', 
                     pad=12, loc='left', x=0.0)
    
    # Position window at top-left of screen
    manager = plt.get_current_fig_manager()
    if manager is not None:
        try:
            backend = plt.get_backend()
            if 'Tk' in backend:
                manager.window.wm_geometry("+0+0")
            elif 'Qt' in backend:
                manager.window.move(0, 0)
            elif 'WX' in backend:
                manager.SetPosition((0, 0))
            else:
                if hasattr(manager, 'window'):
                    if hasattr(manager.window, 'geometry'):
                        manager.window.setGeometry(0, 0, 
                                                  manager.window.width(), 
                                                  manager.window.height())
        except:
            pass
    
    plt.show()
    
    #--------------------------------------------------------
    
def resize_one_to_other_cylindric(arr1, arr2, mode='bilinear', larger_to_smaller=True):
    """
    Resize the larger tensor to match the smaller one with cylindrical horizontal resizing.
    
    Args:
        arr1: (C, H1, W1) or (H1, W1)
        arr2: (C, H2, W2) or (H2, W2)
        mode: interpolation mode ('bilinear', 'nearest', etc.)
    
    Returns:
        Two tensors with the same dimensions (the smaller dimension)
    """
    import torch
    import torch.nn.functional as F
    
    tensor1 = torch.from_numpy(arr1)
    tensor2 = torch.from_numpy(arr2)
    
    # Get shapes
    if tensor1.dim() == 3:
        C1, H1, W1 = tensor1.shape
    else:
        H1, W1 = tensor1.shape
        C1 = 1
        tensor1 = tensor1.unsqueeze(0)
    
    if tensor2.dim() == 3:
        C2, H2, W2 = tensor2.shape
    else:
        H2, W2 = tensor2.shape
        C2 = 1
        tensor2 = tensor2.unsqueeze(0)
    
    # Check if they already have the same size
    if H1 == H2 and W1 == W2:
        return tensor1, tensor2
    
    # Determine which tensor is larger (by total pixels)
    size1 = H1 * W1
    size2 = H2 * W2
    
    if size1 > size2:
        # tensor1 is larger, resize it to match tensor2
        if larger_to_smaller:
            target_size = (H2, W2)
            tensor1_resized = resize_continuous_bilinear_channel_wise_cylindric(
                tensor1, target_size, mode
            )
            tensor2_resized = tensor2
            # print(f"Resized tensor1: {tensor1.shape}->{tensor1_resized.shape} to match tensor2: {tensor2.shape}")
        else:
            target_size = (H1, W1)
            tensor2_resized = resize_continuous_bilinear_channel_wise_cylindric(
                tensor2, target_size, mode
            )
            tensor1_resized = tensor1
            # print(f"Resized tensor2: {tensor2.shape}->{tensor2_resized.shape} to match tensor1: {tensor1.shape}")           
            
    elif size2 > size1:
        if larger_to_smaller:
            # tensor2 is larger, resize it to match tensor1
            target_size = (H1, W1)
            tensor2_resized = resize_continuous_bilinear_channel_wise_cylindric(
                tensor2, target_size, mode
            )
            tensor1_resized = tensor1
            # print(f"Resized tensor2: {tensor2.shape}->{tensor2_resized.shape} to match tensor1: {tensor1.shape}")
        else:
            target_size = (H2, W2)
            tensor1_resized = resize_continuous_bilinear_channel_wise_cylindric(
                tensor1, target_size, mode
            )
            tensor2_resized = tensor2
            # print(f"Resized tensor1: {tensor1.shape}->{tensor1_resized.shape} to match tensor2: {tensor2.shape}")
    else:
        pass
    
    # Ensure both have same number of channels
    if tensor1_resized.dim() != tensor2_resized.dim():
        if tensor1_resized.dim() == 2:
            tensor1_resized = tensor1_resized.unsqueeze(0)
        if tensor2_resized.dim() == 2:
            tensor2_resized = tensor2_resized.unsqueeze(0)
    
    return tensor1_resized.numpy(), tensor2_resized.numpy()

import torch
import torch.nn.functional as F

# Your original function (unchanged except for better comments)
def resize_continuous_bilinear_channel_wise_cylindric(tensor, target_size, mode='bilinear'):
    """
    CORRECT cylindrical resize for cyclic longitude dimension.
    
    Algorithm:
    1. Unwrap cyclic data to larger size (wrap on BOTH sides)
    2. Resize the unwrapped (larger) tensor
    3. Crop the center portion corresponding to original data
    
    Args:
        tensor: (C, H, W) where W is cyclic (longitude)
        target_size: (target_h, target_w)
    
    Returns:
        Resized tensor (C, target_h, target_w) with preserved cyclic continuity
    """
    
    if tensor.dim() == 3:
        x = tensor.unsqueeze(0)  # Add batch: (1, C, H, W)
        has_batch = False
    else:
        x = tensor
        has_batch = True
    
    target_h, target_w = target_size
    B, C, H, W = x.shape
    
    # If already at target size, return
    if H == target_h and W == target_w:
        if not has_batch:
            return tensor
        else:
            return x
    
    # --- STEP 1: Determine wrap amount ---
    # For cyclic data, wrap HALF the width on EACH side
    # This ensures continuity when we later crop the center
    wrap_W = W // 2  # Wrap half the width
    
    # --- STEP 2: Create unwrapped tensor ---
    # Pattern: [RIGHT-HALF, FULL, LEFT-HALF]
    # This creates a continuous representation
    right_half = x[..., -wrap_W:]  # Last wrap_W columns
    left_half = x[..., :wrap_W]    # First wrap_W columns
    
    # Unwrapped tensor: (1, C, H, W + 2*wrap_W)
    x_unwrapped = torch.cat([right_half, x, left_half], dim=-1)
    
    # --- STEP 3: Resize the UNWRAPPED tensor ---
    # Calculate new wrap size proportionally
    unwrapped_W = W + 2 * wrap_W
    scale_factor = target_w / W
    
    # New dimensions for unwrapped tensor
    target_unwrapped_W = int(target_w + 2 * wrap_W * scale_factor)
    
    x_resized_unwrapped = F.interpolate(
        x_unwrapped,
        size=(target_h, target_unwrapped_W),
        mode=mode,
        align_corners=False
    )
    
    # --- STEP 4: Crop the CENTER portion ---
    # After resizing, we need to find where the original data is
    # It's in the middle, between the wrapped portions
    current_unwrapped_W = x_resized_unwrapped.shape[-1]
    
    # The wrapped portions scaled proportionally
    new_wrap_W = int(wrap_W * scale_factor)
    
    # Crop the middle part (original data)
    start = new_wrap_W
    end = current_unwrapped_W - new_wrap_W
    result = x_resized_unwrapped[..., start:end]
    
    # --- STEP 5: Ensure exact target size ---
    # Sometimes floating point math causes off-by-one
    if result.shape[-1] != target_w:
        # Final adjustment if needed
        result = F.interpolate(
            result,
            size=(target_h, target_w),
            mode=mode,
            align_corners=False
        )
    
    # Remove batch dimension if we added it
    if not has_batch:
        result = result.squeeze(0)
    
    return result

def resize_categorical_nearest_channelwise_cylindric(tensor, target_resolution):
    """
    Perfect cylindrical resize for categorical data.
    Ensures left and right edges have similar distributions.
    """
    
    if tensor.dim() == 3:
        x = tensor.unsqueeze(0)  # Add batch: (1, C, H, W)
        has_batch = False
    else:
        x = tensor
        has_batch = True
    
    target_h, target_w = target_resolution
    B, C, H, W = x.shape
    
    # If already at target size, return
    if H == target_h and W == target_w:
        if not has_batch:
            return tensor
        else:
            return x
    
    target_H, target_W = target_resolution
    C, H, W = tensor.shape
    device = tensor.device
    
    # --- Create mapping that ensures edge continuity ---
    # The key: make sure j_norm for leftmost and rightmost pixels
    # map to similar source regions
    
    # Create j_norm from -0.5 to 0.5 (centered)
    # This ensures continuity at edges
    j_norm = torch.linspace(-0.5, 0.5, target_W, device=device)
    
    # Map to source coordinates
    j_source_float = (j_norm + 0.5) * W  # Now in [0, W]
    
    # Convert to indices with modulo
    j_source = torch.floor(j_source_float).long() % W
    
    # For latitude
    i_norm = torch.linspace(0, 1, target_H, device=device)
    i_source_float = i_norm * (H - 1)
    i_source = torch.floor(i_source_float).long().clamp(0, H-1)
    
    # Create meshgrid
    I, J = torch.meshgrid(i_source, j_source, indexing='ij')
    
    # Sample
    result = tensor[:, I, J]
    
    # Remove batch dimension if we added it
    if not has_batch:
        result = result.squeeze(0)
    
    return result

import importlib, hashlib
def get_module_hash(module_name):
    """Get hash of a module's source code."""
    module = importlib.import_module(module_name)
    
    # Get module file path
    module_path = Path(module.__file__)
    
    # Read and hash the source file
    with open(module_path, 'rb') as f:
        content = f.read()
    
    return hashlib.sha256(content).hexdigest()

def compare_two_arrays(arr1, arr2=None, dt=datetime(2000,1,1,0), modulo=6):
    assert arr1 is not None
    assert arr2 is not None
    
    assert len(arr1.shape) == 2
    assert len(arr2.shape) == 2
    
    assert not np.isnan(arr1).any()
    assert not np.isnan(arr2).any()
    
    # Resize arrays if they have different shapes
    if arr1.shape != arr2.shape:
        arr1_resized, arr2_resized = resize_one_to_other_cylindric(arr1[None, :, :], arr2[None, :, :], larger_to_smaller=False)
        arr1_resized, arr2_resized = arr1_resized[0], arr2_resized[0]
        
    delta = arr2_resized - arr1_resized

    return delta.mean(), delta.std()

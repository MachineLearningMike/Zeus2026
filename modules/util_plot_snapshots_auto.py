import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict
import re

from matplotlib.widgets import CheckButtons  # Add this import!
from tabulate import tabulate  # Optional, for nice tables

class SnapshotAnalyzer:
    def __init__(self, snapshot_dir='D:\zeus_data\snapshots_Zeus_uNet_Feb22_300'): #'./snapshots'):
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshots = {}
        
    def load_all_snapshots(self):
        """Load all snapshot files in increasing order"""
        snapshot_files = sorted(self.snapshot_dir.glob('snapshot_*.jsonl'))
        
        for filepath in snapshot_files:
            ss = int(filepath.stem.replace('snapshot_', ''))
            snapshot_data = {}
            with open(filepath, 'r') as f:
                for line in f:
                    record = json.loads(line)
                    snapshot_data[record['var']] = record['stats']
            self.snapshots[ss] = snapshot_data
        
        return sorted(self.snapshots.keys())
    
    def get_variable_history(self, var_pattern=None, key_filters=None, match_all=True):
        """
        Extract history of variables across snapshots.
        
        Args:
            var_pattern: Optional regex pattern to filter variables
            key_filters: Optional list of strings to search for in variable name
            match_all: If True (default), variable must contain ALL key_filters (AND)
                      If False, variable must contain ANY key_filters (OR)
        """
        if not self.snapshots:
            self.load_all_snapshots()
        
        all_vars = set()
        for data in self.snapshots.values():
            all_vars.update(data.keys())
        
        # Apply filters
        filtered_vars = set()
        for var in all_vars:
            # Check var_pattern if provided
            if var_pattern and not re.search(var_pattern, var):
                continue
            
            # Check key_filters if provided
            if key_filters:
                if match_all:
                    # AND logic: must contain ALL filters
                    matches = True
                    for filter_str in key_filters:
                        if filter_str not in var:
                            matches = False
                            break
                else:
                    # OR logic: must contain ANY filter
                    matches = False
                    for filter_str in key_filters:
                        if filter_str in var:
                            matches = True
                            break
                
                if not matches:
                    continue
            
            filtered_vars.add(var)
        
        history = {}
        sorted_ss = sorted(self.snapshots.keys())
        
        for var in sorted(filtered_vars):
            means = []
            stds = []
            valid_ss = []
            
            for ss in sorted_ss:
                if var in self.snapshots[ss]:
                    stats = self.snapshots[ss][var]
                    if 'mean' in stats and 'std' in stats:
                        means.append(stats['mean'])
                        stds.append(stats['std'])
                        valid_ss.append(ss)
            
            if means:
                history[var] = {
                    'snapshots': valid_ss,
                    'means': means,
                    'stds': stds,
                    'last_segment': var.split('.')[-1]
                }
        
        return history
    
    def filter_by_keys(self, key_filters, match_all=True):
        """
        Get history filtered by keys anywhere in variable name.
        
        Args:
            key_filters: String or list of strings to search for anywhere in variable name
            match_all: If True (default), variable must contain ALL key_filters (AND)
                      If False, variable must contain ANY key_filters (OR)
        """
        if isinstance(key_filters, str):
            key_filters = [key_filters]
        return self.get_variable_history(key_filters=key_filters, match_all=match_all)
    
    def get_filter_description(self, filters, match_all=True):
        """Create a readable description of the filters"""
        if not filters:
            return ""
        if isinstance(filters, str):
            return f" (name contains '{filters}')"
        if len(filters) == 1:
            return f" (name contains '{filters[0]}')"
        
        logic = "ALL of" if match_all else "ANY of"
        return f" (name contains {logic}: {', '.join(filters)})"
    
    # ========== PLOT FUNCTIONS WITH KEY FILTERS ==========
    
    def plot_all_lines(self, key_filters=None, match_all=True, value_type='both', figsize=(12, 6), title=None, linewidth=0.5):
        """Plot variables as lines, optionally filtered by keys anywhere in name
        
        Args:
            key_filters: Optional list of strings to filter variable names
            match_all: If True, variable must contain ALL key_filters (AND)
            value_type: 'mean', 'std', or 'both' (default)
            figsize: Figure size
            title: Optional custom title (if None, auto-generated)
            linewidth: Width of the plot lines (default: 0.5)
        """
        if key_filters:
            history = self.filter_by_keys(key_filters, match_all=match_all)
            title_suffix = self.get_filter_description(key_filters, match_all)
        else:
            history = self.get_variable_history()
            title_suffix = ""
        
        if not history:
            filter_desc = self.get_filter_description(key_filters, match_all) if key_filters else "any"
            print(f"No variables found{filter_desc}")
            return
        
        plt.figure(figsize=figsize)
        
        # Use a colormap for many lines
        colors = plt.cm.tab20(np.linspace(0, 1, len(history)))
        
        for idx, (var, data) in enumerate(history.items()):
            color = colors[idx % len(colors)]
            label = f'{var[-30:]}...' if len(var) > 30 else var
            
            if value_type in ['mean', 'both']:
                plt.plot(data['snapshots'], data['means'], 
                        '-', linewidth=linewidth, color=color, 
                        label=f'{label} (mean)' if value_type == 'both' else label)
            
            if value_type in ['std', 'both']:
                plt.plot(data['snapshots'], data['stds'], 
                        ':', linewidth=linewidth, color=color, 
                        label=f'{label} (std)' if value_type == 'both' else None)
        
        plt.xlabel('Snapshot ID', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        
        # Use custom title if provided, otherwise auto-generate
        if title:
            plt.title(title, fontsize=14)
        else:
            if value_type == 'both':
                plt.title(f'Variables{title_suffix} - Mean (solid) vs Std (dotted) ({len(history)} variables)', fontsize=14)
            else:
                plt.title(f'Variables{title_suffix} - {value_type.capitalize()} ({len(history)} variables)', fontsize=14)
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    def plot_variable_list(self, var_list, value_type='both', figsize=(12, 6), title=None, linewidth=0.5):
        """Plot a specific list of variables
        
        Args:
            var_list: List of exact variable names to plot
            value_type: 'mean', 'std', or 'both' (default)
            figsize: Figure size
            title: Optional custom title (if None, auto-generated)
            linewidth: Width of the plot lines (default: 0.5)
        """
        if not self.snapshots:
            self.load_all_snapshots()
        
        # Get history for all variables
        all_history = self.get_variable_history()
        
        # Filter to only the requested variables
        history = {}
        for var in var_list:
            if var in all_history:
                history[var] = all_history[var]
            else:
                print(f"Warning: Variable '{var}' not found in snapshots")
        
        if not history:
            print("No valid variables found in the provided list")
            return
        
        plt.figure(figsize=figsize)
        
        # Use a colormap for many lines
        colors = plt.cm.tab20(np.linspace(0, 1, len(history)))
        
        for idx, (var, data) in enumerate(history.items()):
            color = colors[idx % len(colors)]
            label = f'{var[-30:]}...' if len(var) > 30 else var
            
            if value_type in ['mean', 'both']:
                plt.plot(data['snapshots'], data['means'], 
                        '-', linewidth=linewidth, color=color,
                        label=f'{label} (mean)' if value_type == 'both' else label)
            
            if value_type in ['std', 'both']:
                plt.plot(data['snapshots'], data['stds'], 
                        ':', linewidth=linewidth, color=color,
                        label=f'{label} (std)' if value_type == 'both' else None)
        
        plt.xlabel('Snapshot ID', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        
        var_count = len(history)
        # Use custom title if provided, otherwise auto-generate
        if title:
            plt.title(title, fontsize=14)
        else:
            if value_type == 'both':
                plt.title(f'Selected Variables - Mean (solid) vs Std (dotted) ({var_count} variables)', fontsize=14)
            else:
                plt.title(f'Selected Variables - {value_type.capitalize()} ({var_count} variables)', fontsize=14)
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Return the list of variables that were actually plotted
        return list(history.keys())
    
    def plot_all_heatmap(self, key_filters=None, match_all=True, figsize=(16, 10), title=None, linewidth=0.5):
        """Plot heatmap of variables, optionally filtered by keys anywhere in name"""
        if key_filters:
            history = self.filter_by_keys(key_filters, match_all=match_all)
            title_suffix = self.get_filter_description(key_filters, match_all)
        else:
            history = self.get_variable_history()
            title_suffix = ""
        
        if not history:
            filter_desc = self.get_filter_description(key_filters, match_all) if key_filters else "any"
            print(f"No variables found{filter_desc}")
            return
        
        # Get all snapshot IDs
        all_ss = sorted(self.snapshots.keys())
        
        # Create matrix of means
        var_names = sorted(history.keys())
        matrix = np.zeros((len(var_names), len(all_ss)))
        
        for i, var in enumerate(var_names):
            var_history = history[var]
            ss_to_mean = dict(zip(var_history['snapshots'], var_history['means']))
            for j, ss in enumerate(all_ss):
                matrix[i, j] = ss_to_mean.get(ss, np.nan)
        
        # Create heatmap
        fig, (ax, cax) = plt.subplots(1, 2, figsize=figsize, 
                                       gridspec_kw={'width_ratios': [20, 1]})
        
        im = ax.imshow(matrix, aspect='auto', cmap='RdBu_r', interpolation='nearest')
        
        # Labels
        ax.set_xlabel('Snapshot ID', fontsize=12)
        ax.set_ylabel('Variable', fontsize=12)
        
        # Use custom title if provided, otherwise auto-generate
        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title(f'Variables Heatmap{title_suffix}\n({len(var_names)} variables, {len(all_ss)} snapshots)', fontsize=14)
        
        # Set ticks
        ax.set_xticks(range(len(all_ss)))
        ax.set_xticklabels(all_ss, rotation=45)
        
        # Show variable names
        ax.set_yticks(range(len(var_names)))
        ax.set_yticklabels([var[-30:] if len(var) > 30 else var for var in var_names])
        
        # Colorbar
        plt.colorbar(im, cax=cax, label='Mean Value')
        
        plt.tight_layout()
        plt.show()
        
        return matrix, var_names, all_ss
    
    def plot_all_boxplot(self, key_filters=None, match_all=True, figsize=(15, 8), title=None, linewidth=0.5):
        """Box plot of variables, optionally filtered by keys anywhere in name"""
        if key_filters:
            history = self.filter_by_keys(key_filters, match_all=match_all)
            title_suffix = self.get_filter_description(key_filters, match_all)
        else:
            history = self.get_variable_history()
            title_suffix = ""
        
        if not history:
            filter_desc = self.get_filter_description(key_filters, match_all) if key_filters else "any"
            print(f"No variables found{filter_desc}")
            return
        
        # Prepare data for boxplot
        var_names = sorted(history.keys())
        data = [history[var]['means'] for var in var_names]
        
        plt.figure(figsize=figsize)
        
        # Create boxplot
        bp = plt.boxplot(data, labels=[var[-20:] if len(var) > 20 else var for var in var_names], 
                         vert=True, patch_artist=True)
        
        # Color boxes
        for box in bp['boxes']:
            box.set_alpha(0.7)
            box.set_linewidth(linewidth)
        
        plt.xticks(rotation=90, fontsize=8)
        plt.ylabel('Mean Value', fontsize=12)
        
        # Use custom title if provided, otherwise auto-generate
        if title:
            plt.title(title, fontsize=14)
        else:
            plt.title(f'Distribution of Variables{title_suffix} ({len(var_names)} total)', fontsize=14)
        
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.show()
    
    def plot_all_trajectories(self, key_filters=None, match_all=True, figsize=(15, 10), title=None, linewidth=0.5):
        """Plot trajectories colored by final value"""
        if key_filters:
            history = self.filter_by_keys(key_filters, match_all=match_all)
            title_suffix = self.get_filter_description(key_filters, match_all)
        else:
            history = self.get_variable_history()
            title_suffix = ""
        
        if not history:
            filter_desc = self.get_filter_description(key_filters, match_all) if key_filters else "any"
            print(f"No variables found{filter_desc}")
            return
        
        plt.figure(figsize=figsize)
        
        # Color by final value
        final_means = {}
        for var, data in history.items():
            if data['snapshots']:
                final_means[var] = data['means'][-1]
        
        if final_means:
            norm = plt.Normalize(min(final_means.values()), max(final_means.values()))
            cmap = plt.cm.viridis
            
            for var, data in history.items():
                color = cmap(norm(final_means[var]))
                plt.plot(data['snapshots'], data['means'], 
                        '-', linewidth=linewidth, color=color, alpha=0.7, 
                        label=var[-30:] if len(var) > 30 else var)
            
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            plt.colorbar(sm, label='Final Mean Value')
        
        plt.xlabel('Snapshot ID', fontsize=12)
        plt.ylabel('Mean Value', fontsize=12)
        
        # Use custom title if provided, otherwise auto-generate
        if title:
            plt.title(title, fontsize=14)
        else:
            plt.title(f'Variable Trajectories{title_suffix}', fontsize=14)
        
        plt.grid(True, alpha=0.3)
        if len(history) < 20:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.tight_layout()
        plt.show()
    
    def plot_all_stacked(self, key_filters=None, match_all=True, figsize=(12, 12), title=None, linewidth=0):
        """Stacked area plot, filtered by keys anywhere in name
        
        Args:
            linewidth: Width of the edges between areas (default: 0 for no edges)
        """
        if key_filters:
            history = self.filter_by_keys(key_filters, match_all=match_all)
            title_suffix = self.get_filter_description(key_filters, match_all)
        else:
            history = self.get_variable_history()
            title_suffix = ""
        
        if not history:
            filter_desc = self.get_filter_description(key_filters, match_all) if key_filters else "any"
            print(f"No variables found{filter_desc}")
            return
        
        # Get all snapshots
        all_ss = sorted(self.snapshots.keys())
        
        # Create matrix of means
        var_names = sorted(history.keys())
        matrix = np.zeros((len(var_names), len(all_ss)))
        
        for i, var in enumerate(var_names):
            var_history = history[var]
            ss_to_mean = dict(zip(var_history['snapshots'], var_history['means']))
            for j, ss in enumerate(all_ss):
                matrix[i, j] = ss_to_mean.get(ss, 0)
        
        # Stacked area plot
        plt.figure(figsize=figsize)
        
        plt.stackplot(all_ss, matrix, labels=[var[-20:] if len(var) > 20 else var for var in var_names], 
                     alpha=0.7, edgecolor='none', linewidth=linewidth)
        
        plt.xlabel('Snapshot ID', fontsize=12)
        plt.ylabel('Cumulative Mean Value', fontsize=12)
        
        # Use custom title if provided, otherwise auto-generate
        if title:
            plt.title(title, fontsize=14)
        else:
            plt.title(f'Stacked Contribution of Variables{title_suffix}', fontsize=14)
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_all_parallel(self, key_filters=None, match_all=True, figsize=(14, 8), title=None, linewidth=0.5):
        """Parallel coordinates plot, filtered by keys anywhere in name"""
        if key_filters:
            history = self.filter_by_keys(key_filters, match_all=match_all)
            title_suffix = self.get_filter_description(key_filters, match_all)
        else:
            history = self.get_variable_history()
            title_suffix = ""
        
        if not history:
            filter_desc = self.get_filter_description(key_filters, match_all) if key_filters else "any"
            print(f"No variables found{filter_desc}")
            return
        
        # Get all snapshots
        all_ss = sorted(self.snapshots.keys())
        
        # Prepare data
        var_names = sorted(history.keys())
        data = []
        
        for var in var_names:
            var_history = history[var]
            ss_to_mean = dict(zip(var_history['snapshots'], var_history['means']))
            row = [ss_to_mean.get(ss, np.nan) for ss in all_ss]
            data.append(row)
        
        data = np.array(data)
        
        # Parallel coordinates plot
        plt.figure(figsize=figsize)
        
        # Normalize each variable for better visualization
        data_norm = (data - np.nanmin(data, axis=1, keepdims=True)) / \
                    (np.nanmax(data, axis=1, keepdims=True) - np.nanmin(data, axis=1, keepdims=True))
        
        for i in range(len(var_names)):
            var_name = var_names[i]
            plt.plot(range(len(all_ss)), data_norm[i], '-', 
                    linewidth=linewidth, alpha=0.7,
                    label=var_name[-30:] if len(var_name) > 30 else var_name)
        
        plt.xticks(range(len(all_ss)), all_ss)
        plt.xlabel('Snapshot ID', fontsize=12)
        plt.ylabel('Normalized Value', fontsize=12)
        
        # Use custom title if provided, otherwise auto-generate
        if title:
            plt.title(title, fontsize=14)
        else:
            plt.title(f'Parallel Coordinates Plot{title_suffix}', fontsize=14)
        
        plt.grid(True, alpha=0.3)
        
        if len(history) < 20:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.tight_layout()
        plt.show()
    
    def plot_all_stats(self, key_filters=None, match_all=True, figsize=(15, 10), title=None, linewidth=0.5):
        """Statistical overview, filtered by keys anywhere in name"""
        if key_filters:
            history = self.filter_by_keys(key_filters, match_all=match_all)
            title_suffix = self.get_filter_description(key_filters, match_all)
        else:
            history = self.get_variable_history()
            title_suffix = ""
        
        if not history:
            filter_desc = self.get_filter_description(key_filters, match_all) if key_filters else "any"
            print(f"No variables found{filter_desc}")
            return
        
        # Calculate statistics for each variable
        stats = {}
        for var, data in history.items():
            means = data['means']
            stats[var] = {
                'mean': np.mean(means),
                'std': np.std(means),
                'min': np.min(means),
                'max': np.max(means),
                'range': np.max(means) - np.min(means),
                'trend': means[-1] - means[0] if len(means) > 1 else 0,
                'short_name': var[-30:] if len(var) > 30 else var
            }
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # 1. Distribution of means
        axes[0, 0].hist([s['mean'] for s in stats.values()], bins=30, alpha=0.7, edgecolor='black', linewidth=linewidth)
        axes[0, 0].set_xlabel('Mean Value')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title(f'Distribution of Means{title_suffix}')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Distribution of stds
        axes[0, 1].hist([s['std'] for s in stats.values()], bins=30, alpha=0.7, color='green', edgecolor='black', linewidth=linewidth)
        axes[0, 1].set_xlabel('Std Value')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title(f'Distribution of Standard Deviations{title_suffix}')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Mean vs Std scatter
        axes[0, 2].scatter([s['mean'] for s in stats.values()], 
                          [s['std'] for s in stats.values()], 
                          alpha=0.6, s=10, edgecolor='black', linewidth=linewidth)
        axes[0, 2].set_xlabel('Mean')
        axes[0, 2].set_ylabel('Std')
        axes[0, 2].set_title(f'Mean vs Std{title_suffix}')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Range distribution
        axes[1, 0].hist([s['range'] for s in stats.values()], bins=30, alpha=0.7, color='red', edgecolor='black', linewidth=linewidth)
        axes[1, 0].set_xlabel('Range (Max - Min)')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title(f'Distribution of Ranges{title_suffix}')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Trend distribution
        axes[1, 1].hist([s['trend'] for s in stats.values()], bins=30, alpha=0.7, color='purple', edgecolor='black', linewidth=linewidth)
        axes[1, 1].set_xlabel('Trend (Final - Initial)')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title(f'Distribution of Trends{title_suffix}')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Top variables by mean magnitude
        top_vars = sorted(stats.items(), key=lambda x: abs(x[1]['mean']), reverse=True)[:10]
        var_names = [v[1]['short_name'] for v in top_vars]
        var_means = [v[1]['mean'] for v in top_vars]
        
        axes[1, 2].barh(range(len(var_names)), var_means, alpha=0.7, color='orange', edgecolor='black', linewidth=linewidth)
        axes[1, 2].set_yticks(range(len(var_names)))
        axes[1, 2].set_yticklabels(var_names)
        axes[1, 2].set_xlabel('Mean Value')
        axes[1, 2].set_title('Top 10 Variables by |Mean|')
        axes[1, 2].grid(True, alpha=0.3)
        
        # Use custom title if provided, otherwise auto-generate
        if title:
            plt.suptitle(title, fontsize=14)
        else:
            plt.suptitle(f'Statistical Overview of {len(history)} Variables{title_suffix}', fontsize=14)
        
        plt.tight_layout()
        plt.show()
    
    def print_summary(self, key_filters=None, match_all=True):
        """Print summary, optionally filtered by keys anywhere in name"""
        if not self.snapshots:
            self.load_all_snapshots()
        
        if key_filters:
            history = self.filter_by_keys(key_filters, match_all=match_all)
            title = self.get_filter_description(key_filters, match_all)
        else:
            history = self.get_variable_history()
            title = ""
        
        print("\n" + "="*70)
        print(f"SNAPSHOT SUMMARY{title}")
        print("="*70)
        
        sorted_ss = sorted(self.snapshots.keys())
        print(f"\n📁 Found {len(sorted_ss)} snapshots: {sorted_ss}")
        
        print(f"\n📈 Total unique variables{title}: {len(history)}")
        
        if history:
            # Calculate global statistics
            all_means = []
            for data in history.values():
                all_means.extend(data['means'])
            
            print(f"\n📉 Global mean statistics:")
            print(f"  Overall mean: {np.mean(all_means):.4f}")
            print(f"  Overall std: {np.std(all_means):.4f}")
            print(f"  Min value: {np.min(all_means):.4f}")
            print(f"  Max value: {np.max(all_means):.4f}")
            
            # Show sample variables
            print(f"\n📝 Sample variables (first 5):")
            for var in list(history.keys())[:5]:
                print(f"  {var[:70]}...")
        
        print("="*70)

    def interactive_variable_selector(self, key_filters=None, match_all=True, value_type='both', figsize=(14, 8), title=None, linewidth=0.5):
        """Create an interactive plot with checkboxes to select which variables to display
        
        Args:
            key_filters: Optional list of strings to filter variable names
            match_all: If True, variable must contain ALL key_filters (AND)
            value_type: 'mean', 'std', or 'both'
            figsize: Figure size
            title: Optional custom title (if None, auto-generated)
            linewidth: Width of the plot lines (default: 0.5)
        """
        if key_filters:
            history = self.filter_by_keys(key_filters, match_all=match_all)
            title_suffix = self.get_filter_description(key_filters, match_all)
        else:
            history = self.get_variable_history()
            title_suffix = ""
        
        if not history:
            filter_desc = self.get_filter_description(key_filters, match_all) if key_filters else "any"
            print(f"No variables found{filter_desc}")
            return
        
        # Get variable names and sort them
        var_names = sorted(history.keys())
        n_vars = len(var_names)
        
        # Limit to first 15 variables to keep checkbox manageable
        if n_vars > 15:
            print(f"Found {n_vars} variables. Showing first 15 for checkbox selection.")
            var_names = var_names[:15]
            n_vars = 15
        
        # Create figure with extra space for checkboxes
        fig = plt.figure(figsize=figsize)
        fig.subplots_adjust(left=0.4, bottom=0.1, right=0.95, top=0.95)
        
        # Main axis for plotting
        ax = fig.add_subplot(111)
        
        # Colors for lines
        colors = plt.cm.tab20(np.linspace(0, 1, n_vars))
        
        # Store line objects and their corresponding checkbox indices
        lines = []
        line_labels = []
        line_to_checkbox_idx = {}
        
        # Initial plot (all variables shown)
        line_idx = 0
        for idx, var in enumerate(var_names):
            data = history[var]
            color = colors[idx % len(colors)]
            label = f'{var[-30:]}...' if len(var) > 30 else var
            
            if value_type in ['mean', 'both']:
                mean_line, = ax.plot(data['snapshots'], data['means'], 
                                    '-', linewidth=linewidth, color=color, visible=True, picker=True)
                lines.append(mean_line)
                line_labels.append(f'{label} (mean)')
                line_to_checkbox_idx[mean_line] = line_idx
                line_idx += 1
            
            if value_type in ['std', 'both']:
                std_line, = ax.plot(data['snapshots'], data['stds'], 
                                ':', linewidth=linewidth, color=color, visible=True, picker=True)
                lines.append(std_line)
                line_labels.append(f'{label} (std)')
                line_to_checkbox_idx[std_line] = line_idx
                line_idx += 1
        
        # Set labels and title
        ax.set_xlabel('Snapshot ID', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        
        # Use custom title if provided, otherwise auto-generate
        if title:
            ax.set_title(title, fontsize=14)
        else:
            if value_type == 'both':
                ax.set_title(f'Interactive Variable Selection{title_suffix} - Click checkboxes to toggle', fontsize=14)
            else:
                ax.set_title(f'Interactive Variable Selection{title_suffix} - {value_type.capitalize()}', fontsize=14)
        
        ax.grid(True, alpha=0.3)
        
        # Create checkboxes axis
        checkbox_height = min(0.8, 0.035 * len(line_labels))
        checkbox_ax = fig.add_axes([0.02, 0.5 - checkbox_height/2, 0.3, checkbox_height])
        
        # Create checkboxes (all checked by default)
        check = CheckButtons(checkbox_ax, line_labels, [True] * len(line_labels))
        
        # Define callback function for checkboxes
        def toggle_visibility(label):
            # Find the index of the label
            idx = line_labels.index(label)
            # Toggle the corresponding line
            lines[idx].set_visible(not lines[idx].get_visible())
            # Redraw
            fig.canvas.draw_idle()
        
        check.on_clicked(toggle_visibility)
        
        # Also allow clicking on legend to toggle
        def on_pick(event):
            if event.artist in lines:
                idx = lines.index(event.artist)
                # Toggle the line
                lines[idx].set_visible(not lines[idx].get_visible())
                # Update checkbox
                check.set_active(idx)  # This toggles the checkbox
                fig.canvas.draw_idle()
        
        fig.canvas.mpl_connect('pick_event', on_pick)
        
        plt.show()
        
        return fig, check, lines, line_labels


    def plot_with_legend_selector(self, key_filters=None, match_all=True, value_type='mean', figsize=(16, 8), title=None, linewidth=0.5):
        """Plot with clickable legend to toggle variables - OPTIMIZED LAYOUT
        
        Args:
            key_filters: Optional list of strings to filter variable names
            match_all: If True, variable must contain ALL key_filters (AND)
            value_type: 'mean', 'std', or 'both' (default 'mean' for cleaner plot)
            figsize: Figure size (wider default to accommodate legend)
            title: Optional custom title (if None, auto-generated)
            linewidth: Width of the plot lines (default: 0.5)
        """
        if key_filters:
            history = self.filter_by_keys(key_filters, match_all=match_all)
            title_suffix = self.get_filter_description(key_filters, match_all)
        else:
            history = self.get_variable_history()
            title_suffix = ""
        
        if not history:
            filter_desc = self.get_filter_description(key_filters, match_all) if key_filters else "any"
            print(f"No variables found{filter_desc}")
            return
        
        # Limit number of variables to prevent overcrowding
        var_names = sorted(history.keys())
        if len(var_names) > 20:
            print(f"Found {len(var_names)} variables. Showing first 20 for clarity.")
            var_names = var_names[:20]
            # Rebuild history with limited vars
            history = {k: history[k] for k in var_names}
        
        # Create figure with adjusted layout
        fig = plt.figure(figsize=figsize)
        
        # Use gridspec for better control
        gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.05)
        ax = fig.add_subplot(gs[0])  # Main plot on left
        legend_ax = fig.add_subplot(gs[1])  # Legend on right
        legend_ax.axis('off')  # Hide axes of legend subplot
        
        # Use a colormap for lines
        colors = plt.cm.tab20(np.linspace(0, 1, len(history)))
        
        # Store plot lines and their corresponding legend lines
        plot_lines = []
        legend_lines = []
        legend_labels = []
        
        for idx, (var, data) in enumerate(history.items()):
            color = colors[idx % len(colors)]
            
            # Create short label (last 30 chars or less)
            if len(var) > 40:
                # Try to show meaningful parts
                parts = var.split('.')
                if len(parts) >= 3:
                    short_label = '.'.join(parts[-3:])  # Last 3 components
                else:
                    short_label = var[-30:]
            else:
                short_label = var
            
            # Plot the line
            if value_type == 'mean':
                line, = ax.plot(data['snapshots'], data['means'], 
                            '-', linewidth=linewidth, color=color, label=short_label, picker=True, pickradius=5)
            elif value_type == 'std':
                line, = ax.plot(data['snapshots'], data['stds'], 
                            ':', linewidth=linewidth, color=color, label=short_label, picker=True, pickradius=5)
            else:  # 'both'
                mean_line, = ax.plot(data['snapshots'], data['means'], 
                                    '-', linewidth=linewidth, color=color, 
                                    label=f'{short_label} (mean)', picker=True, pickradius=5)
                std_line, = ax.plot(data['snapshots'], data['stds'], 
                                ':', linewidth=linewidth, color=color, 
                                label=f'{short_label} (std)', picker=True, pickradius=5)
                plot_lines.extend([mean_line, std_line])
                continue
            
            plot_lines.append(line)
        
        # Create legend in the right subplot
        if value_type == 'both':
            # For 'both', we need to create custom legend entries
            from matplotlib.lines import Line2D
            legend_elements = []
            for i, var in enumerate(history.keys()):
                color = colors[i % len(colors)]
                short_label = var.split('.')[-1] if len(var) > 40 else var
                legend_elements.append(Line2D([0], [0], color=color, lw=linewidth, 
                                            label=f'{short_label} (mean)'))
                legend_elements.append(Line2D([0], [0], color=color, lw=linewidth, ls=':', 
                                            label=f'{short_label} (std)'))
            legend = legend_ax.legend(handles=legend_elements, loc='upper left', 
                                    fontsize=9, frameon=True, fancybox=True, shadow=True)
        else:
            legend = legend_ax.legend(loc='upper left', fontsize=9, 
                                    frameon=True, fancybox=True, shadow=True)
            # Get the legend lines
            legend_lines = legend.get_lines()
        
        # Store mapping from legend line to plot line
        legend_to_plot = {}
        if value_type != 'both':
            for leg_line, plot_line in zip(legend_lines, plot_lines):
                leg_line.set_picker(True)
                leg_line.set_pickradius(5)
                legend_to_plot[leg_line] = plot_line
        
        # Define pick event handler
        def on_pick(event):
            if value_type == 'both':
                # For 'both', we need to handle differently
                return
            
            legend_line = event.artist
            if legend_line in legend_to_plot:
                plot_line = legend_to_plot[legend_line]
                # Toggle visibility
                visible = not plot_line.get_visible()
                plot_line.set_visible(visible)
                # Change legend line alpha
                legend_line.set_alpha(1.0 if visible else 0.3)
                fig.canvas.draw_idle()
        
        fig.canvas.mpl_connect('pick_event', on_pick)
        
        # Set labels and title for main plot
        ax.set_xlabel('Snapshot ID', fontsize=12)
        ax.set_ylabel('Mean Value' if value_type == 'mean' else 
                    'Std Value' if value_type == 'std' else 'Value', fontsize=12)
        
        # Use custom title if provided, otherwise auto-generate
        if title:
            ax.set_title(title, fontsize=14)
        else:
            if value_type == 'mean':
                ax.set_title(f'Variables{title_suffix} - Mean Values (click legend to toggle)', fontsize=14)
            elif value_type == 'std':
                ax.set_title(f'Variables{title_suffix} - Std Values (click legend to toggle)', fontsize=14)
            else:
                ax.set_title(f'Variables{title_suffix} - Mean & Std (click legend to toggle)', fontsize=14)
        
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig

    def list_variables_interactive(self, key_filters=None, match_all=True):
        """Display an interactive list of variables with indices to select
        
        Args:
            key_filters: Optional list of strings to filter variable names
            match_all: If True, variable must contain ALL key_filters (AND)
        """
        if key_filters:
            history = self.filter_by_keys(key_filters, match_all=match_all)
            title = self.get_filter_description(key_filters, match_all)
        else:
            history = self.get_variable_history()
            title = ""
        
        if not history:
            filter_desc = self.get_filter_description(key_filters, match_all) if key_filters else "any"
            print(f"No variables found{filter_desc}")
            return None
        
        var_names = sorted(history.keys())
        
        print("\n" + "="*80)
        print(f"AVAILABLE VARIABLES{title}")
        print("="*80)
        print(f"Total: {len(var_names)} variables\n")
        
        # Display variables in columns
        from tabulate import tabulate
        
        # Create chunks of 3 columns
        table_data = []
        for i in range(0, len(var_names), 3):
            row = []
            for j in range(3):
                if i + j < len(var_names):
                    idx = i + j
                    row.append(f"[{idx}] {var_names[idx][:40]}")
                else:
                    row.append("")
            table_data.append(row)
        
        print(tabulate(table_data, tablefmt="grid"))
        
        print("\nEnter variable indices to plot (comma-separated), or 'q' to quit:")
        
        while True:
            user_input = input(">> ").strip()
            
            if user_input.lower() == 'q':
                return None
            
            try:
                # Parse indices
                indices = []
                for part in user_input.split(','):
                    part = part.strip()
                    if '-' in part:
                        # Handle range like "5-10"
                        start, end = map(int, part.split('-'))
                        indices.extend(range(start, end + 1))
                    else:
                        indices.append(int(part))
                
                # Remove duplicates and sort
                indices = sorted(set(indices))
                
                # Validate indices
                valid_indices = [i for i in indices if 0 <= i < len(var_names)]
                invalid_indices = [i for i in indices if i not in valid_indices]
                
                if invalid_indices:
                    print(f"Invalid indices: {invalid_indices}. Using valid indices only.")
                
                if not valid_indices:
                    print("No valid indices selected.")
                    continue
                
                selected_vars = [var_names[i] for i in valid_indices]
                
                print(f"\nSelected {len(selected_vars)} variables:")
                for i, var in enumerate(selected_vars):
                    print(f"  {i+1}. {var}")
                
                return selected_vars
                
            except ValueError:
                print("Invalid input. Please enter numbers separated by commas (e.g., '1,3,5' or '2-7')")


    def plot_selected_interactive(self, key_filters=None, match_all=True, value_type='both', figsize=(14, 8), title=None, linewidth=0.5):
        """Interactive workflow: list variables, let user select by index, then plot
        
        Args:
            key_filters: Optional list of strings to filter variable names
            match_all: If True, variable must contain ALL key_filters (AND)
            value_type: 'mean', 'std', or 'both'
            figsize: Figure size
            title: Optional custom title for the plot
            linewidth: Width of the plot lines (default: 0.5)
        """
        # First, list variables and get user selection
        selected_vars = self.list_variables_interactive(key_filters, match_all)
        
        if selected_vars is None:
            print("No variables selected.")
            return
        
        # Then plot the selected variables
        self.plot_variable_list(selected_vars, value_type=value_type, figsize=figsize, title=title, linewidth=linewidth)
        
        return selected_vars


# ========== USAGE WITH AND LOGIC ==========
if __name__ == "__main__":
    analyzer = SnapshotAnalyzer(snapshot_dir = 'd:/projects/zeus2/snapshots')
    analyzer.load_all_snapshots()
    
    # Print summary for all variables
    analyzer.print_summary()

    linewidth = 1.2

    for i in range(3, 9):
        for j in range(8):
            vlist = [f"B{i}.e.L{j}.sl_attn.weight", f"B{i}.d.L{j}.sl_attn.weight", f"B{i}.e.L{j}.sl_attn.bias", f"B{i}.d.L{j}.sl_attn.bias"]
            analyzer.plot_variable_list(vlist, title=f"attn skip_learer, B{i} L{j}", linewidth=linewidth)

    for i in range(3, 9):
        for j in range(8):
            vlist = [f"B{i}.e.L{j}.sl_ffn.weight", f"B{i}.d.L{j}.sl_ffn.weight", f"B{i}.e.L{j}.sl_ffn.bias", f"B{i}.d.L{j}.sl_ffn.bias"]
            analyzer.plot_variable_list(vlist, title=f"ffn skip_learer, B{i} L{j}", linewidth=linewidth)


    vlist = [f"B{i}.skip_learner.weight" for i in range(10)]
    analyzer.plot_variable_list(vlist, title="Skip_learners weight", linewidth=linewidth)

    vlist = [f"B{i}.skip_learner.bias" for i in range(10)]
    analyzer.plot_variable_list(vlist, title="Skip_learners bias", linewidth=linewidth)

    for i in range(10):
        vlist = [f"Be{i}_in", f"Bd{i}_in", f"Be{i}_out", f"Bd{i}_out"]
        analyzer.plot_variable_list(vlist, title=f"Block outputs {i}", linewidth=linewidth)

    for i in range(3, 10):
        for j in range(6):
            vlist = [f'B{i}.e.L{j}.h.mha.wq.weight', f'B{i}.e.L{j}.h.mha.wk.weight', f'B{i}.e.L{j}.h.mha.wv.weight',]
            analyzer.plot_variable_list(vlist, title=f"Q K V weights, B{i} Enc L{j}, h_axis", linewidth=linewidth)

    for i in range(3, 10):
        for j in range(6):
            vlist = [f'B{i}.e.L{j}.h.mha.wq.bias', f'B{i}.e.L{j}.h.mha.wk.bias', f'B{i}.e.L{j}.h.mha.wv.bias',]
            analyzer.plot_variable_list(vlist, title=f"Q K V bias B{i}, Enc L{j}, h_axis", linewidth=linewidth)


    for i in range(3, 10):
        vlist = [f'B{i}.e.L{j}.h.pre_act' for j in range(8)]
        analyzer.plot_variable_list(vlist, title=f"Pre_Act at B{i} L<j> Enc", linewidth=linewidth)

    for i in range(3, 10):
        vlist = [f'B{i}.e.L{j}.pre_act' for j in range(8)]
        analyzer.plot_variable_list(vlist, title=f"Pre_Act at B{i} L<j> Enc", linewidth=linewidth)

    for i in range(3, 10):
        vlist = [f'B{i}.d.L{j}.h.pre_act' for j in range(8)]
        analyzer.plot_variable_list(vlist, title=f"Pre_Act at B{i} L<j> Dec", linewidth=linewidth)

    for i in range(3, 10):
        for j in range(6):
            vlist = [f'B{i}.e.L{j}.h.pre_act', f'B{i}.d.L{j}.h.pre_act', f'B{i}.e.L{j}.w.pre_act', f'B{i}.d.L{j}.w.pre_act']
            analyzer.plot_variable_list(vlist, title=f"Pre_Act at B{i} L{j} Enc h Dec h Enc w Dec w", linewidth=linewidth)

    for i in range(3, 10):
        for j in range(6):
            vlist = [f'B{i}.e.L{j}.pre_act', f'B{i}.d.L{j}.pre_act']
            analyzer.plot_variable_list(vlist, title=f"Pre_Act at B{i} L{j} Enc Dec", linewidth=linewidth)

    for i in range(3, 10):
        for j in range(6):
            vlist = [f'B{i}.e.L{j}.pre_act', f'B{i}.d.L{j}.pre_act']
            analyzer.plot_variable_list(vlist, title=f"Pre_Act at B{i} L{j} Enc Dec", linewidth=linewidth)
    
    for i in range(3, 10):
        for j in range(6):
            vlist = [f'B{i}.e.L{j}.mha.wq.weight', f'B{i}.d.L{j}.mha.wq.bias', f'B{i}.e.L{j}.mha.wk.weight', f'B{i}.d.L{j}.mha.wk.bias', f'B{i}.e.L{j}.mha.w.weight', f'B{i}.d.L{j}.mha.wv.bias']
            analyzer.plot_variable_list(vlist, title=f"WQ, WK, WV at B{i} L{j} Enc Dec", linewidth=linewidth)
    
    # # Method 1: Interactive checkbox selector
    # analyzer.interactive_variable_selector(key_filters=['B1', 'mha'], value_type='both', linewidth=0.5)

    # # Method 2: Clickable legend (click on legend items to toggle)
    # analyzer.plot_with_legend_selector(key_filters=['weight'], value_type='mean', linewidth=0.5)

    # # Method 3: List variables with indices, then select by number
    # selected = analyzer.plot_selected_interactive(key_filters=['B1'], linewidth=0.5)
    # # Then enter indices like: 1,3,5 or 2-7

    # # Method 4: First list, then plot manually
    # vars_list = analyzer.list_variables_interactive(key_filters=['B1', 'L0'])
    # # After seeing the list, plot specific ones:
    # analyzer.plot_variable_list([vars_list[0], vars_list[2], vars_list[5]], linewidth=0.5)
    
    # # 2. Heatmap of key variables
    # analyzer.plot_all_heatmap(key_filters=search_keys, match_all=True)
    
    # # 3. Boxplot of key variables
    # analyzer.plot_all_boxplot(key_filters=search_keys, match_all=True)
    
    # # 4. Trajectories of key variables
    # analyzer.plot_all_trajectories(key_filters=search_keys, match_all=True)
    
    # # 5. Stacked area of key variables
    # analyzer.plot_all_stacked(key_filters=search_keys, match_all=True)
    
    # # 6. Parallel coordinates of key variables
    # analyzer.plot_all_parallel(key_filters=search_keys, match_all=True)
    
    # # 7. Statistical overview of key variables
    # analyzer.plot_all_stats(key_filters=search_keys, match_all=True)
    
    # # You can also use OR logic if needed:
    # # analyzer.plot_all_lines(key_filters=search_keys, match_all=False)
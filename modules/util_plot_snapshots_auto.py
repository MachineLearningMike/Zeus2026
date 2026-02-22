import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict
import re

class SnapshotAnalyzer:
    def __init__(self, snapshot_dir='./snapshots'):
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
    
    def plot_all_lines(self, key_filters=None, match_all=True, value_type='both', figsize=(15, 8)):
        """Plot variables as lines, optionally filtered by keys anywhere in name
        
        Args:
            key_filters: Optional list of strings to filter variable names
            match_all: If True, variable must contain ALL key_filters (AND)
            value_type: 'mean', 'std', or 'both' (default)
            figsize: Figure size
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
                        '-', linewidth=2, markersize=4, marker='o',
                        color=color, label=f'{label} (mean)' if value_type == 'both' else label)
            
            if value_type in ['std', 'both']:
                linestyle = '--' if value_type == 'both' else '-'
                linewidth = 1.5 if value_type == 'both' else 2
                markersize = 3 if value_type == 'both' else 4
                marker = 's' if value_type == 'both' else 'o'
                alpha = 0.7 if value_type == 'both' else 1.0
                
                plt.plot(data['snapshots'], data['stds'], 
                        linestyle, linewidth=linewidth, markersize=markersize, marker=marker,
                        color=color, alpha=alpha, label=f'{label} (std)' if value_type == 'both' else None)
        
        plt.xlabel('Snapshot ID', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        
        if value_type == 'both':
            plt.title(f'Variables{title_suffix} - Mean (solid) vs Std (dashed) ({len(history)} variables)', fontsize=14)
        else:
            plt.title(f'Variables{title_suffix} - {value_type.capitalize()} ({len(history)} variables)', fontsize=14)
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_all_heatmap(self, key_filters=None, match_all=True, figsize=(16, 10)):
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
        ax.set_title(f'Variables Heatmap{title_suffix}\n({len(var_names)} variables, {len(all_ss)} snapshots)', 
                    fontsize=14)
        
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
    
    def plot_all_boxplot(self, key_filters=None, match_all=True, figsize=(15, 8)):
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
        
        plt.xticks(rotation=90, fontsize=8)
        plt.ylabel('Mean Value', fontsize=12)
        plt.title(f'Distribution of Variables{title_suffix} ({len(var_names)} total)', fontsize=14)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.show()
    
    def plot_all_trajectories(self, key_filters=None, match_all=True, figsize=(15, 10)):
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
                        'o-', linewidth=1, markersize=2,
                        color=color, alpha=0.7, label=var[-30:] if len(var) > 30 else var)
            
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            plt.colorbar(sm, label='Final Mean Value')
        
        plt.xlabel('Snapshot ID', fontsize=12)
        plt.ylabel('Mean Value', fontsize=12)
        plt.title(f'Variable Trajectories{title_suffix}', fontsize=14)
        
        plt.grid(True, alpha=0.3)
        if len(history) < 20:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.tight_layout()
        plt.show()
    
    def plot_all_stacked(self, key_filters=None, match_all=True, figsize=(12, 12)):
        """Stacked area plot, filtered by keys anywhere in name"""
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
                     alpha=0.7, edgecolor='none')
        
        plt.xlabel('Snapshot ID', fontsize=12)
        plt.ylabel('Cumulative Mean Value', fontsize=12)
        plt.title(f'Stacked Contribution of Variables{title_suffix}', fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_all_parallel(self, key_filters=None, match_all=True, figsize=(14, 8)):
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
            plt.plot(range(len(all_ss)), data_norm[i], 'o-', 
                    linewidth=1, alpha=0.7,
                    label=var_name[-30:] if len(var_name) > 30 else var_name)
        
        plt.xticks(range(len(all_ss)), all_ss)
        plt.xlabel('Snapshot ID', fontsize=12)
        plt.ylabel('Normalized Value', fontsize=12)
        plt.title(f'Parallel Coordinates Plot{title_suffix}', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        if len(history) < 20:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.tight_layout()
        plt.show()
    
    def plot_all_stats(self, key_filters=None, match_all=True, figsize=(15, 10)):
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
        axes[0, 0].hist([s['mean'] for s in stats.values()], bins=30, alpha=0.7)
        axes[0, 0].set_xlabel('Mean Value')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title(f'Distribution of Means{title_suffix}')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Distribution of stds
        axes[0, 1].hist([s['std'] for s in stats.values()], bins=30, alpha=0.7, color='green')
        axes[0, 1].set_xlabel('Std Value')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title(f'Distribution of Standard Deviations{title_suffix}')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Mean vs Std scatter
        axes[0, 2].scatter([s['mean'] for s in stats.values()], 
                          [s['std'] for s in stats.values()], 
                          alpha=0.6)
        axes[0, 2].set_xlabel('Mean')
        axes[0, 2].set_ylabel('Std')
        axes[0, 2].set_title(f'Mean vs Std{title_suffix}')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Range distribution
        axes[1, 0].hist([s['range'] for s in stats.values()], bins=30, alpha=0.7, color='red')
        axes[1, 0].set_xlabel('Range (Max - Min)')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title(f'Distribution of Ranges{title_suffix}')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Trend distribution
        axes[1, 1].hist([s['trend'] for s in stats.values()], bins=30, alpha=0.7, color='purple')
        axes[1, 1].set_xlabel('Trend (Final - Initial)')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title(f'Distribution of Trends{title_suffix}')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Top variables by mean magnitude
        top_vars = sorted(stats.items(), key=lambda x: abs(x[1]['mean']), reverse=True)[:10]
        var_names = [v[1]['short_name'] for v in top_vars]
        var_means = [v[1]['mean'] for v in top_vars]
        
        axes[1, 2].barh(range(len(var_names)), var_means, alpha=0.7, color='orange')
        axes[1, 2].set_yticks(range(len(var_names)))
        axes[1, 2].set_yticklabels(var_names)
        axes[1, 2].set_xlabel('Mean Value')
        axes[1, 2].set_title('Top 10 Variables by |Mean|')
        axes[1, 2].grid(True, alpha=0.3)
        
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


# ========== USAGE WITH AND LOGIC ==========
if __name__ == "__main__":
    analyzer = SnapshotAnalyzer()
    analyzer.load_all_snapshots()
    
    # Print summary for all variables
    analyzer.print_summary()
    
    # Define multiple search keys - MUST contain ALL of them (AND logic)
    search_keys = ['B1', '']  # Variable must contain BOTH 'sc_b' AND 'B3'
    value_type = 'both' # 'both', 'mean', 'std'
    
    # Print summary for variables containing ALL keys
    analyzer.print_summary(key_filters=search_keys, match_all=True)
    
    # Plot only variables containing ALL keys:
    
    # 1. All variables with keys as lines
    analyzer.plot_all_lines(key_filters=search_keys, match_all=True, value_type=value_type)
    
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
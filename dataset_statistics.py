#!/usr/bin/env python3
"""
Dataset Statistics and Measurement Repeatability Analysis
==========================================================

Analyzes the robocasting dataset to quantify:
- Measurement repeatability (within-sample variation)
- Distribution of process parameters
- Correlations between inputs and outputs
- Baseline uncertainty that limits model performance

Author: Nazarii
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Configure plotting
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10


class DatasetAnalyzer:
    """Comprehensive dataset statistics and quality analysis."""
    
    def __init__(self, data_path, output_dir='dataset_analysis'):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.df = None
        
    def load_data(self):
        """Load data."""
        print("=" * 80)
        print("LOADING DATA")
        print("=" * 80)
        
        self.df = pd.read_csv(self.data_path)
        print(f"\nDataset shape: {self.df.shape}")
        print(f"Number of samples: {len(self.df)}")
        
    def analyze_measurement_repeatability(self):
        """Analyze within-sample measurement variation."""
        print("\n" + "=" * 80)
        print("MEASUREMENT REPEATABILITY ANALYSIS")
        print("=" * 80)
        
        # Calculate individual measurements
        heights = self.df[['height_1', 'height_2', 'height_3']].values
        widths = self.df[['width_1', 'width_2', 'width_3']].values
        
        # Calculate statistics for each sample
        height_means = np.mean(heights, axis=1)
        height_stds = np.std(heights, axis=1, ddof=1)
        height_ranges = np.max(heights, axis=1) - np.min(heights, axis=1)
        height_cv = (height_stds / height_means) * 100  # Coefficient of variation (%)
        
        width_means = np.mean(widths, axis=1)
        width_stds = np.std(widths, axis=1, ddof=1)
        width_ranges = np.max(widths, axis=1) - np.min(widths, axis=1)
        width_cv = (width_stds / width_means) * 100
        
        print("\n" + "-" * 80)
        print("HEIGHT MEASUREMENTS")
        print("-" * 80)
        print(f"Mean height: {height_means.mean():.3f} ± {height_means.std():.3f} mm")
        print(f"\nWithin-sample variation:")
        print(f"  Average std dev:  {height_stds.mean():.4f} mm")
        print(f"  Std dev range:    [{height_stds.min():.4f}, {height_stds.max():.4f}] mm")
        print(f"  Average range:    {height_ranges.mean():.4f} mm")
        print(f"  Max range:        {height_ranges.max():.4f} mm")
        print(f"  Avg CV:           {height_cv.mean():.2f}%")
        print(f"\nRelative to signal:")
        print(f"  SNR (mean/std):   {height_means.mean() / height_stds.mean():.1f}")
        
        print("\n" + "-" * 80)
        print("WIDTH MEASUREMENTS")
        print("-" * 80)
        print(f"Mean width: {width_means.mean():.3f} ± {width_means.std():.3f} mm")
        print(f"\nWithin-sample variation:")
        print(f"  Average std dev:  {width_stds.mean():.4f} mm")
        print(f"  Std dev range:    [{width_stds.min():.4f}, {width_stds.max():.4f}] mm")
        print(f"  Average range:    {width_ranges.mean():.4f} mm")
        print(f"  Max range:        {width_ranges.max():.4f} mm")
        print(f"  Avg CV:           {width_cv.mean():.2f}%")
        print(f"\nRelative to signal:")
        print(f"  SNR (mean/std):   {width_means.mean() / width_stds.mean():.1f}")
        
        # Compare height vs width measurement quality
        print("\n" + "-" * 80)
        print("COMPARISON: HEIGHT vs WIDTH")
        print("-" * 80)
        print(f"Height CV: {height_cv.mean():.2f}% ± {height_cv.std():.2f}%")
        print(f"Width CV:  {width_cv.mean():.2f}% ± {width_cv.std():.2f}%")
        
        if width_cv.mean() > height_cv.mean():
            ratio = width_cv.mean() / height_cv.mean()
            print(f"\n⚠️  Width measurements are {ratio:.1f}x more variable than height")
            print("   This explains why width models have lower R²!")
        
        # Statistical test
        t_stat, p_value = stats.ttest_ind(height_cv, width_cv)
        print(f"\nt-test: p-value = {p_value:.4f}")
        if p_value < 0.05:
            print("   ✅ Measurement variability is significantly different")
        else:
            print("   → No significant difference in measurement variability")
        
        # Create visualization
        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        
        # Row 1: Individual measurements
        ax = axes[0, 0]
        for i in range(len(self.df)):
            ax.plot([1, 2, 3], heights[i], 'o-', alpha=0.3, color='blue')
        ax.set_xlabel('Measurement Number', fontweight='bold')
        ax.set_ylabel('Height (mm)', fontweight='bold')
        ax.set_title('Height: Individual Measurements per Sample', fontweight='bold')
        ax.set_xticks([1, 2, 3])
        ax.grid(alpha=0.3)
        
        ax = axes[0, 1]
        for i in range(len(self.df)):
            ax.plot([1, 2, 3], widths[i], 'o-', alpha=0.3, color='green')
        ax.set_xlabel('Measurement Number', fontweight='bold')
        ax.set_ylabel('Width (mm)', fontweight='bold')
        ax.set_title('Width: Individual Measurements per Sample', fontweight='bold')
        ax.set_xticks([1, 2, 3])
        ax.grid(alpha=0.3)
        
        ax = axes[0, 2]
        ax.boxplot([height_cv, width_cv], labels=['Height', 'Width'])
        ax.set_ylabel('Coefficient of Variation (%)', fontweight='bold')
        ax.set_title('Measurement Variability Comparison', fontweight='bold')
        ax.grid(alpha=0.3)
        
        # Row 2: Standard deviation distributions
        ax = axes[1, 0]
        ax.hist(height_stds, bins=20, alpha=0.7, edgecolor='black', color='blue')
        ax.axvline(height_stds.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {height_stds.mean():.4f}')
        ax.set_xlabel('Standard Deviation (mm)', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('Height: Within-Sample Std Dev Distribution', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        ax = axes[1, 1]
        ax.hist(width_stds, bins=20, alpha=0.7, edgecolor='black', color='green')
        ax.axvline(width_stds.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {width_stds.mean():.4f}')
        ax.set_xlabel('Standard Deviation (mm)', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('Width: Within-Sample Std Dev Distribution', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        ax = axes[1, 2]
        ax.scatter(height_means, height_stds, alpha=0.6, edgecolors='k', linewidth=0.5, label='Height')
        ax.scatter(width_means, width_stds, alpha=0.6, edgecolors='k', linewidth=0.5, label='Width')
        ax.set_xlabel('Mean Value (mm)', fontweight='bold')
        ax.set_ylabel('Std Dev (mm)', fontweight='bold')
        ax.set_title('Heteroscedasticity Check', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Row 3: Range distributions
        ax = axes[2, 0]
        ax.hist(height_ranges, bins=20, alpha=0.7, edgecolor='black', color='blue')
        ax.axvline(height_ranges.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {height_ranges.mean():.4f}')
        ax.set_xlabel('Range (max - min) (mm)', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('Height: Measurement Range Distribution', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        ax = axes[2, 1]
        ax.hist(width_ranges, bins=20, alpha=0.7, edgecolor='black', color='green')
        ax.axvline(width_ranges.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {width_ranges.mean():.4f}')
        ax.set_xlabel('Range (max - min) (mm)', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('Width: Measurement Range Distribution', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Bland-Altman style plot
        ax = axes[2, 2]
        height_diff_12 = heights[:, 0] - heights[:, 1]
        height_avg_12 = (heights[:, 0] + heights[:, 1]) / 2
        ax.scatter(height_avg_12, height_diff_12, alpha=0.6, label='Height (1 vs 2)', edgecolors='k', linewidth=0.5)
        ax.axhline(0, color='black', linestyle='-', linewidth=1)
        ax.axhline(height_diff_12.mean(), color='red', linestyle='--', linewidth=2)
        ax.axhline(height_diff_12.mean() + 1.96*height_diff_12.std(), color='red', linestyle=':', linewidth=2)
        ax.axhline(height_diff_12.mean() - 1.96*height_diff_12.std(), color='red', linestyle=':', linewidth=2)
        ax.set_xlabel('Average of Two Measurements (mm)', fontweight='bold')
        ax.set_ylabel('Difference (mm)', fontweight='bold')
        ax.set_title('Bland-Altman: Measurement Agreement', fontweight='bold')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'measurement_repeatability.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✅ Visualization saved: measurement_repeatability.png")
        
        # Save the three specific plots as separate images
        self._save_individual_plots(height_stds, width_stds, height_means, width_means)
        
        # Save statistics to file
        stats_df = pd.DataFrame({
            'Metric': [
                'Mean', 'Std Dev of Means', 
                'Avg Within-Sample Std', 'Max Within-Sample Std',
                'Avg Range', 'Max Range',
                'Avg CV (%)', 'SNR'
            ],
            'Height': [
                f"{height_means.mean():.3f}",
                f"{height_means.std():.3f}",
                f"{height_stds.mean():.4f}",
                f"{height_stds.max():.4f}",
                f"{height_ranges.mean():.4f}",
                f"{height_ranges.max():.4f}",
                f"{height_cv.mean():.2f}",
                f"{height_means.mean() / height_stds.mean():.1f}"
            ],
            'Width': [
                f"{width_means.mean():.3f}",
                f"{width_means.std():.3f}",
                f"{width_stds.mean():.4f}",
                f"{width_stds.max():.4f}",
                f"{width_ranges.mean():.4f}",
                f"{width_ranges.max():.4f}",
                f"{width_cv.mean():.2f}",
                f"{width_means.mean() / width_stds.mean():.1f}"
            ]
        })
        
        stats_df.to_csv(self.output_dir / 'measurement_statistics.csv', index=False)
        print(f"✅ Statistics saved: measurement_statistics.csv")
        
        return height_stds.mean(), width_stds.mean()
    
    def _save_individual_plots(self, height_stds, width_stds, height_means, width_means):
        """Save three specific plots as individual images."""
        
        # Plot 1: Height Std Dev Distribution
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(height_stds, bins=20, alpha=0.7, edgecolor='black', color='blue')
        ax.axvline(height_stds.mean(), color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {height_stds.mean():.4f}')
        ax.set_xlabel('Standard Deviation (mm)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Frequency', fontweight='bold', fontsize=12)
        ax.set_title('Height: Within-Sample Std Dev Distribution', fontweight='bold', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'height_std_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: height_std_distribution.png")
        
        # Plot 2: Width Std Dev Distribution
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(width_stds, bins=20, alpha=0.7, edgecolor='black', color='green')
        ax.axvline(width_stds.mean(), color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {width_stds.mean():.4f}')
        ax.set_xlabel('Standard Deviation (mm)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Frequency', fontweight='bold', fontsize=12)
        ax.set_title('Width: Within-Sample Std Dev Distribution', fontweight='bold', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'width_std_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: width_std_distribution.png")
        
        # Plot 3: Heteroscedasticity Check
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(height_means, height_stds, alpha=0.6, edgecolors='k', 
                  linewidth=0.5, label='Height', s=80, color='blue')
        ax.scatter(width_means, width_stds, alpha=0.6, edgecolors='k', 
                  linewidth=0.5, label='Width', s=80, color='green')
        ax.set_xlabel('Mean Value (mm)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Std Dev (mm)', fontweight='bold', fontsize=12)
        ax.set_title('Heteroscedasticity Check', fontweight='bold', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'heteroscedasticity_check.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: heteroscedasticity_check.png")
    
    def analyze_parameter_distributions(self):
        """Analyze distribution of process parameters."""
        print("\n" + "=" * 80)
        print("PROCESS PARAMETER DISTRIBUTIONS")
        print("=" * 80)
        
        # Define parameters
        params = ['slicer_nozzle_speed', 'slicer_extrusion_multiplier', 'layer_count',
                 'temp', 'humidity', 'nozzle_diameter']
        available_params = [p for p in params if p in self.df.columns]
        
        print(f"\nAnalyzing {len(available_params)} parameters:")
        
        summary_data = []
        for param in available_params:
            values = self.df[param]
            summary_data.append({
                'Parameter': param,
                'Mean': values.mean(),
                'Std': values.std(),
                'Min': values.min(),
                'Max': values.max(),
                'Unique Values': values.nunique(),
                'Range': values.max() - values.min()
            })
            
            print(f"\n{param}:")
            print(f"  Range: [{values.min():.3f}, {values.max():.3f}]")
            print(f"  Mean ± Std: {values.mean():.3f} ± {values.std():.3f}")
            print(f"  Unique values: {values.nunique()}")
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(self.output_dir / 'parameter_summary.csv', index=False)
        
        # Visualize
        n_params = len(available_params)
        n_cols = 3
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        axes = axes.flatten() if n_params > 1 else [axes]
        
        for i, param in enumerate(available_params):
            ax = axes[i]
            values = self.df[param]
            
            if values.nunique() < 10:
                # Categorical/discrete
                value_counts = values.value_counts().sort_index()
                ax.bar(range(len(value_counts)), value_counts.values, 
                      tick_label=value_counts.index, alpha=0.7, edgecolor='black')
                ax.set_ylabel('Frequency', fontweight='bold')
            else:
                # Continuous
                ax.hist(values, bins=20, alpha=0.7, edgecolor='black')
                ax.set_ylabel('Frequency', fontweight='bold')
            
            ax.set_xlabel(param.replace('_', ' ').title(), fontweight='bold')
            ax.set_title(f'{param}\n(n={values.nunique()} unique)', fontweight='bold')
            ax.grid(alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_params, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'parameter_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✅ Parameter distributions saved")
    
    def analyze_correlations(self):
        """Analyze correlations between parameters and outputs."""
        print("\n" + "=" * 80)
        print("CORRELATION ANALYSIS")
        print("=" * 80)
        
        # Calculate averages
        self.df['height_average'] = (self.df['height_1'] + self.df['height_2'] + self.df['height_3']) / 3
        self.df['width_average'] = (self.df['width_1'] + self.df['width_2'] + self.df['width_3']) / 3
        
        # Select relevant columns (exclude constant variables)
        features = ['slicer_nozzle_speed', 'slicer_extrusion_multiplier', 'layer_count']
        targets = ['height_average', 'width_average']
        
        # Only include features that exist and have variation
        available_features = []
        for f in features:
            if f in self.df.columns:
                if self.df[f].nunique() > 1:  # Has variation
                    available_features.append(f)
                else:
                    print(f"\n⚠️  Excluding {f} (no variation: all values = {self.df[f].iloc[0]})")
        
        cols = available_features + targets
        
        corr_matrix = self.df[cols].corr()
        
        # Print key correlations
        print("\nCorrelations with HEIGHT:")
        height_corr = corr_matrix['height_average'].drop('height_average').sort_values(ascending=False)
        for param, corr in height_corr.items():
            print(f"  {param:30s}: {corr:+.3f}")
        
        print("\nCorrelations with WIDTH:")
        width_corr = corr_matrix['width_average'].drop('width_average').sort_values(ascending=False)
        for param, corr in width_corr.items():
            print(f"  {param:30s}: {corr:+.3f}")
        
        # Visualize
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, vmin=-1, vmax=1, square=True, ax=ax,
                   cbar_kws={'label': 'Correlation Coefficient'})
        ax.set_title('Correlation Matrix: Parameters vs Outputs', fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        corr_matrix.to_csv(self.output_dir / 'correlation_matrix.csv')
        print(f"\n✅ Correlation analysis saved")
    
    def analyze_width_height_correlation(self):
        """Analyze correlation between average width and height."""
        print("\n" + "=" * 80)
        print("WIDTH vs HEIGHT CORRELATION")
        print("=" * 80)
        
        # Ensure averages are calculated
        if 'height_average' not in self.df.columns:
            self.df['height_average'] = (self.df['height_1'] + self.df['height_2'] + self.df['height_3']) / 3
        if 'width_average' not in self.df.columns:
            self.df['width_average'] = (self.df['width_1'] + self.df['width_2'] + self.df['width_3']) / 3
        
        height = self.df['height_average'].values
        width = self.df['width_average'].values
        # Use layer_count for color coding
        layer_count = None
        if 'layer_count' in self.df.columns:
            layer_count = self.df['layer_count'].values
            print(f"\nUsing 'layer_count' for color coding")
        else:
            print("\n⚠️  Layer count column not found in dataset")
        
        # Calculate correlation
        corr_coef = np.corrcoef(height, width)[0, 1]
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(height, width)
        
        print(f"\nCorrelation coefficient: {corr_coef:.4f}")
        print(f"R² value: {r_value**2:.4f}")
        print(f"p-value: {p_value:.6f}")
        print(f"\nLinear fit: width = {slope:.4f} * height + {intercept:.4f}")
        print(f"Standard error: {std_err:.4f}")
        
        if p_value < 0.001:
            print("\n✅ Highly significant correlation (p < 0.001)")
        elif p_value < 0.05:
            print("\n✅ Significant correlation (p < 0.05)")
        else:
            print("\n⚠️  No significant correlation")
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Scatter plot colored by layer count
        if layer_count is not None:
            scatter = ax.scatter(height, width, c=layer_count, cmap='viridis', 
                               alpha=0.7, s=100, edgecolors='k', linewidth=0.5)
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Layer Count', fontweight='bold', fontsize=11)
            
            # Print layer count info
            unique_layers = np.unique(layer_count)
            print(f"\nLayer counts in dataset: {unique_layers}")
            print(f"Number of unique layer counts: {len(unique_layers)}")
        else:
            scatter = ax.scatter(height, width, alpha=0.6, s=100, 
                               edgecolors='k', linewidth=0.5, color='steelblue')
            print("\n⚠️  Layer count column not found in dataset")
        
        # Formatting
        ax.set_xlabel('Average Height (mm)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Average Width (mm)', fontweight='bold', fontsize=12)
        ax.set_title(f'Width vs Height Correlation\n(r = {corr_coef:.4f}, p = {p_value:.6f})', 
                    fontweight='bold', fontsize=14)
        ax.grid(alpha=0.3)
        
        # Add text box with statistics
        textstr = f'Pearson r = {corr_coef:.4f}\nn = {len(height)} samples'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'width_height_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✅ Width-Height correlation plot saved: width_height_correlation.png")
        
        # Save statistics
        stats_dict = {
            'Correlation_coefficient': corr_coef,
            'R_squared': r_value**2,
            'p_value': p_value,
            'Slope': slope,
            'Intercept': intercept,
            'Std_error': std_err,
            'n_samples': len(height)
        }
        
        stats_df = pd.DataFrame([stats_dict])
        stats_df.to_csv(self.output_dir / 'width_height_correlation_stats.csv', index=False)
        print(f"✅ Statistics saved: width_height_correlation_stats.csv")
        
        return corr_coef, r_value**2


def main():
    """Main execution."""
    print("\n" + "="*80)
    print("DATASET STATISTICS AND QUALITY ANALYSIS")
    print("="*80 + "\n")
    
    data_path = '/Users/nazarii/projects/ipms/robocasting/data/cleaned_df.csv'
    analyzer = DatasetAnalyzer(data_path, output_dir='dataset_analysis')
    
    try:
        # Load data
        analyzer.load_data()
        
        # Measurement repeatability
        height_error, width_error = analyzer.analyze_measurement_repeatability()
        
        # Parameter distributions
        analyzer.analyze_parameter_distributions()
        
        # Correlations
        analyzer.analyze_correlations()
        
        # Width vs Height correlation
        corr_coef, r_squared = analyzer.analyze_width_height_correlation()
        
        print("\n" + "="*80)
        print("✅ ANALYSIS COMPLETE!")
        print("="*80)
        print(f"\nResults in: {analyzer.output_dir}")
        print("\n📊 Key findings:")
        print(f"   Average measurement error (height): {height_error:.4f} mm")
        print(f"   Average measurement error (width):  {width_error:.4f} mm")
        print("\n📝 For paper:")
        print("   'Measurement repeatability was assessed using three")
        print("   independent measurements per sample. Within-sample")
        print(f"   standard deviation was {height_error:.4f} mm for height")
        print(f"   and {width_error:.4f} mm for width.")
        print(f"   Width and height showed a correlation of r = {corr_coef:.3f}")
        print(f"   (R² = {r_squared:.3f}).'") 
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

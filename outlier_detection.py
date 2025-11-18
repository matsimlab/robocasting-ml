#!/usr/bin/env python3
"""
Outlier Detection Analysis for Robocasting Dataset
===================================================

Applies multiple outlier detection methods to identify anomalous samples.
Two methods are used:
1. Interquartile Range (IQR) method
2. Z-score method

Author: Nazarii
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configure plotting
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


def iqr_outliers(data, multiplier=1.5):
    """Detect outliers using IQR method."""
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_fence = Q1 - multiplier * IQR
    upper_fence = Q3 + multiplier * IQR
    
    outlier_mask = (data < lower_fence) | (data > upper_fence)
    
    return {
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'lower_fence': lower_fence,
        'upper_fence': upper_fence,
        'outlier_indices': data[outlier_mask].index.tolist(),
        'outlier_values': data[outlier_mask].values,
        'n_outliers': outlier_mask.sum()
    }


def zscore_outliers(data, threshold=3.0):
    """Detect outliers using Z-score method."""
    mean = data.mean()
    std = data.std()
    
    z_scores = np.abs((data - mean) / std)
    outlier_mask = z_scores > threshold
    
    return {
        'mean': mean,
        'std': std,
        'threshold': threshold,
        'outlier_indices': data[outlier_mask].index.tolist(),
        'outlier_values': data[outlier_mask].values,
        'z_scores': z_scores[outlier_mask].values,
        'n_outliers': outlier_mask.sum()
    }


def analyze_outliers(data_path, output_dir='dataset_analysis'):
    """Comprehensive outlier analysis."""
    
    print("\n" + "="*80)
    print("OUTLIER DETECTION ANALYSIS")
    print("="*80)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    df = pd.read_csv(data_path)
    print(f"\nDataset: {len(df)} samples")
    
    # Calculate averages
    df['height_average'] = (df['height_1'] + df['height_2'] + df['height_3']) / 3
    df['width_average'] = (df['width_1'] + df['width_2'] + df['width_3']) / 3
    
    # Analyze both dimensions
    results = {}
    
    for dimension in ['height_average', 'width_average']:
        dim_name = dimension.replace('_average', '').upper()
        print(f"\n{'-'*80}")
        print(f"{dim_name} OUTLIER ANALYSIS")
        print('-'*80)
        
        data = df[dimension]
        
        # Method 1: IQR
        print(f"\nMethod 1: Interquartile Range (IQR) with 1.5×IQR threshold")
        iqr_result = iqr_outliers(data, multiplier=1.5)
        
        print(f"  Q1 = {iqr_result['Q1']:.3f} mm")
        print(f"  Q3 = {iqr_result['Q3']:.3f} mm")
        print(f"  IQR = {iqr_result['IQR']:.3f} mm")
        print(f"  Lower fence = {iqr_result['lower_fence']:.3f} mm")
        print(f"  Upper fence = {iqr_result['upper_fence']:.3f} mm")
        print(f"  Data range: [{data.min():.3f}, {data.max():.3f}] mm")
        print(f"  Outliers detected: {iqr_result['n_outliers']}")
        
        if iqr_result['n_outliers'] > 0:
            print(f"  Outlier indices: {iqr_result['outlier_indices']}")
            print(f"  Outlier values: {iqr_result['outlier_values']}")
        
        # Method 2: Z-score
        print(f"\nMethod 2: Z-score with threshold = 3.0")
        zscore_result = zscore_outliers(data, threshold=3.0)
        
        print(f"  Mean = {zscore_result['mean']:.3f} mm")
        print(f"  Std Dev = {zscore_result['std']:.3f} mm")
        print(f"  Threshold = {zscore_result['threshold']:.1f} standard deviations")
        print(f"  Outliers detected: {zscore_result['n_outliers']}")
        
        if zscore_result['n_outliers'] > 0:
            print(f"  Outlier indices: {zscore_result['outlier_indices']}")
            print(f"  Outlier values: {zscore_result['outlier_values']}")
            print(f"  Z-scores: {zscore_result['z_scores']}")
        
        results[dimension] = {
            'iqr': iqr_result,
            'zscore': zscore_result
        }
    
    # Find samples that are outliers in BOTH dimensions by BOTH methods
    print("\n" + "="*80)
    print("CONSENSUS OUTLIER ANALYSIS")
    print("="*80)
    
    iqr_height_outliers = set(results['height_average']['iqr']['outlier_indices'])
    iqr_width_outliers = set(results['width_average']['iqr']['outlier_indices'])
    
    zscore_height_outliers = set(results['height_average']['zscore']['outlier_indices'])
    zscore_width_outliers = set(results['width_average']['zscore']['outlier_indices'])
    
    # IQR: outliers in both dimensions
    iqr_both = iqr_height_outliers & iqr_width_outliers
    print(f"\nIQR Method: Samples flagged in BOTH height AND width: {len(iqr_both)}")
    if iqr_both:
        print(f"  Indices: {sorted(iqr_both)}")
    
    # Z-score: outliers in both dimensions
    zscore_both = zscore_height_outliers & zscore_width_outliers
    print(f"\nZ-score Method: Samples flagged in BOTH height AND width: {len(zscore_both)}")
    if zscore_both:
        print(f"  Indices: {sorted(zscore_both)}")
    
    # Consensus: flagged by both methods
    consensus_outliers = iqr_both | zscore_both
    print(f"\nConsensus outliers (flagged by either method in both dimensions): {len(consensus_outliers)}")
    if consensus_outliers:
        print(f"  Indices: {sorted(consensus_outliers)}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    for idx, dimension in enumerate(['height_average', 'width_average']):
        dim_name = dimension.replace('_average', '').title()
        data = df[dimension]
        
        iqr_res = results[dimension]['iqr']
        zscore_res = results[dimension]['zscore']
        
        # Box plot with IQR fences
        ax = axes[idx, 0]
        bp = ax.boxplot([data], widths=0.5, patch_artist=True,
                        boxprops=dict(facecolor='lightblue', alpha=0.7),
                        medianprops=dict(color='red', linewidth=2))
        ax.axhline(iqr_res['lower_fence'], color='orange', linestyle='--', 
                  linewidth=2, label='IQR fences')
        ax.axhline(iqr_res['upper_fence'], color='orange', linestyle='--', linewidth=2)
        ax.set_ylabel(f'{dim_name} (mm)', fontweight='bold')
        ax.set_title(f'{dim_name}: Box Plot with IQR Fences', fontweight='bold')
        ax.set_xticks([])
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Histogram with IQR fences
        ax = axes[idx, 1]
        ax.hist(data, bins=20, alpha=0.7, edgecolor='black', color='steelblue')
        ax.axvline(iqr_res['lower_fence'], color='orange', linestyle='--', 
                  linewidth=2, label='IQR fences')
        ax.axvline(iqr_res['upper_fence'], color='orange', linestyle='--', linewidth=2)
        ax.axvline(data.mean(), color='red', linestyle='-', linewidth=2, label='Mean')
        ax.set_xlabel(f'{dim_name} (mm)', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title(f'{dim_name}: Distribution with IQR Method', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Z-score plot
        ax = axes[idx, 2]
        z_scores = np.abs((data - zscore_res['mean']) / zscore_res['std'])
        sample_indices = np.arange(len(data))
        
        colors = ['red' if z > 3.0 else 'blue' for z in z_scores]
        ax.scatter(sample_indices, z_scores, c=colors, alpha=0.6, edgecolors='k', linewidth=0.5)
        ax.axhline(3.0, color='orange', linestyle='--', linewidth=2, label='Z = 3.0 threshold')
        ax.set_xlabel('Sample Index', fontweight='bold')
        ax.set_ylabel('|Z-score|', fontweight='bold')
        ax.set_title(f'{dim_name}: Z-score Analysis', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'outlier_detection.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Visualization saved: outlier_detection.png")
    
    # Save summary report
    summary = []
    summary.append("OUTLIER DETECTION SUMMARY")
    summary.append("=" * 80)
    summary.append(f"\nDataset: {len(df)} samples")
    summary.append("\n" + "-" * 80)
    summary.append("HEIGHT ANALYSIS")
    summary.append("-" * 80)
    summary.append(f"\nIQR Method (1.5×IQR):")
    summary.append(f"  Outliers detected: {results['height_average']['iqr']['n_outliers']}")
    summary.append(f"\nZ-score Method (threshold = 3.0):")
    summary.append(f"  Outliers detected: {results['height_average']['zscore']['n_outliers']}")
    
    summary.append("\n" + "-" * 80)
    summary.append("WIDTH ANALYSIS")
    summary.append("-" * 80)
    summary.append(f"\nIQR Method (1.5×IQR):")
    summary.append(f"  Outliers detected: {results['width_average']['iqr']['n_outliers']}")
    summary.append(f"\nZ-score Method (threshold = 3.0):")
    summary.append(f"  Outliers detected: {results['width_average']['zscore']['n_outliers']}")
    
    summary.append("\n" + "=" * 80)
    summary.append("CONSENSUS")
    summary.append("=" * 80)
    summary.append(f"\nSamples flagged in BOTH dimensions:")
    summary.append(f"  IQR method: {len(iqr_both)}")
    summary.append(f"  Z-score method: {len(zscore_both)}")
    summary.append(f"  Either method: {len(consensus_outliers)}")
    
    if len(consensus_outliers) == 0:
        summary.append("\n✅ NO OUTLIERS DETECTED by either method in both dimensions")
        summary.append("   All 58 samples retained for analysis")
    else:
        summary.append(f"\n⚠️  {len(consensus_outliers)} samples flagged as potential outliers")
        summary.append("   Recommended: Visual inspection before removal")
    
    with open(output_dir / 'outlier_detection_summary.txt', 'w') as f:
        f.write('\n'.join(summary))
    
    print(f"✅ Summary saved: outlier_detection_summary.txt")
    
    # Final verdict
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    
    if len(consensus_outliers) == 0:
        print("\n✅ NO OUTLIERS DETECTED")
        print("\nBoth the IQR method (1.5×IQR threshold) and Z-score method")
        print("(|Z| > 3.0 threshold) found zero samples that were outliers in")
        print("both height AND width dimensions.")
        print("\nRecommendation: Retain all 58 samples for modeling.")
        print("\nFor paper, write:")
        print("  'Outlier detection was performed using two methods: the")
        print("   interquartile range (IQR) method with 1.5×IQR threshold and")
        print("   the Z-score method with |Z| > 3.0 threshold. Neither method")
        print("   identified samples as outliers in both height and width")
        print("   dimensions, confirming that dimensional variations were")
        print("   consistent with the explored parameter ranges. All 58 samples")
        print("   were retained for analysis.'")
    else:
        print(f"\n⚠️  {len(consensus_outliers)} POTENTIAL OUTLIERS DETECTED")
        print(f"\nIndices: {sorted(consensus_outliers)}")
        print("\nRecommendation:")
        print("  1. Visually inspect these samples (check printed samples photo)")
        print("  2. Check measurement data for these samples")
        print("  3. If they look like printing failures → remove")
        print("  4. If they look fine → retain (extreme parameters ≠ outliers)")


def main():
    """Main execution."""
    print("\n" + "="*80)
    print("OUTLIER DETECTION FOR ROBOCASTING DATASET")
    print("="*80)
    
    data_path = '/Users/nazarii/projects/ipms/robocasting/data/cleaned_df.csv'
    
    try:
        analyze_outliers(data_path, output_dir='dataset_analysis')
        
        print("\n" + "="*80)
        print("✅ OUTLIER ANALYSIS COMPLETE")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

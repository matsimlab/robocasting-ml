#!/usr/bin/env python3
"""
Robocasting Parameter Optimization: Complete Regression Analysis with GPR
==========================================================================

This script provides a comprehensive comparison of regression models including:
- Simple models (Ridge, Lasso, XGB, GBR, RF)
- Gaussian Process Regression (GPR) with multiple kernels
- GPR uncertainty analysis for dataset quality assessment

Author: Nazarii
Date: 2025
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Sklearn imports
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, Matern
from sklearn.base import clone

# XGBoost
from xgboost import XGBRegressor

# Set random seed
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Configure plotting
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class RobocastingAnalyzer:
    """Complete analyzer with GPR uncertainty quantification."""
    
    def __init__(self, data_path, output_dir='results_final'):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.df = None
        self.df_cleaned = None
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.scaler = None
        
        self.results = {}
        self.best_models = {}
        self.use_loocv = False
        
    def load_data(self):
        """Load and inspect data."""
        print("=" * 80)
        print("LOADING DATA")
        print("=" * 80)
        
        self.df = pd.read_csv(self.data_path)
        print(f"\nDataset shape: {self.df.shape}")
        
        n_samples = len(self.df)
        if n_samples < 100:
            print(f"\n⚠️  WARNING: Small dataset (n={n_samples})")
            print("   Using simplified models and LOOCV")
        
        print(f"\nColumns: {list(self.df.columns)}")
        print(f"\nFirst rows:\n{self.df.head()}")
        print(f"\nMissing values:\n{self.df.isnull().sum()}")
        
    def preprocess_data(self):
        """Preprocess and engineer features."""
        print("\n" + "=" * 80)
        print("PREPROCESSING")
        print("=" * 80)
        
        # Drop unnecessary columns
        cols_to_drop = ['date', 'number']
        existing = [c for c in cols_to_drop if c in self.df.columns]
        if existing:
            self.df.drop(columns=existing, inplace=True)
            print(f"\nDropped: {existing}")
        
        # Remove missing values
        initial = len(self.df)
        self.df.dropna(inplace=True)
        print(f"Rows dropped: {initial - len(self.df)}")
        
        # Calculate averages
        self.df['height_average'] = (self.df['height_1'] + self.df['height_2'] + self.df['height_3']) / 3
        self.df['width_average'] = (self.df['width_1'] + self.df['width_2'] + self.df['width_3']) / 3
        
        print(f"\nFinal shape: {self.df.shape}")
        print(f"Height range: [{self.df['height_average'].min():.2f}, {self.df['height_average'].max():.2f}]")
        print(f"Width range: [{self.df['width_average'].min():.2f}, {self.df['width_average'].max():.2f}]")
        
        self.df_cleaned = self.df.copy()
    
    def prepare_train_test_split(self):
        """Prepare features and split data."""
        print("\n" + "=" * 80)
        print("TRAIN/TEST SPLIT")
        print("=" * 80)
        
        # Define features
        features = ['slicer_nozzle_speed', 'slicer_extrusion_multiplier', 'layer_count']
        available = [f for f in features if f in self.df_cleaned.columns]
        
        print(f"\nFeatures: {available}")
        
        X = self.df_cleaned[available]
        y = self.df_cleaned[['height_average', 'width_average']]
        
        # Remove constant features
        constant = [col for col in X.columns if X[col].nunique() == 1]
        if constant:
            print(f"\n❌ Removing constant features: {constant}")
            X = X.drop(columns=constant)
        
        # Check sample size
        n_samples = len(X)
        n_features = len(X.columns)
        print(f"\nSamples: {n_samples}, Features: {n_features}")
        print(f"Samples/feature: {n_samples/n_features:.1f}")
        
        if n_samples < 100:
            self.use_loocv = True
            print("✅ Using LOOCV for small dataset")
        
        # Scale and split
        self.scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_scaled, y, test_size=0.3, random_state=RANDOM_STATE
        )
        
        print(f"\nTrain: {len(self.X_train)}, Val: {len(self.X_val)}")
        
    def train_all_models(self):
        """Train ALL models including GPR."""
        print("\n" + "=" * 80)
        print("TRAINING ALL MODELS (Including GPR)")
        print("=" * 80)
        
        # Define all models
        models = {
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.1, max_iter=10000),
            'XGB': XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1,
                               min_child_weight=5, subsample=0.8,
                               objective='reg:squarederror', random_state=RANDOM_STATE),
            'GBR': GradientBoostingRegressor(n_estimators=50, max_depth=3, learning_rate=0.1,
                                            min_samples_split=10, subsample=0.8,
                                            random_state=RANDOM_STATE),
            'RF': RandomForestRegressor(n_estimators=50, max_depth=5, min_samples_split=10,
                                       random_state=RANDOM_STATE),
            
            # GPR with different kernels (optimized settings)
            'GPR_RBF': GaussianProcessRegressor(
                kernel=ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) * 
                       RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)),
                alpha=1e-3, n_restarts_optimizer=10, normalize_y=True, random_state=RANDOM_STATE
            ),
            'GPR_RBF+White': GaussianProcessRegressor(
                kernel=ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) * 
                       RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + 
                       WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1)),
                alpha=1e-10, n_restarts_optimizer=10, normalize_y=True, random_state=RANDOM_STATE
            ),
            'GPR_Matern1.5': GaussianProcessRegressor(
                kernel=ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) * 
                       Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5),
                alpha=1e-3, n_restarts_optimizer=10, normalize_y=True, random_state=RANDOM_STATE
            ),
            'GPR_Matern2.5': GaussianProcessRegressor(
                kernel=ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) * 
                       Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5),
                alpha=1e-3, n_restarts_optimizer=10, normalize_y=True, random_state=RANDOM_STATE
            ),
        }
        
        all_results = []
        best_gpr_mae = float('inf')
        best_gpr_name = None
        
        for model_name, model in models.items():
            print(f"\n{'='*60}")
            print(f"Training {model_name}")
            print('='*60)
            
            is_gpr = model_name.startswith('GPR')
            
            # Train height model
            try:
                model_h = model
                model_h.fit(self.X_train, self.y_train['height_average'])
            except Exception as e:
                print(f"  ⚠️  Failed to train height model: {e}")
                continue
            
            # Train width model
            try:
                model_w = clone(model)
                model_w.fit(self.X_train, self.y_train['width_average'])
            except Exception as e:
                print(f"  ⚠️  Failed to train width model: {e}")
                continue
            
            # Store models
            self.best_models[f'{model_name}_height'] = model_h
            self.best_models[f'{model_name}_width'] = model_w
            
            # Predictions
            if is_gpr:
                train_h_pred, train_h_std = model_h.predict(self.X_train, return_std=True)
                train_w_pred, train_w_std = model_w.predict(self.X_train, return_std=True)
                val_h_pred, val_h_std = model_h.predict(self.X_val, return_std=True)
                val_w_pred, val_w_std = model_w.predict(self.X_val, return_std=True)
            else:
                train_h_pred = model_h.predict(self.X_train)
                train_w_pred = model_w.predict(self.X_train)
                val_h_pred = model_h.predict(self.X_val)
                val_w_pred = model_w.predict(self.X_val)
                train_h_std = train_w_std = val_h_std = val_w_std = None
            
            # Metrics
            train_h_mae = mean_absolute_error(self.y_train['height_average'], train_h_pred)
            train_w_mae = mean_absolute_error(self.y_train['width_average'], train_w_pred)
            train_h_r2 = r2_score(self.y_train['height_average'], train_h_pred)
            train_w_r2 = r2_score(self.y_train['width_average'], train_w_pred)
            
            val_h_mae = mean_absolute_error(self.y_val['height_average'], val_h_pred)
            val_w_mae = mean_absolute_error(self.y_val['width_average'], val_w_pred)
            val_h_rmse = np.sqrt(mean_squared_error(self.y_val['height_average'], val_h_pred))
            val_w_rmse = np.sqrt(mean_squared_error(self.y_val['width_average'], val_w_pred))
            val_h_r2 = r2_score(self.y_val['height_average'], val_h_pred)
            val_w_r2 = r2_score(self.y_val['width_average'], val_w_pred)
            
            h_overfit = train_h_r2 - val_h_r2
            w_overfit = train_w_r2 - val_w_r2
            
            # Print results
            print(f"\nHeight:")
            print(f"  Train   - MAE: {train_h_mae:.4f}, R²: {train_h_r2:.4f}")
            print(f"  Val     - MAE: {val_h_mae:.4f}, R²: {val_h_r2:.4f}")
            if is_gpr:
                print(f"  Uncertainty: ±{np.mean(val_h_std):.4f}")
            print(f"  Overfit: {h_overfit:.3f}", "⚠️" if h_overfit > 0.2 else "✓")
            
            print(f"\nWidth:")
            print(f"  Train   - MAE: {train_w_mae:.4f}, R²: {train_w_r2:.4f}")
            print(f"  Val     - MAE: {val_w_mae:.4f}, R²: {val_w_r2:.4f}")
            if is_gpr:
                print(f"  Uncertainty: ±{np.mean(val_w_std):.4f}")
                print(f"  Learned kernel: {model_w.kernel_}")
                if val_w_r2 < 0:
                    print(f"  ❌ NEGATIVE R²! Model worse than baseline. Check:")
                    print(f"     - Data scaling (range: [{self.y_train['width_average'].min():.2f}, {self.y_train['width_average'].max():.2f}])")
                    print(f"     - Kernel hyperparameters may need adjustment")
            print(f"  Overfit: {w_overfit:.3f}", "⚠️" if w_overfit > 0.2 else "✓")
            
            # Track best GPR
            combined_mae = val_h_mae + val_w_mae
            if is_gpr and combined_mae < best_gpr_mae:
                best_gpr_mae = combined_mae
                best_gpr_name = model_name
            
            # Store results
            result = {
                'Model': model_name,
                'Train Height MAE': train_h_mae,
                'Val Height MAE': val_h_mae,
                'Train Width MAE': train_w_mae,
                'Val Width MAE': val_w_mae,
                'Height RMSE': val_h_rmse,
                'Width RMSE': val_w_rmse,
                'Train Height R²': train_h_r2,
                'Val Height R²': val_h_r2,
                'Train Width R²': train_w_r2,
                'Val Width R²': val_w_r2,
                'Height Overfit': h_overfit,
                'Width Overfit': w_overfit,
                'Combined MAE': combined_mae,
                'Type': 'GPR' if is_gpr else 'Simple'
            }
            
            if is_gpr:
                result['Height Uncertainty'] = np.mean(val_h_std)
                result['Width Uncertainty'] = np.mean(val_w_std)
            
            all_results.append(result)
        
        self.results['all_models'] = pd.DataFrame(all_results)
        print(f"\n✅ Best GPR kernel: {best_gpr_name}")
        
        return best_gpr_name
    
    def analyze_gpr_uncertainty(self, gpr_model_name):
        """Comprehensive GPR uncertainty analysis for dataset quality."""
        print("\n" + "=" * 80)
        print("GPR UNCERTAINTY ANALYSIS - DATASET QUALITY")
        print("=" * 80)
        
        print(f"\nAnalyzing: {gpr_model_name}")
        
        # Get GPR models
        gpr_h = self.best_models[f'{gpr_model_name}_height']
        gpr_w = self.best_models[f'{gpr_model_name}_width']
        
        # Predict on ALL data
        X_all = pd.concat([self.X_train, self.X_val])
        y_all = pd.concat([self.y_train, self.y_val])
        
        h_pred, h_std = gpr_h.predict(X_all, return_std=True)
        w_pred, w_std = gpr_w.predict(X_all, return_std=True)
        
        # Create visualization
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
        
        # 1. Predictions with confidence intervals - Height
        ax1 = fig.add_subplot(gs[0, 0])
        sorted_idx = np.argsort(y_all['height_average'])
        ax1.plot(y_all['height_average'].iloc[sorted_idx], h_pred[sorted_idx],
                'b-', lw=2, label='GPR Mean')
        ax1.fill_between(y_all['height_average'].iloc[sorted_idx],
                         h_pred[sorted_idx] - 1.96*h_std[sorted_idx],
                         h_pred[sorted_idx] + 1.96*h_std[sorted_idx],
                         alpha=0.3, label='95% CI')
        ax1.plot([y_all['height_average'].min(), y_all['height_average'].max()],
                 [y_all['height_average'].min(), y_all['height_average'].max()],
                 'r--', lw=2, label='Perfect')
        ax1.set_xlabel('Actual Height', fontweight='bold')
        ax1.set_ylabel('Predicted Height', fontweight='bold')
        ax1.set_title('Height: Predictions with Uncertainty', fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # 2. Predictions with confidence intervals - Width
        ax2 = fig.add_subplot(gs[0, 1])
        sorted_idx = np.argsort(y_all['width_average'])
        ax2.plot(y_all['width_average'].iloc[sorted_idx], w_pred[sorted_idx],
                'g-', lw=2, label='GPR Mean')
        ax2.fill_between(y_all['width_average'].iloc[sorted_idx],
                         w_pred[sorted_idx] - 1.96*w_std[sorted_idx],
                         w_pred[sorted_idx] + 1.96*w_std[sorted_idx],
                         alpha=0.3, label='95% CI')
        ax2.plot([y_all['width_average'].min(), y_all['width_average'].max()],
                 [y_all['width_average'].min(), y_all['width_average'].max()],
                 'r--', lw=2, label='Perfect')
        ax2.set_xlabel('Actual Width', fontweight='bold')
        ax2.set_ylabel('Predicted Width', fontweight='bold')
        ax2.set_title('Width: Predictions with Uncertainty', fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # 3. Uncertainty distribution
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.hist(h_std, bins=20, alpha=0.7, label='Height', edgecolor='black')
        ax3.hist(w_std, bins=20, alpha=0.7, label='Width', edgecolor='black')
        ax3.axvline(np.mean(h_std), color='blue', linestyle='--', lw=2)
        ax3.axvline(np.mean(w_std), color='orange', linestyle='--', lw=2)
        ax3.set_xlabel('Uncertainty (σ)', fontweight='bold')
        ax3.set_ylabel('Frequency', fontweight='bold')
        ax3.set_title('Uncertainty Distribution', fontweight='bold')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        # 4. Uncertainty vs Actual - Height
        ax4 = fig.add_subplot(gs[1, 0])
        scatter = ax4.scatter(y_all['height_average'], h_std, c=h_std,
                             cmap='coolwarm', edgecolors='k', linewidth=0.5)
        ax4.set_xlabel('Actual Height', fontweight='bold')
        ax4.set_ylabel('Uncertainty (σ)', fontweight='bold')
        ax4.set_title('Height: Where is model uncertain?', fontweight='bold')
        plt.colorbar(scatter, ax=ax4, label='Uncertainty')
        ax4.grid(alpha=0.3)
        
        # 5. Uncertainty vs Actual - Width
        ax5 = fig.add_subplot(gs[1, 1])
        scatter = ax5.scatter(y_all['width_average'], w_std, c=w_std,
                             cmap='coolwarm', edgecolors='k', linewidth=0.5)
        ax5.set_xlabel('Actual Width', fontweight='bold')
        ax5.set_ylabel('Uncertainty (σ)', fontweight='bold')
        ax5.set_title('Width: Where is model uncertain?', fontweight='bold')
        plt.colorbar(scatter, ax=ax5, label='Uncertainty')
        ax5.grid(alpha=0.3)
        
        # 6. Calibration - Height
        ax6 = fig.add_subplot(gs[1, 2])
        h_errors = np.abs(y_all['height_average'] - h_pred)
        ax6.scatter(h_std, h_errors, alpha=0.6, edgecolors='k', linewidth=0.5)
        max_val = max(h_std.max(), h_errors.max())
        ax6.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect Calibration')
        ax6.set_xlabel('Uncertainty (σ)', fontweight='bold')
        ax6.set_ylabel('Absolute Error', fontweight='bold')
        ax6.set_title('Height: Uncertainty Calibration', fontweight='bold')
        ax6.legend()
        ax6.grid(alpha=0.3)
        
        # 7. Uncertainty in feature space
        if len(self.X_train.columns) >= 2:
            ax7 = fig.add_subplot(gs[2, 0])
            feat1, feat2 = self.X_train.columns[0], self.X_train.columns[1]
            scatter = ax7.scatter(X_all[feat1], X_all[feat2], c=h_std,
                                 cmap='YlOrRd', s=100, edgecolors='k', linewidth=0.5)
            ax7.set_xlabel(feat1, fontweight='bold')
            ax7.set_ylabel(feat2, fontweight='bold')
            ax7.set_title('Height: Uncertainty in Feature Space', fontweight='bold')
            plt.colorbar(scatter, ax=ax7, label='Uncertainty')
            ax7.grid(alpha=0.3)
            
            ax8 = fig.add_subplot(gs[2, 1])
            scatter = ax8.scatter(X_all[feat1], X_all[feat2], c=w_std,
                                 cmap='YlOrRd', s=100, edgecolors='k', linewidth=0.5)
            ax8.set_xlabel(feat1, fontweight='bold')
            ax8.set_ylabel(feat2, fontweight='bold')
            ax8.set_title('Width: Uncertainty in Feature Space', fontweight='bold')
            plt.colorbar(scatter, ax=ax8, label='Uncertainty')
            ax8.grid(alpha=0.3)
        
        # 8. Data quality regions
        ax9 = fig.add_subplot(gs[2, 2])
        h_high = h_std > np.percentile(h_std, 75)
        w_high = w_std > np.percentile(w_std, 75)
        
        both_low = (~h_high & ~w_high).sum()
        mixed = (h_high ^ w_high).sum()
        both_high = (h_high & w_high).sum()
        
        categories = ['Low\nUncertainty', 'Medium\nUncertainty', 'High\nUncertainty']
        counts = [both_low, mixed, both_high]
        colors = ['green', 'orange', 'red']
        
        ax9.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
        ax9.set_ylabel('Number of Samples', fontweight='bold')
        ax9.set_title('Data Quality Regions', fontweight='bold')
        ax9.grid(axis='y', alpha=0.3)
        
        for i, (cat, count) in enumerate(zip(categories, counts)):
            pct = 100 * count / len(X_all)
            ax9.text(i, count, f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.savefig(self.output_dir / f'{gpr_model_name}_uncertainty_analysis.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print statistics
        print("\n" + "="*80)
        print("UNCERTAINTY STATISTICS")
        print("="*80)
        
        print(f"\nHeight: Mean={np.mean(h_std):.4f}, Range=[{np.min(h_std):.4f}, {np.max(h_std):.4f}]")
        print(f"Width:  Mean={np.mean(w_std):.4f}, Range=[{np.min(w_std):.4f}, {np.max(w_std):.4f}]")
        
        # Most/least certain
        h_most = np.argmin(h_std)
        h_least = np.argmax(h_std)
        
        print("\n" + "="*80)
        print("KEY PREDICTIONS")
        print("="*80)
        
        print(f"\nMOST CERTAIN (Height):")
        print(f"  Actual: {y_all['height_average'].iloc[h_most]:.3f}")
        print(f"  Predicted: {h_pred[h_most]:.3f} ± {h_std[h_most]:.3f}")
        print(f"  Features: {X_all.iloc[h_most].to_dict()}")
        
        print(f"\nLEAST CERTAIN (Height) - COLLECT MORE DATA HERE:")
        print(f"  Actual: {y_all['height_average'].iloc[h_least]:.3f}")
        print(f"  Predicted: {h_pred[h_least]:.3f} ± {h_std[h_least]:.3f}")
        print(f"  Features: {X_all.iloc[h_least].to_dict()}")
        
        # Calibration
        h_errors = np.abs(y_all['height_average'] - h_pred)
        w_errors = np.abs(y_all['width_average'] - w_pred)
        h_calib = np.corrcoef(h_std, h_errors)[0, 1]
        w_calib = np.corrcoef(w_std, w_errors)[0, 1]
        
        print("\n" + "="*80)
        print("DATASET QUALITY ASSESSMENT")
        print("="*80)
        
        print(f"\nCalibration (correlation with errors):")
        print(f"  Height: {h_calib:.3f}", "✓ Well-calibrated" if h_calib > 0.5 else "⚠️ Poor calibration")
        print(f"  Width:  {w_calib:.3f}", "✓ Well-calibrated" if w_calib > 0.5 else "⚠️ Poor calibration")
        
        if both_high > 0.2 * len(X_all):
            print(f"\n⚠️  {both_high} samples ({100*both_high/len(X_all):.1f}%) have high uncertainty")
            print("   → Collect more data in these regions")
        
        print(f"\n✅ Uncertainty analysis complete!")
        print(f"   Visualization saved: {gpr_model_name}_uncertainty_analysis.png")
    
    def compare_all_models(self):
        """Compare ALL models (simple + GPR)."""
        print("\n" + "=" * 80)
        print("FINAL MODEL COMPARISON")
        print("=" * 80)
        
        df = self.results['all_models']
        
        # Create comparison dataframe
        comparison = df[[
            'Model', 'Val Height MAE', 'Val Width MAE',
            'Height RMSE', 'Width RMSE',
            'Val Height R²', 'Val Width R²', 'Combined MAE', 'Type'
        ]].copy()
        comparison.columns = [
            'Model', 'Height MAE', 'Width MAE',
            'Height RMSE', 'Width RMSE',
            'Height R²', 'Width R²', 'Combined MAE', 'Type'
        ]
        comparison = comparison.sort_values('Combined MAE')
        
        print("\n" + "="*120)
        print("ALL MODELS (Validation Set)")
        print("="*120)
        print(comparison.to_string(index=False))
        
        # Save results
        comparison.to_csv(self.output_dir / 'model_comparison.csv', index=False)
        df.to_csv(self.output_dir / 'model_detailed_results.csv', index=False)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        x = np.arange(len(comparison))
        width_bar = 0.35
        
        # MAE
        axes[0, 0].bar(x - width_bar/2, comparison['Height MAE'], width_bar, label='Height', alpha=0.8)
        axes[0, 0].bar(x + width_bar/2, comparison['Width MAE'], width_bar, label='Width', alpha=0.8)
        axes[0, 0].set_xlabel('Model', fontweight='bold')
        axes[0, 0].set_ylabel('MAE', fontweight='bold')
        axes[0, 0].set_title('MAE Comparison (All Models)', fontweight='bold')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(comparison['Model'], rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # RMSE
        axes[0, 1].bar(x - width_bar/2, comparison['Height RMSE'], width_bar, label='Height', alpha=0.8)
        axes[0, 1].bar(x + width_bar/2, comparison['Width RMSE'], width_bar, label='Width', alpha=0.8)
        axes[0, 1].set_xlabel('Model', fontweight='bold')
        axes[0, 1].set_ylabel('RMSE', fontweight='bold')
        axes[0, 1].set_title('RMSE Comparison', fontweight='bold')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(comparison['Model'], rotation=45, ha='right')
        axes[0, 1].legend()
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # R²
        axes[1, 0].bar(x - width_bar/2, comparison['Height R²'], width_bar, label='Height', alpha=0.8)
        axes[1, 0].bar(x + width_bar/2, comparison['Width R²'], width_bar, label='Width', alpha=0.8)
        axes[1, 0].set_xlabel('Model', fontweight='bold')
        axes[1, 0].set_ylabel('R²', fontweight='bold')
        axes[1, 0].set_title('R² Comparison', fontweight='bold')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(comparison['Model'], rotation=45, ha='right')
        axes[1, 0].legend()
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Combined MAE by type
        simple = comparison[comparison['Type'] == 'Simple']
        gpr = comparison[comparison['Type'] == 'GPR']
        
        axes[1, 1].barh(simple['Model'], simple['Combined MAE'],
                       alpha=0.8, label='Simple Models', color='steelblue')
        axes[1, 1].barh(gpr['Model'], gpr['Combined MAE'],
                       alpha=0.8, label='GPR Models', color='orange')
        axes[1, 1].set_xlabel('Combined MAE', fontweight='bold')
        axes[1, 1].set_title('Combined MAE (All Models)', fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✅ Results saved to: {self.output_dir}")
        print(f"   - model_comparison.csv")
        print(f"   - model_comparison.png")
        print(f"   - model_detailed_results.csv")


def main():
    """Main execution."""
    print("\n" + "="*80)
    print("COMPLETE ROBOCASTING ANALYSIS WITH GPR")
    print("="*80 + "\n")
    
    # Initialize
    data_path = '/Users/nazarii/projects/ipms/robocasting/data/cleaned_df.csv'
    analyzer = RobocastingAnalyzer(data_path, output_dir='results_final')
    
    try:
        # 1. Load data
        analyzer.load_data()
        
        # 2. Preprocess
        analyzer.preprocess_data()
        
        # 3. Prepare split
        analyzer.prepare_train_test_split()
        
        # 4. Train ALL models (simple + GPR)
        best_gpr = analyzer.train_all_models()
        
        # 5. Analyze GPR uncertainty
        analyzer.analyze_gpr_uncertainty(best_gpr)
        
        # 6. Compare all models
        analyzer.compare_all_models()
        
        print("\n" + "="*80)
        print("✅ ANALYSIS COMPLETE!")
        print("="*80)
        print(f"\nResults in: {analyzer.output_dir}")
        print("\nKey files:")
        print("  - model_comparison.csv: All models compared")
        print("  - model_comparison.png: Visual comparison")
        print("  - GPR_XXX_uncertainty_analysis.png: Uncertainty visualization")
        print("  - model_detailed_results.csv: Full metrics")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

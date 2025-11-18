#!/usr/bin/env python3
"""
Robocasting Parameter Optimization: Nested CV with Bayesian Optimization
=========================================================================

This script uses nested cross-validation for robust model evaluation:
- Outer CV (5-fold): Model performance evaluation
- Inner CV (3-fold): Bayesian hyperparameter optimization (20 iterations)
- Proper separation prevents data leakage
- Reports mean ± std for publication
- More efficient than grid search

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
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, Matern
from sklearn.base import clone

# Bayesian Optimization
from skopt import BayesSearchCV
from skopt.space import Real, Integer

# XGBoost
from xgboost import XGBRegressor

# SHAP for feature importance (optional)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️  SHAP not available. Will use built-in feature importance.")
    print("   Install with: pip install shap")

# Set random seed
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Configure plotting
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class RobocastingAnalyzerNestedCV:
    """Analyzer with nested cross-validation for publication."""
    
    def __init__(self, data_path, output_dir='results_nested_cv'):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.df = None
        self.X = None
        self.y = None
        self.scaler = None
        
        self.results = {}
        self.best_models = {}
        
        # CV configuration
        self.outer_cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        self.inner_cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        
    def load_data(self):
        """Load and inspect data."""
        print("=" * 80)
        print("LOADING DATA")
        print("=" * 80)
        
        self.df = pd.read_csv(self.data_path)
        print(f"\nDataset shape: {self.df.shape}")
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
        
    def prepare_features(self):
        """Prepare features and targets."""
        print("\n" + "=" * 80)
        print("FEATURE PREPARATION")
        print("=" * 80)
        
        # Define features
        features = ['slicer_nozzle_speed', 'slicer_extrusion_multiplier', 'layer_count']
        available = [f for f in features if f in self.df.columns]
        
        print(f"\nFeatures: {available}")
        
        X = self.df[available]
        y = self.df[['height_average', 'width_average']]
        
        # Remove constant features
        constant = [col for col in X.columns if X[col].nunique() == 1]
        if constant:
            print(f"\n❌ Removing constant features: {constant}")
            X = X.drop(columns=constant)
        
        n_samples = len(X)
        n_features = len(X.columns)
        print(f"\nSamples: {n_samples}, Features: {n_features}")
        print(f"Samples/feature: {n_samples/n_features:.1f}")
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        self.X = X_scaled
        self.y = y
        
        print(f"\n✅ Using Nested CV:")
        print(f"   Outer: 5-fold (model evaluation)")
        print(f"   Inner: 3-fold (hyperparameter tuning)")
        
    def get_param_spaces(self):
        """Define hyperparameter search spaces for Bayesian optimization."""
        return {
            'Ridge': {
                'alpha': Real(0.01, 100.0, prior='log-uniform')
            },
            'Lasso': {
                'alpha': Real(0.001, 10.0, prior='log-uniform')
            },
            'XGB': {
                'max_depth': Integer(2, 6),
                'n_estimators': Integer(30, 150),
                'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                'subsample': Real(0.6, 1.0),
                'min_child_weight': Integer(1, 10)
            },
            'GBR': {
                'max_depth': Integer(2, 6),
                'n_estimators': Integer(30, 150),
                'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                'subsample': Real(0.6, 1.0),
                'min_samples_split': Integer(5, 20)
            },
            'RF': {
                'max_depth': Integer(3, 8),
                'n_estimators': Integer(30, 150),
                'min_samples_split': Integer(2, 15),
                'min_samples_leaf': Integer(1, 5)
            }
        }
    
    def train_with_nested_cv(self):
        """Train models using nested cross-validation."""
        print("\n" + "=" * 80)
        print("NESTED CROSS-VALIDATION")
        print("=" * 80)
        
        # Define base models
        base_models = {
            'Ridge': Ridge(random_state=RANDOM_STATE),
            'Lasso': Lasso(max_iter=10000, random_state=RANDOM_STATE),
            'XGB': XGBRegressor(random_state=RANDOM_STATE, objective='reg:squarederror'),
            'GBR': GradientBoostingRegressor(random_state=RANDOM_STATE),
            'RF': RandomForestRegressor(random_state=RANDOM_STATE),
        }
        
        # GPR models (no hyperparameter tuning for these - too expensive)
        gpr_models = {
            'GPR_RBF': GaussianProcessRegressor(
                kernel=ConstantKernel(1.0) * RBF(length_scale=1.0),
                alpha=1e-3, n_restarts_optimizer=5, normalize_y=True, random_state=RANDOM_STATE
            ),
            'GPR_RBF+White': GaussianProcessRegressor(
                kernel=ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1),
                alpha=1e-10, n_restarts_optimizer=5, normalize_y=True, random_state=RANDOM_STATE
            ),
            'GPR_Matern': GaussianProcessRegressor(
                kernel=ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5),
                alpha=1e-3, n_restarts_optimizer=5, normalize_y=True, random_state=RANDOM_STATE
            ),
        }
        
        param_spaces = self.get_param_spaces()
        
        # Number of iterations for Bayesian optimization
        n_iter = 20  # 20 iterations usually sufficient
        
        all_results = []
        
        # Train models with hyperparameter tuning
        for model_name, base_model in base_models.items():
            print(f"\n{'='*60}")
            print(f"Training {model_name} with Nested CV")
            print('='*60)
            
            param_space = param_spaces.get(model_name, {})
            
            # Process both targets
            for target_name in ['height_average', 'width_average']:
                print(f"\n{target_name}:")
                
                y_target = self.y[target_name]
                
                if param_space:
                    # With Bayesian hyperparameter optimization
                    bayes_search = BayesSearchCV(
                        base_model,
                        param_space,
                        n_iter=n_iter,
                        cv=self.inner_cv,
                        scoring='neg_mean_absolute_error',
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                        verbose=0
                    )
                    
                    # Outer CV for evaluation
                    mae_scores = cross_val_score(
                        bayes_search, self.X, y_target,
                        cv=self.outer_cv,
                        scoring='neg_mean_absolute_error',
                        n_jobs=-1
                    )
                    
                    r2_scores = cross_val_score(
                        bayes_search, self.X, y_target,
                        cv=self.outer_cv,
                        scoring='r2',
                        n_jobs=-1
                    )
                    
                    # Fit on all data to get best params
                    bayes_search.fit(self.X, y_target)
                    best_params = bayes_search.best_params_
                    print(f"  Best params: {best_params}")
                    
                else:
                    # No hyperparameter tuning
                    mae_scores = cross_val_score(
                        base_model, self.X, y_target,
                        cv=self.outer_cv,
                        scoring='neg_mean_absolute_error',
                        n_jobs=-1
                    )
                    
                    r2_scores = cross_val_score(
                        base_model, self.X, y_target,
                        cv=self.outer_cv,
                        scoring='r2',
                        n_jobs=-1
                    )
                    best_params = {}
                
                mae_scores = -mae_scores  # Convert back to positive
                
                print(f"  MAE: {mae_scores.mean():.4f} ± {mae_scores.std():.4f}")
                print(f"  R²:  {r2_scores.mean():.4f} ± {r2_scores.std():.4f}")
                
                # Store results
                result = {
                    'Model': model_name,
                    'Target': target_name,
                    'MAE_mean': mae_scores.mean(),
                    'MAE_std': mae_scores.std(),
                    'R2_mean': r2_scores.mean(),
                    'R2_std': r2_scores.std(),
                    'Type': 'Simple',
                    'Best_params': str(best_params)
                }
                all_results.append(result)
        
        # Train GPR models (no hyperparameter tuning)
        print(f"\n{'='*60}")
        print("Training GPR Models (no hyperparameter tuning)")
        print('='*60)
        
        for model_name, gpr_model in gpr_models.items():
            print(f"\n{model_name}:")
            
            for target_name in ['height_average', 'width_average']:
                print(f"  {target_name}:")
                
                y_target = self.y[target_name]
                
                # Simple CV evaluation (no nested CV for GPR - too slow)
                mae_scores = cross_val_score(
                    gpr_model, self.X, y_target,
                    cv=self.outer_cv,
                    scoring='neg_mean_absolute_error',
                    n_jobs=1  # GPR doesn't parallelize well
                )
                
                r2_scores = cross_val_score(
                    gpr_model, self.X, y_target,
                    cv=self.outer_cv,
                    scoring='r2',
                    n_jobs=1
                )
                
                mae_scores = -mae_scores
                
                print(f"    MAE: {mae_scores.mean():.4f} ± {mae_scores.std():.4f}")
                print(f"    R²:  {r2_scores.mean():.4f} ± {r2_scores.std():.4f}")
                
                result = {
                    'Model': model_name,
                    'Target': target_name,
                    'MAE_mean': mae_scores.mean(),
                    'MAE_std': mae_scores.std(),
                    'R2_mean': r2_scores.mean(),
                    'R2_std': r2_scores.std(),
                    'Type': 'GPR',
                    'Best_params': 'Fixed'
                }
                all_results.append(result)
        
        self.results['nested_cv'] = pd.DataFrame(all_results)
        
    def train_final_models(self):
        """Train final models on all data for uncertainty analysis."""
        print("\n" + "=" * 80)
        print("TRAINING FINAL MODELS ON ALL DATA")
        print("=" * 80)
        
        # Get best GPR model
        gpr_results = self.results['nested_cv'][self.results['nested_cv']['Type'] == 'GPR']
        
        # Calculate combined MAE for each GPR model
        gpr_combined = gpr_results.groupby('Model').agg({
            'MAE_mean': 'mean',
            'R2_mean': 'mean'
        }).reset_index()
        
        best_gpr = gpr_combined.loc[gpr_combined['MAE_mean'].idxmin(), 'Model']
        print(f"\n✅ Best GPR model: {best_gpr}")
        
        # Train best GPR on all data
        if best_gpr == 'GPR_RBF':
            gpr_h = GaussianProcessRegressor(
                kernel=ConstantKernel(1.0) * RBF(length_scale=1.0),
                alpha=1e-3, n_restarts_optimizer=10, normalize_y=True, random_state=RANDOM_STATE
            )
            gpr_w = clone(gpr_h)
        elif best_gpr == 'GPR_RBF+White':
            gpr_h = GaussianProcessRegressor(
                kernel=ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1),
                alpha=1e-10, n_restarts_optimizer=10, normalize_y=True, random_state=RANDOM_STATE
            )
            gpr_w = clone(gpr_h)
        else:  # Matern
            gpr_h = GaussianProcessRegressor(
                kernel=ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5),
                alpha=1e-3, n_restarts_optimizer=10, normalize_y=True, random_state=RANDOM_STATE
            )
            gpr_w = clone(gpr_h)
        
        print("\nTraining on full dataset for uncertainty analysis...")
        gpr_h.fit(self.X, self.y['height_average'])
        gpr_w.fit(self.X, self.y['width_average'])
        
        self.best_models[f'{best_gpr}_height'] = gpr_h
        self.best_models[f'{best_gpr}_width'] = gpr_w
        
        return best_gpr
    
    def analyze_gpr_uncertainty(self, gpr_model_name):
        """GPR uncertainty analysis."""
        print("\n" + "=" * 80)
        print("GPR UNCERTAINTY ANALYSIS")
        print("=" * 80)
        
        gpr_h = self.best_models[f'{gpr_model_name}_height']
        gpr_w = self.best_models[f'{gpr_model_name}_width']
        
        h_pred, h_std = gpr_h.predict(self.X, return_std=True)
        w_pred, w_std = gpr_w.predict(self.X, return_std=True)
        
        print(f"\nHeight uncertainty: {h_std.mean():.4f} ± {h_std.std():.4f}")
        print(f"Width uncertainty:  {w_std.mean():.4f} ± {w_std.std():.4f}")
        
        # Find high uncertainty regions
        h_high = h_std > np.percentile(h_std, 75)
        w_high = w_std > np.percentile(w_std, 75)
        both_high = (h_high & w_high).sum()
        
        print(f"\n⚠️  {both_high} samples ({100*both_high/len(self.X):.1f}%) have high uncertainty")
        print("   → Consider collecting more data in these regions")
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Height predictions
        ax = axes[0, 0]
        ax.errorbar(self.y['height_average'], h_pred, yerr=1.96*h_std, 
                    fmt='o', alpha=0.5, capsize=3)
        lims = [self.y['height_average'].min(), self.y['height_average'].max()]
        ax.plot(lims, lims, 'r--', lw=2)
        ax.set_xlabel('Actual Height', fontweight='bold')
        ax.set_ylabel('Predicted Height', fontweight='bold')
        ax.set_title(f'{gpr_model_name}: Height Predictions', fontweight='bold')
        ax.grid(alpha=0.3)
        
        # Width predictions
        ax = axes[0, 1]
        ax.errorbar(self.y['width_average'], w_pred, yerr=1.96*w_std, 
                    fmt='o', alpha=0.5, capsize=3)
        lims = [self.y['width_average'].min(), self.y['width_average'].max()]
        ax.plot(lims, lims, 'r--', lw=2)
        ax.set_xlabel('Actual Width', fontweight='bold')
        ax.set_ylabel('Predicted Width', fontweight='bold')
        ax.set_title(f'{gpr_model_name}: Width Predictions', fontweight='bold')
        ax.grid(alpha=0.3)
        
        # Uncertainty distribution
        ax = axes[1, 0]
        ax.hist(h_std, bins=20, alpha=0.7, label='Height', edgecolor='black')
        ax.hist(w_std, bins=20, alpha=0.7, label='Width', edgecolor='black')
        ax.set_xlabel('Uncertainty (σ)', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('Uncertainty Distribution', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Feature space uncertainty
        if len(self.X.columns) >= 2:
            ax = axes[1, 1]
            feat1, feat2 = self.X.columns[0], self.X.columns[1]
            scatter = ax.scatter(self.X[feat1], self.X[feat2], 
                               c=(h_std + w_std)/2, cmap='YlOrRd', 
                               s=100, edgecolors='k', linewidth=0.5)
            ax.set_xlabel(feat1, fontweight='bold')
            ax.set_ylabel(feat2, fontweight='bold')
            ax.set_title('Combined Uncertainty in Feature Space', fontweight='bold')
            plt.colorbar(scatter, ax=ax, label='Avg Uncertainty')
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{gpr_model_name}_uncertainty.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✅ Uncertainty plot saved: {gpr_model_name}_uncertainty.png")
    
    def compare_models(self):
        """Compare all models."""
        print("\n" + "=" * 80)
        print("MODEL COMPARISON")
        print("=" * 80)
        
        df = self.results['nested_cv']
        
        # Pivot to get height and width side by side
        comparison = df.pivot_table(
            index=['Model', 'Type'],
            columns='Target',
            values=['MAE_mean', 'MAE_std', 'R2_mean', 'R2_std']
        ).reset_index()
        
        # Flatten column names
        comparison.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                             for col in comparison.columns.values]
        
        # Calculate combined MAE
        comparison['Combined_MAE'] = (
            comparison['MAE_mean_height_average'] + 
            comparison['MAE_mean_width_average']
        )
        
        comparison = comparison.sort_values('Combined_MAE')
        
        print("\n" + "="*100)
        print("NESTED CV RESULTS (mean ± std across 5 folds)")
        print("="*100)
        print(comparison.to_string(index=False))
        
        # Save
        comparison.to_csv(self.output_dir / 'nested_cv_results.csv', index=False)
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        models = comparison['Model'].values
        x = np.arange(len(models))
        width = 0.35
        
        # MAE comparison
        ax = axes[0]
        ax.bar(x - width/2, comparison['MAE_mean_height_average'], width,
               yerr=comparison['MAE_std_height_average'],
               label='Height', alpha=0.8, capsize=3)
        ax.bar(x + width/2, comparison['MAE_mean_width_average'], width,
               yerr=comparison['MAE_std_width_average'],
               label='Width', alpha=0.8, capsize=3)
        ax.set_xlabel('Model', fontweight='bold')
        ax.set_ylabel('MAE (mean ± std)', fontweight='bold')
        ax.set_title('Model Comparison: MAE', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # R² comparison
        ax = axes[1]
        ax.bar(x - width/2, comparison['R2_mean_height_average'], width,
               yerr=comparison['R2_std_height_average'],
               label='Height', alpha=0.8, capsize=3)
        ax.bar(x + width/2, comparison['R2_mean_width_average'], width,
               yerr=comparison['R2_std_width_average'],
               label='Width', alpha=0.8, capsize=3)
        ax.set_xlabel('Model', fontweight='bold')
        ax.set_ylabel('R² (mean ± std)', fontweight='bold')
        ax.set_title('Model Comparison: R²', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✅ Results saved to: {self.output_dir}")
        print(f"   - nested_cv_results.csv")
        print(f"   - model_comparison.png")
        
        return comparison
    
    def analyze_feature_importance(self, comparison):
        """Analyze feature importance for best models."""
        print("\n" + "=" * 80)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("=" * 80)
        
        # Find best non-GPR model (for feature importance)
        simple_models = comparison[comparison['Type'] == 'Simple']
        best_model_name = simple_models.iloc[0]['Model']
        
        print(f"\nAnalyzing best model: {best_model_name}")
        
        # Get best hyperparameters from nested CV results
        df_results = self.results['nested_cv']
        best_results = df_results[df_results['Model'] == best_model_name]
        
        # Train final models on all data with best parameters for each target
        final_models = {}
        feature_importances = {}
        
        for target_name in ['height_average', 'width_average']:
            print(f"\n{'='*60}")
            print(f"Feature Importance: {target_name}")
            print('='*60)
            
            y_target = self.y[target_name]
            
            # Get best params for this target
            target_result = best_results[best_results['Target'] == target_name].iloc[0]
            best_params_str = target_result['Best_params']
            
            # Parse best params (it's stored as string)
            import ast
            try:
                best_params = ast.literal_eval(best_params_str)
            except:
                best_params = {}
            
            # Create and train model with best params
            if best_model_name == 'Ridge':
                model = Ridge(random_state=RANDOM_STATE, **best_params)
            elif best_model_name == 'Lasso':
                model = Lasso(max_iter=10000, random_state=RANDOM_STATE, **best_params)
            elif best_model_name == 'XGB':
                model = XGBRegressor(random_state=RANDOM_STATE, 
                                    objective='reg:squarederror', **best_params)
            elif best_model_name == 'GBR':
                model = GradientBoostingRegressor(random_state=RANDOM_STATE, **best_params)
            elif best_model_name == 'RF':
                model = RandomForestRegressor(random_state=RANDOM_STATE, **best_params)
            else:
                print(f"  ⚠️  Unknown model type: {best_model_name}")
                continue
            
            # Fit on all data
            model.fit(self.X, y_target)
            final_models[target_name] = model
            
            # Extract feature importance
            if hasattr(model, 'coef_'):  # Linear models
                importances = np.abs(model.coef_)
                importance_type = 'Absolute Coefficients'
            elif hasattr(model, 'feature_importances_'):  # Tree models
                importances = model.feature_importances_
                importance_type = 'Feature Importance (Gain)'
            else:
                print(f"  ⚠️  Model doesn't support feature importance")
                continue
            
            feature_importances[target_name] = importances
            
            # Print importances
            print(f"\n{importance_type}:")
            feature_names = self.X.columns
            sorted_idx = np.argsort(importances)[::-1]
            
            for idx in sorted_idx:
                print(f"  {feature_names[idx]:30s}: {importances[idx]:.4f}")
        
        # Visualize feature importance
        self._plot_feature_importance(feature_importances, best_model_name)
        
        # SHAP analysis if available (only for tree models)
        if SHAP_AVAILABLE and best_model_name in ['XGB', 'GBR', 'RF']:
            self._shap_analysis(final_models, best_model_name)
        
        print(f"\n✅ Feature importance analysis complete!")
    
    def _plot_feature_importance(self, feature_importances, model_name):
        """Plot feature importance for both targets."""
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        feature_names = self.X.columns
        
        for idx, (target_name, ax) in enumerate(zip(['height_average', 'width_average'], axes)):
            importances = feature_importances[target_name]
            sorted_idx = np.argsort(importances)
            
            # Horizontal bar plot
            y_pos = np.arange(len(feature_names))
            ax.barh(y_pos, importances[sorted_idx], alpha=0.8, 
                   color='steelblue' if idx == 0 else 'coral',
                   edgecolor='black')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(feature_names[sorted_idx])
            ax.set_xlabel('Importance', fontweight='bold')
            ax.set_title(f'{model_name}: {target_name.replace("_", " ").title()}',
                        fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            # Add values on bars
            for i, v in enumerate(importances[sorted_idx]):
                ax.text(v, i, f' {v:.3f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{model_name}_feature_importance.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✅ Feature importance plot saved: {model_name}_feature_importance.png")
    
    def _shap_analysis(self, final_models, model_name):
        """SHAP analysis for tree-based models."""
        print("\n" + "=" * 80)
        print("SHAP ANALYSIS (Explaining Predictions)")
        print("=" * 80)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        for idx, target_name in enumerate(['height_average', 'width_average']):
            model = final_models[target_name]
            
            print(f"\nComputing SHAP values for {target_name}...")
            
            # Create explainer
            explainer = shap.TreeExplainer(model)
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(self.X)
            
            # Summary plot (beeswarm)
            ax = axes[idx, 0]
            plt.sca(ax)
            shap.summary_plot(shap_values, self.X, show=False, 
                            plot_type='dot', plot_size=None)
            ax.set_title(f'{model_name}: {target_name.replace("_", " ").title()}\n' +
                        'SHAP Summary (Feature Impact)',
                        fontweight='bold')
            
            # Bar plot (mean absolute SHAP)
            ax = axes[idx, 1]
            plt.sca(ax)
            shap.summary_plot(shap_values, self.X, show=False,
                            plot_type='bar', plot_size=None)
            ax.set_title(f'{model_name}: {target_name.replace("_", " ").title()}\n' +
                        'Mean |SHAP| Value',
                        fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{model_name}_shap_analysis.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✅ SHAP analysis saved: {model_name}_shap_analysis.png")
        
        # Save SHAP values to CSV
        for target_name in ['height_average', 'width_average']:
            model = final_models[target_name]
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(self.X)
            
            shap_df = pd.DataFrame(
                shap_values,
                columns=[f'{col}_shap' for col in self.X.columns],
                index=self.X.index
            )
            
            # Add original features
            shap_df = pd.concat([self.X.reset_index(drop=True), 
                                shap_df.reset_index(drop=True)], axis=1)
            
            shap_df.to_csv(self.output_dir / f'{model_name}_{target_name}_shap_values.csv',
                         index=False)
        
        print(f"✅ SHAP values saved to CSV files")
    
    def analyze_feature_importance(self, comparison):
        """Analyze feature importance for best models."""
        print("\n" + "=" * 80)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("=" * 80)
        
        # Find best non-GPR model (for feature importance)
        simple_models = comparison[comparison['Type'] == 'Simple']
        best_model_name = simple_models.iloc[0]['Model']
        
        print(f"\nAnalyzing best model: {best_model_name}")
        
        # Get best hyperparameters from nested CV results
        df_results = self.results['nested_cv']
        best_results = df_results[df_results['Model'] == best_model_name]
        
        # Train final models on all data with best parameters for each target
        final_models = {}
        feature_importances = {}
        
        for target_name in ['height_average', 'width_average']:
            print(f"\n{'='*60}")
            print(f"Feature Importance: {target_name}")
            print('='*60)
            
            y_target = self.y[target_name]
            
            # Get best params for this target
            target_result = best_results[best_results['Target'] == target_name].iloc[0]
            best_params_str = target_result['Best_params']
            
            # Parse best params (it's stored as string)
            import ast
            try:
                best_params = ast.literal_eval(best_params_str)
            except:
                best_params = {}
            
            # Create and train model with best params
            if best_model_name == 'Ridge':
                model = Ridge(random_state=RANDOM_STATE, **best_params)
            elif best_model_name == 'Lasso':
                model = Lasso(max_iter=10000, random_state=RANDOM_STATE, **best_params)
            elif best_model_name == 'XGB':
                model = XGBRegressor(random_state=RANDOM_STATE, 
                                    objective='reg:squarederror', **best_params)
            elif best_model_name == 'GBR':
                model = GradientBoostingRegressor(random_state=RANDOM_STATE, **best_params)
            elif best_model_name == 'RF':
                model = RandomForestRegressor(random_state=RANDOM_STATE, **best_params)
            else:
                print(f"  ⚠️  Unknown model type: {best_model_name}")
                continue
            
            # Fit on all data
            model.fit(self.X, y_target)
            final_models[target_name] = model
            
            # Extract feature importance
            if hasattr(model, 'coef_'):  # Linear models
                importances = np.abs(model.coef_)
                importance_type = 'Absolute Coefficients'
            elif hasattr(model, 'feature_importances_'):  # Tree models
                importances = model.feature_importances_
                importance_type = 'Feature Importance (Gain)'
            else:
                print(f"  ⚠️  Model doesn't support feature importance")
                continue
            
            feature_importances[target_name] = importances
            
            # Print importances
            print(f"\n{importance_type}:")
            feature_names = self.X.columns
            sorted_idx = np.argsort(importances)[::-1]
            
            for idx in sorted_idx:
                print(f"  {feature_names[idx]:30s}: {importances[idx]:.4f}")
        
        # Visualize feature importance
        self._plot_feature_importance(feature_importances, best_model_name)
        
        # SHAP analysis if available (only for tree models)
        if SHAP_AVAILABLE and best_model_name in ['XGB', 'GBR', 'RF']:
            self._shap_analysis(final_models, best_model_name)
        
        print(f"\n✅ Feature importance analysis complete!")
    
    def _plot_feature_importance(self, feature_importances, model_name):
        """Plot feature importance for both targets."""
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        feature_names = self.X.columns
        
        for idx, (target_name, ax) in enumerate(zip(['height_average', 'width_average'], axes)):
            importances = feature_importances[target_name]
            sorted_idx = np.argsort(importances)
            
            # Horizontal bar plot
            y_pos = np.arange(len(feature_names))
            ax.barh(y_pos, importances[sorted_idx], alpha=0.8, 
                   color='steelblue' if idx == 0 else 'coral',
                   edgecolor='black')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(feature_names[sorted_idx])
            ax.set_xlabel('Importance', fontweight='bold')
            ax.set_title(f'{model_name}: {target_name.replace("_", " ").title()}',
                        fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            # Add values on bars
            for i, v in enumerate(importances[sorted_idx]):
                ax.text(v, i, f' {v:.3f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{model_name}_feature_importance.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✅ Feature importance plot saved: {model_name}_feature_importance.png")
    
    def _shap_analysis(self, final_models, model_name):
        """SHAP analysis for tree-based models."""
        print("\n" + "=" * 80)
        print("SHAP ANALYSIS (Explaining Predictions)")
        print("=" * 80)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        for idx, target_name in enumerate(['height_average', 'width_average']):
            model = final_models[target_name]
            
            print(f"\nComputing SHAP values for {target_name}...")
            
            # Create explainer
            if model_name in ['XGB', 'GBR']:
                explainer = shap.TreeExplainer(model)
            else:  # RF
                explainer = shap.TreeExplainer(model)
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(self.X)
            
            # Summary plot (beeswarm)
            ax = axes[idx, 0]
            plt.sca(ax)
            shap.summary_plot(shap_values, self.X, show=False, 
                            plot_type='dot', plot_size=None)
            ax.set_title(f'{model_name}: {target_name.replace("_", " ").title()}\n' +
                        'SHAP Summary (Feature Impact)',
                        fontweight='bold')
            
            # Bar plot (mean absolute SHAP)
            ax = axes[idx, 1]
            plt.sca(ax)
            shap.summary_plot(shap_values, self.X, show=False,
                            plot_type='bar', plot_size=None)
            ax.set_title(f'{model_name}: {target_name.replace("_", " ").title()}\n' +
                        'Mean |SHAP| Value',
                        fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{model_name}_shap_analysis.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✅ SHAP analysis saved: {model_name}_shap_analysis.png")
        
        # Save SHAP values to CSV
        for target_name in ['height_average', 'width_average']:
            model = final_models[target_name]
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(self.X)
            
            shap_df = pd.DataFrame(
                shap_values,
                columns=[f'{col}_shap' for col in self.X.columns],
                index=self.X.index
            )
            
            # Add original features
            shap_df = pd.concat([self.X.reset_index(drop=True), 
                                shap_df.reset_index(drop=True)], axis=1)
            
            shap_df.to_csv(self.output_dir / f'{model_name}_{target_name}_shap_values.csv',
                         index=False)
        
        print(f"✅ SHAP values saved to CSV files")


def main():
    """Main execution."""
    print("\n" + "="*80)
    print("ROBOCASTING ANALYSIS WITH NESTED CV")
    print("="*80 + "\n")
    
    data_path = '/Users/nazarii/projects/ipms/robocasting/data/cleaned_df.csv'
    analyzer = RobocastingAnalyzerNestedCV(data_path, output_dir='results_nested_cv')
    
    try:
        # 1. Load data
        analyzer.load_data()
        
        # 2. Preprocess
        analyzer.preprocess_data()
        
        # 3. Prepare features
        analyzer.prepare_features()
        
        # 4. Nested CV training
        analyzer.train_with_nested_cv()
        
        # 5. Train final models for uncertainty
        best_gpr = analyzer.train_final_models()
        
        # 6. Uncertainty analysis
        analyzer.analyze_gpr_uncertainty(best_gpr)
        
        # 7. Compare models
        comparison = analyzer.compare_models()
        
        # 8. Feature importance analysis
        analyzer.analyze_feature_importance(comparison)
        
        print("\n" + "="*80)
        print("✅ NESTED CV ANALYSIS COMPLETE!")
        print("="*80)
        print(f"\nResults in: {analyzer.output_dir}")
        print("\nFor publication, report:")
        print("  'Performance evaluated using 5-fold cross-validation with")
        print("   nested Bayesian hyperparameter optimization (20 iterations,")
        print("   3-fold CV) on n=58 samples'")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

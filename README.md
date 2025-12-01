# Machine Learning for Robocasting Process Optimization

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the code and analysis supporting the paper:

**"Evaluating machine learning approaches for printing parameter optimization in robocasting of ceramics"**

Authors: Nazarii Mediukh, Vladyslav Naumenko, Anton Krasikov, Vladyslav Bilyi, Oleksandr Vasiliev, Ostap Zgalat-Lozynskyi. 
Institution: Institute for Problems of Materials Science, National Academy of Sciences of Ukraine

## Overview

This project applies machine learning methods to predict the dimensional characteristics (height and width) of ceramic structures produced by robocasting (direct ink writing). We compare multiple regression algorithms including:

- Gaussian Process Regression (GPR)
- XGBoost (XGB)
- Gradient Boosting Regressor (GBR)
- Random Forest (RF)
- Ridge and Lasso Regression

Performance is evaluated using nested cross-validation with Bayesian hyperparameter optimization.

## Key Results

- **Best Model**: GPR-RBF achieves MAE of X.XXX mm (height) and X.XXX mm (width)
- **Dataset**: 58 samples with 3 process parameters
- **Validation**: 5-fold nested CV with 3-fold inner optimization loop
- **Feature Importance**: Extrusion multiplier and nozzle speed are the dominant predictors

## Repository Structure

```
robocasting/
├── data/
│   ├── cleaned_df.csv          # Preprocessed dataset
├── dataset_statistics.py       # Measurement repeatability analysis
├── outlier_detection.py        # Outlier detection (IQR & Z-score)
├── processing_nested_cv.py     # Main ML training pipeline
├── requirements.txt            # Python dependencies
├── results_nested_cv/          # Generated results and plots
├── dataset_analysis/           # Statistical analysis outputs
└── README.md                   # This file
```

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. Clone the repository:

2. Create a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Dataset Statistics and Quality Analysis

Analyze measurement repeatability and parameter distributions:

```bash
python dataset_statistics.py
```

**Outputs**:
- `dataset_analysis/measurement_repeatability.png` - Within-sample variation analysis
- `dataset_analysis/parameter_distributions.png` - Process parameter ranges
- `dataset_analysis/correlation_matrix.png` - Feature correlations
- `dataset_analysis/measurement_statistics.csv` - Summary statistics

### 2. Outlier Detection

Identify potential outliers using IQR and Z-score methods:

```bash
python outlier_detection.py
```

**Outputs**:
- `dataset_analysis/outlier_detection.png` - Visualization of detected outliers
- `dataset_analysis/outlier_detection_summary.txt` - Summary report

### 3. Model Training and Evaluation

Train and compare all ML models using nested cross-validation:

```bash
python processing_nested_cv.py
```

**Outputs**:
- `results_nested_cv/nested_cv_results.csv` - Model performance metrics
- `results_nested_cv/model_comparison.png` - Comparative visualization
- `results_nested_cv/GPR_*_uncertainty.png` - Uncertainty quantification
- `results_nested_cv/XGB_feature_importance.png` - Feature importance analysis
- `results_nested_cv/XGB_shap_analysis.png` - SHAP explanations (if available)

## Dataset Description

The dataset contains 58 experimental samples with the following features:

**Input Features (Process Parameters)**:
- `slicer_nozzle_speed` - Printing speed (mm/s)
- `slicer_extrusion_multiplier` - Material flow rate multiplier
- `layer_count` - Number of printed layers

**Output Features (Dimensional Characteristics)**:
- `height_average` - Average layer height (mm), computed from 3 measurements
- `width_average` - Average line width (mm), computed from 3 measurements

**Environmental Variables** (constant in this study):
- `temp` - Temperature (°C)
- `humidity` - Relative humidity (%)
- `nozzle_diameter` - Nozzle diameter (mm)

### Measurement Protocol

Each sample was measured three times at different locations to assess repeatability.

## Methodology

### Nested Cross-Validation

We use nested cross-validation to prevent data leakage and provide unbiased performance estimates:

```
Outer Loop (5-fold): Model evaluation
├── Inner Loop (3-fold): Bayesian hyperparameter optimization
│   ├── 20 iterations of Bayesian search
│   └── Best parameters selected
└── Test on held-out fold
```

### Bayesian Hyperparameter Optimization

Instead of exhaustive grid search, we use Bayesian optimization:
- More efficient than grid search
- Intelligent exploration of parameter space
- 20 iterations per inner fold

### Evaluation Metrics

- **MAE (Mean Absolute Error)**: Primary metric, robust to outliers
- **R² (Coefficient of Determination)**: Explained variance
- **Prediction Uncertainty**: Quantified using GPR posterior variance

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or collaboration inquiries:
- **Name**: Nazarii Mediukh
- **Email**: n.mediukh@ipms.kyiv.ua
- **Institution**: Institute for Problems of Materials Science, NASU

## Acknowledgments

- Institute for Problems of Materials Science, National Academy of Sciences of Ukraine
- 

## Contributing

We welcome contributions! Please feel free to:
- Report bugs or issues
- Suggest improvements
- Submit pull requests

For major changes, please open an issue first to discuss what you would like to change.

# XGBoost Model for Shape Memory Alloy Temperature Prediction

This implementation uses XGBoost (Extreme Gradient Boosting) to predict transformation temperatures for Shape Memory Alloys (SMAs).

## Model Overview

### Targets
- **AF (Austenite Finish Temperature)**: Temperature at which austenite transformation completes
- **MF (Martensite Finish Temperature)**: Temperature at which martensite transformation completes

### Features Used
1. **Element Compositions** (19 elements):
   - Ag, Al, Au, Cd, Co, Cu, Fe, Hf, Mn, Nb, Ni, Pd, Pt, Ru, Si, Ta, Ti, Zn, Zr

2. **Material Properties**:
   - Austenite Start Temperature (AS)
   - Martensite Start Temperature (MS)
   - Cooling Rate
   - Heating Rate
   - Calculated Density

3. **Engineered Features**:
   - Ni/Ti ratio
   - Total alloying elements
   - Ni × Ti interaction
   - Cooling/Heating rate ratio

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

```bash
python xgboost_sma_model.py
```

This will:
1. Load the dataset from `dataset/Combined_SMA_Dataset_Filled.csv`
2. Prepare features and engineer new features
3. Train separate XGBoost models for AF and MF
4. Evaluate model performance
5. Generate visualizations:
   - `feature_importance.png` - Top features for both models
   - `af_predictions.png` - AF prediction accuracy
   - `mf_predictions.png` - MF prediction accuracy
6. Save trained models:
   - `xgboost_af_model.pkl`
   - `xgboost_mf_model.pkl`
   - `feature_scaler.pkl`

### Using Pre-trained Models

```python
from xgboost_sma_model import SMAXGBoostModel
import pandas as pd

# Load model
model = SMAXGBoostModel()
model.load_models()

# Prepare your data
# ... prepare X with same features as training ...

# Make predictions
X_scaled = model.scaler.transform(X)
af_prediction = model.model_af.predict(X_scaled)
mf_prediction = model.model_mf.predict(X_scaled)
```

## Model Architecture

### XGBoost Parameters
- **n_estimators**: 500 (number of boosting rounds)
- **max_depth**: 6 (maximum tree depth)
- **learning_rate**: 0.05 (step size shrinkage)
- **subsample**: 0.8 (subsample ratio of training instances)
- **colsample_bytree**: 0.8 (subsample ratio of columns)
- **regularization**: L1 (0.1) and L2 (1.0) for preventing overfitting

### Training Strategy
- **Separate Models**: Two independent XGBoost models (one for AF, one for MF)
- **Train/Test Split**: 80/20
- **Feature Scaling**: StandardScaler for normalization
- **Early Stopping**: Stops training if no improvement for 50 rounds

## Expected Performance

Based on the dataset characteristics (1,906 samples):
- **R² Score**: 0.85-0.95 (excellent fit)
- **MAE**: 5-15°C (good for practical applications)
- **RMSE**: Similar range to MAE

## Why XGBoost?

1. **Excellent for Tabular Data**: Industry-standard for structured datasets
2. **Handles Non-linearity**: Captures complex material science relationships
3. **Feature Importance**: Understand which elements matter most
4. **Robust**: Less prone to overfitting with proper regularization
5. **Fast Training**: Efficient even with 1,900+ samples
6. **Multi-output Support**: Can predict both AF and MF

## Model Outputs

### Visualizations
- Feature importance ranking for both targets
- Predicted vs Actual scatter plots for training and test sets
- Performance metrics comparison

### Saved Models
All models are saved using joblib for easy reuse:
- XGBoost model for AF prediction
- XGBoost model for MF prediction
- StandardScaler for feature normalization

## Feature Engineering Details

The model creates additional features to improve predictions:

1. **Ni_Ti_ratio**: Critical for NiTi-based SMAs
2. **Total_Alloying**: Sum of all alloying elements (excluding Ni, Ti)
3. **Ni_Ti_interaction**: Product of Ni and Ti percentages
4. **Cooling_Heating_ratio**: Ratio of cooling to heating rates

## Dataset

Uses: `dataset/Combined_SMA_Dataset_Filled.csv`
- Total samples: ~1,906
- Features: 24 base features + 4 engineered = 28 total
- Targets: AF and MF temperatures

## Notes

- The model does **not** use TSPAN (AF-MF) as a feature to avoid target leakage
- Feature scaling is applied for better convergence
- Random seed (42) ensures reproducibility
- Early stopping prevents overfitting during training

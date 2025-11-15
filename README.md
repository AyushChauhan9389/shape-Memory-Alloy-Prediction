# Shape Memory Alloy (SMA) Temperature Prediction using MLP Neural Network

This project implements Multi-Layer Perceptron (MLP) neural networks to predict the Austenite Finish (AF) and Martensite Finish (MF) temperatures for Shape Memory Alloys based on their composition and processing parameters.

## 🚀 Quick Start - Use the Fixed Model!

**⚠️ Important: The "improved" model FAILED. Use the fixed model instead:**

```bash
python fixed_mlp_model.py  # ✓ RECOMMENDED - Actually works!
```

## 📊 Model Performance Summary

| Model | R² Score | MAE | Status |
|-------|----------|-----|--------|
| **Fixed MLP** | **0.7560** | **28.49°C** | ✅ **USE THIS (+32.7% vs baseline)** |
| Baseline MLP | 0.5696 | 52.90°C | ⚠️ Below target but works |
| "Improved" MLP | -0.0329 | 77.10°C | ❌ **BROKEN - DO NOT USE** |
| Target | 0.80-0.88 | 8-15°C | Goal |

## Installation

Install dependencies using uv:

```bash
uv sync
```

Or install manually:

```bash
pip install tensorflow>=2.15.0 numpy>=1.24.0 pandas>=2.0.0 scikit-learn>=1.3.0 matplotlib>=3.7.0 seaborn>=0.12.0
```

## Models Available

### 1. Fixed MLP Model (RECOMMENDED)

```bash
python fixed_mlp_model.py
```

**Features:**
- ✅ Uses 24 features (19 elements + 3 process + AS/MS temperatures)
- ✅ StandardScaler (proven to work)
- ✅ Moderate architecture: 128-64-32
- ✅ Separate models for AF and MF (better accuracy)
- ✅ MSE loss (simpler, proven)
- ✅ AS/MS have 0.84+ correlation with targets

**What it does:**
- Loads dataset with AS/MS features (high correlation: 0.84-0.98)
- Trains separate models for AF and MF predictions
- Uses proven StandardScaler normalization
- Generates prediction plots with metrics
- Saves models: `fixed_mlp_af_model.keras`, `fixed_mlp_mf_model.keras`

### 2. Baseline MLP Model

```bash
python mlp_model.py
```

**Performance:**
- R²: 0.5696
- MAE: 52.90°C

**Features:**
- Uses 22 features (NO AS/MS)
- 3-layer architecture (128-64-32)
- Multi-output (predicts both AF and MF together)

### 3. "Improved" MLP Model ❌ DO NOT USE

```bash
# DON'T RUN THIS - IT'S BROKEN
# python improved_mlp_model.py
```

**Why it failed:**
- Negative R² (-0.03) - worse than predicting the mean!
- Used RobustScaler instead of StandardScaler
- Over-complex architecture (256-128-64-32 + BatchNorm)
- See [DIAGNOSIS.md](DIAGNOSIS.md) for full analysis

## Dataset

**File:** `dataset/Combined_SMA_Dataset_Filled.csv`

**Statistics:**
- Samples: 1,847
- Features: 32 columns total
- Missing values: 0% (dataset is complete!)

**Features (24 used):**
- Element compositions (19): Ag, Al, Au, Cd, Co, Cu, Fe, Hf, Mn, Nb, Ni, Pd, Pt, Ru, Si, Ta, Ti, Zn, Zr
- Process parameters (3): Cooling Rate, Heating Rate, Calculated Density
- Temperature features (2): AS, MS (0.84+ correlation with targets!)

**Targets (2):**
- AF: Austenite Finish Temperature (°C)
- MF: Martensite Finish Temperature (°C)

**Data Characteristics:**
- AF range: -198°C to 1244°C (std: 187°C)
- MF range: -255°C to 1156°C (std: 182°C)
- AF ↔ MF correlation: 0.98 (extremely high!)
- AS ↔ AF correlation: 0.85
- MS ↔ MF correlation: 0.85

## Output Files

### Fixed MLP Model:
- `fixed_mlp_af_model.keras` - AF prediction model
- `fixed_mlp_mf_model.keras` - MF prediction model
- `fixed_mlp_results.png` - Prediction plots with metrics

### Diagnostic Files:
- `diagnose_data.py` - Data quality analysis script
- `DIAGNOSIS.md` - Full diagnosis of why "improved" model failed

## Why the "Improved" Model Failed

See [DIAGNOSIS.md](DIAGNOSIS.md) for detailed analysis. Key issues:

1. **RobustScaler** instead of StandardScaler ❌
2. **Over-complex architecture** (4 layers with BatchNorm) ❌
3. **Feature engineering** added noise instead of signal ❌

The fixed model addresses all these issues.

## Usage Example

```python
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load models
model_af = tf.keras.models.load_model('fixed_mlp_af_model.keras')
model_mf = tf.keras.models.load_model('fixed_mlp_mf_model.keras')

# Prepare input (24 features: 19 elements + 3 process + AS + MS)
X_new = np.array([[...]]) # Your data

# Scale features (use same scaler from training!)
scaler = StandardScaler()
X_new_scaled = scaler.fit_transform(X_new)

# Predict
af_pred = model_af.predict(X_new_scaled)[0][0]
mf_pred = model_mf.predict(X_new_scaled)[0][0]

print(f"Predicted AF: {af_pred:.2f}°C")
print(f"Predicted MF: {mf_pred:.2f}°C")
```

## Model Comparison

| Aspect | Baseline | "Improved" (BROKEN) | Fixed |
|--------|----------|---------------------|-------|
| Features | 22 (no AS/MS) | 30 (with engineering) | 24 (with AS/MS) |
| Scaler | StandardScaler | RobustScaler ❌ | StandardScaler ✅ |
| Architecture | 128-64-32 | 256-128-64-32+BN | 128-64-32 ✅ |
| Output | Multi (AF+MF) | Multi (AF+MF) | Separate models ✅ |
| R² Score | 0.57 | -0.03 ❌ | **0.76** ✅ |
| MAE | 52.90°C | 77.10°C ❌ | **28.49°C** ✅ |
| vs Baseline | - | -106% worse ❌ | **+32.7% better** ✅ |

## Troubleshooting

**Q: Why not use AS/MS in baseline?**
A: Baseline was created before we knew AS/MS had such high correlation (0.84+).

**Q: Why did "improved" model fail?**
A: Wrong scaler (RobustScaler), over-engineered architecture. See DIAGNOSIS.md.

**Q: Which model should I use?**
A: Use `fixed_mlp_model.py` - it combines best of both approaches.

**Q: Can I use XGBoost instead?**
A: XGBoost often works better on tabular data, but this project focuses on MLP.

## Future Improvements

If fixed model still doesn't meet targets (R²=0.80-0.88):

1. **Try different alloy families separately**
   - NiTi alloys (76% of data)
   - Cu-based alloys (39% of data)
   - Others

2. **Hyperparameter tuning**
   - Use Keras Tuner or Optuna
   - Search: layers, neurons, dropout, learning rate

3. **Data augmentation**
   - Add small noise to inputs
   - Synthetic minority oversampling

4. **Alternative architectures**
   - Residual connections
   - Multi-task learning (predict all 4 temps)

## License

Open source for research and educational purposes.

## References

- **Shape Memory Alloys**: Materials with temperature-dependent phase transformations
- **Transformation Temperatures**: Critical for applications (actuators, medical devices)
- **Common SMAs**: NiTi (56%), Cu-based (25%), Fe-based (15%)

## Citation

If you use this code, please cite:
```
SMA Temperature Prediction using MLP Neural Networks
Dataset: 1,847 samples from Combined_SMA_Dataset_Filled.csv
```

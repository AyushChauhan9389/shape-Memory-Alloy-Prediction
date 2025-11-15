# Diagnosis: Why the "Improved" Model Failed

## Problem Summary

| Model | R² Score | MAE | Status |
|-------|----------|-----|--------|
| Baseline | 0.5696 | 52.90°C | ⚠ Below target but working |
| "Improved" | -0.0329 | 77.10°C | ❌ **COMPLETE FAILURE** |
| Target | 0.80-0.88 | 8-15°C | Goal |

**Critical Issue**: The "improved" model has **negative R²**, meaning it's worse than just predicting the mean value!

## Root Cause Analysis

### Data Quality (✓ Not the problem)
- **Missing values**: 0% for all features including AS/MS
- **Correlations are excellent**:
  - AF ↔ MF: 0.9768 (extremely high)
  - AS ↔ AF: 0.8451 (very high)
  - MS ↔ MF: 0.8466 (very high)
- Data is complete and well-structured

### Actual Problems Identified

#### 1. **RobustScaler vs StandardScaler** ❌
- **Changed from**: Standard Scaler (baseline)
- **Changed to**: RobustScaler ("improved")
- **Problem**: RobustScaler uses median/IQR which can be problematic for:
  - Element percentages (0-100%)
  - Temperatures with huge ranges (-255°C to 1244°C)
  - Mixed feature types with different distributions

#### 2. **Over-complex Architecture** ❌
- **Baseline**: 128 → 64 → 32 (works, R²=0.57)
- **"Improved"**: 256 → 128 → 64 → 32 + BatchNorm
- **Problem**:
  - 4x more parameters in first layer
  - Batch Normalization can cause issues with small datasets
  - More capacity = more overfitting risk

#### 3. **Training Configuration** ❌
- **Smaller batch size**: 32 vs 64
  - More noise in gradients
  - Can help generalization BUT can also cause instability
- **Longer patience**: 30 vs 20
  - Model stopped at epoch 69 (vs baseline at epoch 98)
  - Might not have converged properly

#### 4. **Feature Engineering Backfire** ❌
- Added 8 engineered features (NiTi_ratio, etc.)
- **Problem**: Without proper feature selection, these might add noise
- Baseline used simple features and worked better

## Why Negative R²?

Negative R² means: **Sum of Squared Errors > Total Variance**

This happens when:
1. **Model predictions are systematically biased**
   - Possibly due to wrong scaling (RobustScaler)
2. **Model didn't learn the relationship**
   - Over-parameterization caused failure to generalize
3. **Train/test distribution mismatch**
   - Unlikely since baseline worked on same split

## Key Insight

**Adding complexity ≠ Better performance**

The baseline model with:
- Simple features (element compositions + process parameters)
- StandardScaler
- 3-layer architecture
- Regular training

Achieved R²=0.57 which, while below target, is **WAY better than negative R²**!

## Correct Approach

### What Actually Works (from baseline):
1. ✅ StandardScaler for normalization
2. ✅ Element compositions (19) + process parameters (3) = 22 features
3. ✅ Moderate architecture: 128-64-32
4. ✅ Standard training: batch=64, early stopping

### What We Should Try:
1. **Keep what works** from baseline
2. **Add AS/MS features carefully** - they have 0.84+ correlation!
3. **Use StandardScaler** (not RobustScaler)
4. **Moderate architecture** (not too deep)
5. **Better loss function**: Maybe MSE instead of Huber
6. **Separate models** for AF and MF (instead of multi-output)

## Hypothesis

The baseline achieved only R²=0.57 because:
1. **Huge temperature ranges**: -255°C to 1244°C (1400°C range!)
2. **Different alloy families**: NiTi, CuAl, FeMn have very different behaviors
3. **Non-linear relationships**: Neural networks might not be ideal for this

## Alternative Solution: XGBoost

Since this is **tabular data** with:
- 1,847 samples (small dataset)
- Mixed feature types
- Non-linear relationships
- Different alloy families

**XGBoost or Random Forest might outperform neural networks significantly**

Expected with XGBoost:
- R²: 0.80-0.90 (closer to target)
- MAE: 10-20°C
- Much faster training
- Better interpretability (feature importance)

## Next Steps

1. ✅ Create a **fixed MLP** with proven baseline approach + AS/MS
2. ✅ Implement **XGBoost alternative** for comparison
3. ✅ Use proper evaluation (cross-validation)
4. ✅ Compare results and choose best model

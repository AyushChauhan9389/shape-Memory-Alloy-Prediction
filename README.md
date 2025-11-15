# Shape Memory Alloy (SMA) Temperature Prediction using MLP Neural Network

This project implements a Multi-Layer Perceptron (MLP) neural network to predict the Austenite Finish (AF) and Martensite Finish (MF) temperatures for Shape Memory Alloys based on their composition and processing parameters.

## Model Architecture

**MLP (3-Layer) Neural Network:**
- Input Layer: 22 features (19 element compositions + 3 process parameters)
- Hidden Layer 1: 128 neurons (ReLU activation, Dropout 0.3, L2 regularization 0.001)
- Hidden Layer 2: 64 neurons (ReLU activation, Dropout 0.3, L2 regularization 0.001)
- Hidden Layer 3: 32 neurons (ReLU activation, Dropout 0.3, L2 regularization 0.001)
- Output Layer: 2 neurons (AF and MF temperatures)

## Configuration

- **Optimizer:** Adam (learning rate = 0.001)
- **Loss Function:** Huber Loss (robust to outliers)
- **Batch Size:** 64
- **Epochs:** 200 with early stopping (patience = 20)
- **Regularization:** Dropout(0.3) + L2(0.001)
- **Normalization:** StandardScaler (REQUIRED!)
- **Learning Rate Scheduler:** ReduceLROnPlateau (factor=0.5, patience=10)

## Expected Performance

- **R² Score:** 0.80-0.88
- **MAE:** 8-15°C
- **Training Time:** 2-5 minutes

## Installation

Install dependencies using uv:

```bash
uv sync
```

Or install manually:

```bash
pip install tensorflow>=2.15.0 numpy>=1.24.0 pandas>=2.0.0 scikit-learn>=1.3.0 matplotlib>=3.7.0 seaborn>=0.12.0
```

## Usage

Run the MLP training script:

```bash
python mlp_model.py
```

The script will:
1. Load and preprocess the `Combined_SMA_Dataset_Filled.csv` dataset
2. Split data into train (64%), validation (16%), and test (20%) sets
3. Build and train the MLP model with early stopping
4. Perform 5-fold cross-validation
5. Evaluate performance on the test set
6. Generate visualizations:
   - `mlp_training_history.png` - Training and validation metrics over epochs
   - `mlp_predictions.png` - Predicted vs actual temperatures for AF and MF
   - `mlp_residuals.png` - Residual analysis for model diagnostics
7. Save the trained model to `mlp_sma_model.keras`

## Dataset

The dataset (`dataset/Combined_SMA_Dataset_Filled.csv`) contains 1,906 samples of various SMA compositions.

**Features (22 total):**
- Element compositions (19): Ag, Al, Au, Cd, Co, Cu, Fe, Hf, Mn, Nb, Ni, Pd, Pt, Ru, Si, Ta, Ti, Zn, Zr
- Process parameters (3): Cooling Rate, Heating Rate, Calculated Density

**Targets (2):**
- AF: Austenite Finish Temperature (°C)
- MF: Martensite Finish Temperature (°C)

## Output

After training, you'll get:

1. **Console Output:**
   - Model architecture summary
   - Training progress with early stopping
   - Test set performance metrics (MAE, R², RMSE)
   - Cross-validation results (mean ± std)

2. **Saved Files:**
   - `mlp_sma_model.keras` - Trained model
   - `mlp_training_history.png` - Training curves
   - `mlp_predictions.png` - Prediction scatter plots
   - `mlp_residuals.png` - Residual analysis

## Model Performance Interpretation

- **R² Score:** Measures how well the model explains variance in the data (1.0 = perfect)
- **MAE (Mean Absolute Error):** Average prediction error in °C
- **RMSE (Root Mean Squared Error):** More sensitive to large errors than MAE

## Advanced Usage

To use the trained model for predictions:

```python
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the model
model = tf.keras.models.load_model('mlp_sma_model.keras')

# Prepare your input data (22 features)
# Make sure to scale it using the same StandardScaler fitted on training data
X_new = np.array([[...]])  # Your feature values

# Scale the data
scaler = StandardScaler()
X_new_scaled = scaler.fit_transform(X_new)  # Note: Use the scaler from training

# Make predictions
predictions = model.predict(X_new_scaled)
af_pred, mf_pred = predictions[0]

print(f"Predicted AF: {af_pred:.2f}°C")
print(f"Predicted MF: {mf_pred:.2f}°C")
```

## Technical Details

**Why This Architecture?**
- 3-layer MLP is optimal for small-medium datasets (1,000-2,000 samples)
- Dropout prevents overfitting
- L2 regularization improves generalization
- Huber loss is robust to outliers in temperature data
- Early stopping prevents overtraining

**Cross-Validation:**
- 5-fold CV provides reliable performance estimates
- Reduces variance in performance metrics
- Helps detect overfitting

**Data Normalization:**
- StandardScaler (z-score normalization) is ESSENTIAL for neural networks
- Features have different scales (percentages vs temperatures)
- Improves training stability and convergence

## Troubleshooting

**Low R² or high MAE?**
- Increase epochs (if early stopping triggered too early)
- Adjust learning rate (try 0.0001 or 0.01)
- Reduce dropout (try 0.2)
- Increase model capacity (more neurons or layers)

**Overfitting (train loss << val loss)?**
- Increase dropout (try 0.4)
- Increase L2 regularization (try 0.01)
- Reduce model capacity
- Get more training data

**Training too slow?**
- Reduce batch size (try 32)
- Use GPU (TensorFlow will auto-detect)
- Reduce epochs or patience

## License

This project is open source and available for research and educational purposes.

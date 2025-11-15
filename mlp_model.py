"""
MLP Neural Network for Shape Memory Alloy (SMA) Dataset
Target: Predict AF (Austenite Finish) and MF (Martensite Finish) temperatures

Architecture:
- Input (25 features) → 128 neurons → 64 neurons → 32 neurons → Output (2: AF, MF)
- Activation: ReLU
- Optimizer: Adam (lr=0.001)
- Loss: Huber Loss
- Batch Size: 64
- Epochs: 200 with early stopping (patience=20)
- Regularization: Dropout(0.3) + L2(0.001)
- Normalization: StandardScaler
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class MLPShapeMemoryAlloy:
    """MLP Model for SMA Temperature Prediction"""

    def __init__(self, input_dim=25, learning_rate=0.001):
        """
        Initialize MLP model

        Args:
            input_dim: Number of input features
            learning_rate: Learning rate for Adam optimizer
        """
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.history = None

    def build_model(self):
        """
        Build MLP architecture:
        Input → 128 (ReLU, Dropout 0.3, L2 0.001)
             → 64 (ReLU, Dropout 0.3, L2 0.001)
             → 32 (ReLU, Dropout 0.3, L2 0.001)
             → 2 (AF, MF outputs)
        """
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=(self.input_dim,)),

            # Hidden Layer 1: 128 neurons
            layers.Dense(
                128,
                activation='relu',
                kernel_regularizer=regularizers.l2(0.001),
                name='hidden_layer_1'
            ),
            layers.Dropout(0.3, name='dropout_1'),

            # Hidden Layer 2: 64 neurons
            layers.Dense(
                64,
                activation='relu',
                kernel_regularizer=regularizers.l2(0.001),
                name='hidden_layer_2'
            ),
            layers.Dropout(0.3, name='dropout_2'),

            # Hidden Layer 3: 32 neurons
            layers.Dense(
                32,
                activation='relu',
                kernel_regularizer=regularizers.l2(0.001),
                name='hidden_layer_3'
            ),
            layers.Dropout(0.3, name='dropout_3'),

            # Output layer: 2 neurons (AF, MF)
            layers.Dense(2, name='output_layer')
        ])

        # Compile with Huber loss and Adam optimizer
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=keras.losses.Huber(delta=1.0),
            metrics=['mae', 'mse']
        )

        self.model = model
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=200, batch_size=64, patience=20):
        """
        Train the MLP model with early stopping

        Args:
            X_train: Training features
            y_train: Training targets (AF, MF)
            X_val: Validation features
            y_val: Validation targets
            epochs: Maximum number of epochs
            batch_size: Batch size for training
            patience: Early stopping patience
        """
        # Define callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )

        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        )

        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )

        return self.history

    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X, verbose=0)

    def evaluate(self, X, y):
        """Evaluate model performance"""
        y_pred = self.predict(X)

        # Calculate metrics for AF (index 0)
        mae_af = mean_absolute_error(y[:, 0], y_pred[:, 0])
        r2_af = r2_score(y[:, 0], y_pred[:, 0])
        rmse_af = np.sqrt(mean_squared_error(y[:, 0], y_pred[:, 0]))

        # Calculate metrics for MF (index 1)
        mae_mf = mean_absolute_error(y[:, 1], y_pred[:, 1])
        r2_mf = r2_score(y[:, 1], y_pred[:, 1])
        rmse_mf = np.sqrt(mean_squared_error(y[:, 1], y_pred[:, 1]))

        # Overall metrics
        mae_overall = (mae_af + mae_mf) / 2
        r2_overall = (r2_af + r2_mf) / 2

        return {
            'AF': {'MAE': mae_af, 'R2': r2_af, 'RMSE': rmse_af},
            'MF': {'MAE': mae_mf, 'R2': r2_mf, 'RMSE': rmse_mf},
            'Overall': {'MAE': mae_overall, 'R2': r2_overall}
        }


def load_and_preprocess_data(file_path):
    """
    Load and preprocess the SMA dataset

    Args:
        file_path: Path to the CSV file

    Returns:
        X: Feature matrix
        y: Target matrix (AF, MF)
        feature_names: List of feature names
    """
    # Load data
    df = pd.read_csv(file_path)

    print(f"Dataset shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")

    # Define feature columns (element compositions + process parameters)
    element_features = [
        'Ag (at.%)', 'Al (at.%)', 'Au (at.%)', 'Cd (at.%)', 'Co (at.%)',
        'Cu (at.%)', 'Fe (at.%)', 'Hf (at.%)', 'Mn (at.%)', 'Nb (at.%)',
        'Ni (at.%)', 'Pd (at.%)', 'Pt (at.%)', 'Ru (at.%)', 'Si (at.%)',
        'Ta (at.%)', 'Ti (at.%)', 'Zn (at.%)', 'Zr (at.%)'
    ]

    process_features = [
        'Cooling Rate (°C/min)',
        'Heating Rate (°C/min)',
        'Calculated Density (g/cm^3)'
    ]

    # Target columns
    target_cols = [
        'Austenite Finish Temperature - AF - (°C)',
        'Martensite Finish Temperature - MF - (°C)'
    ]

    # Combine all features
    feature_cols = element_features + process_features

    # Check if all columns exist
    missing_cols = [col for col in feature_cols + target_cols if col not in df.columns]
    if missing_cols:
        print(f"\nWarning: Missing columns: {missing_cols}")

    # Extract features and targets
    X = df[feature_cols].values
    y = df[target_cols].values

    # Remove rows with NaN values
    valid_idx = ~(np.isnan(X).any(axis=1) | np.isnan(y).any(axis=1))
    X = X[valid_idx]
    y = y[valid_idx]

    print(f"\nAfter removing NaN values: {X.shape[0]} samples")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of targets: {y.shape[1]}")

    # Print target statistics
    print(f"\nTarget Statistics:")
    print(f"AF - Mean: {y[:, 0].mean():.2f}°C, Std: {y[:, 0].std():.2f}°C, Range: [{y[:, 0].min():.2f}, {y[:, 0].max():.2f}]")
    print(f"MF - Mean: {y[:, 1].mean():.2f}°C, Std: {y[:, 1].std():.2f}°C, Range: [{y[:, 1].min():.2f}, {y[:, 1].max():.2f}]")

    return X, y, feature_cols


def cross_validation_evaluation(X, y, n_splits=5):
    """
    Perform k-fold cross-validation

    Args:
        X: Feature matrix
        y: Target matrix
        n_splits: Number of folds
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    cv_results = {
        'AF_MAE': [], 'AF_R2': [], 'AF_RMSE': [],
        'MF_MAE': [], 'MF_R2': [], 'MF_RMSE': []
    }

    print(f"\n{'='*60}")
    print(f"Performing {n_splits}-Fold Cross-Validation")
    print(f"{'='*60}")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        print(f"\nFold {fold}/{n_splits}")
        print("-" * 40)

        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Scale features
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_val_scaled = scaler_X.transform(X_val)

        # Build and train model
        mlp = MLPShapeMemoryAlloy(input_dim=X_train_scaled.shape[1])
        mlp.build_model()

        # Train with reduced verbosity for CV
        mlp.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=200,
            batch_size=64,
            callbacks=[
                callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=0),
                callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7, verbose=0)
            ],
            verbose=0
        )

        # Evaluate
        metrics = mlp.evaluate(X_val_scaled, y_val)

        cv_results['AF_MAE'].append(metrics['AF']['MAE'])
        cv_results['AF_R2'].append(metrics['AF']['R2'])
        cv_results['AF_RMSE'].append(metrics['AF']['RMSE'])
        cv_results['MF_MAE'].append(metrics['MF']['MAE'])
        cv_results['MF_R2'].append(metrics['MF']['R2'])
        cv_results['MF_RMSE'].append(metrics['MF']['RMSE'])

        print(f"AF  - MAE: {metrics['AF']['MAE']:.2f}°C, R²: {metrics['AF']['R2']:.4f}, RMSE: {metrics['AF']['RMSE']:.2f}°C")
        print(f"MF  - MAE: {metrics['MF']['MAE']:.2f}°C, R²: {metrics['MF']['R2']:.4f}, RMSE: {metrics['MF']['RMSE']:.2f}°C")

    # Print summary
    print(f"\n{'='*60}")
    print("Cross-Validation Summary")
    print(f"{'='*60}")
    print(f"\nAF (Austenite Finish):")
    print(f"  MAE:  {np.mean(cv_results['AF_MAE']):.2f} ± {np.std(cv_results['AF_MAE']):.2f}°C")
    print(f"  R²:   {np.mean(cv_results['AF_R2']):.4f} ± {np.std(cv_results['AF_R2']):.4f}")
    print(f"  RMSE: {np.mean(cv_results['AF_RMSE']):.2f} ± {np.std(cv_results['AF_RMSE']):.2f}°C")

    print(f"\nMF (Martensite Finish):")
    print(f"  MAE:  {np.mean(cv_results['MF_MAE']):.2f} ± {np.std(cv_results['MF_MAE']):.2f}°C")
    print(f"  R²:   {np.mean(cv_results['MF_R2']):.4f} ± {np.std(cv_results['MF_R2']):.4f}")
    print(f"  RMSE: {np.mean(cv_results['MF_RMSE']):.2f} ± {np.std(cv_results['MF_RMSE']):.2f}°C")

    return cv_results


def plot_training_history(history, save_path='training_history.png'):
    """Plot training and validation loss/metrics"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Huber Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # MAE
    axes[1].plot(history.history['mae'], label='Train MAE', linewidth=2)
    axes[1].plot(history.history['val_mae'], label='Val MAE', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('MAE (°C)', fontsize=12)
    axes[1].set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    # MSE
    axes[2].plot(history.history['mse'], label='Train MSE', linewidth=2)
    axes[2].plot(history.history['val_mse'], label='Val MSE', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('MSE (°C²)', fontsize=12)
    axes[2].set_title('Mean Squared Error', fontsize=14, fontweight='bold')
    axes[2].legend(fontsize=11)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining history plot saved to: {save_path}")
    plt.close()


def plot_predictions(y_true, y_pred, save_path='predictions.png'):
    """Plot predicted vs actual values for AF and MF"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # AF predictions
    axes[0].scatter(y_true[:, 0], y_pred[:, 0], alpha=0.5, s=30)
    axes[0].plot([y_true[:, 0].min(), y_true[:, 0].max()],
                 [y_true[:, 0].min(), y_true[:, 0].max()],
                 'r--', linewidth=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual AF (°C)', fontsize=12)
    axes[0].set_ylabel('Predicted AF (°C)', fontsize=12)
    axes[0].set_title('AF: Predicted vs Actual', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # MF predictions
    axes[1].scatter(y_true[:, 1], y_pred[:, 1], alpha=0.5, s=30, color='green')
    axes[1].plot([y_true[:, 1].min(), y_true[:, 1].max()],
                 [y_true[:, 1].min(), y_true[:, 1].max()],
                 'r--', linewidth=2, label='Perfect Prediction')
    axes[1].set_xlabel('Actual MF (°C)', fontsize=12)
    axes[1].set_ylabel('Predicted MF (°C)', fontsize=12)
    axes[1].set_title('MF: Predicted vs Actual', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Predictions plot saved to: {save_path}")
    plt.close()


def plot_residuals(y_true, y_pred, save_path='residuals.png'):
    """Plot residuals for AF and MF"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    residuals_af = y_true[:, 0] - y_pred[:, 0]
    residuals_mf = y_true[:, 1] - y_pred[:, 1]

    # AF residuals scatter
    axes[0, 0].scatter(y_pred[:, 0], residuals_af, alpha=0.5, s=30)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Predicted AF (°C)', fontsize=12)
    axes[0, 0].set_ylabel('Residuals (°C)', fontsize=12)
    axes[0, 0].set_title('AF Residuals', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    # AF residuals histogram
    axes[0, 1].hist(residuals_af, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Residuals (°C)', fontsize=12)
    axes[0, 1].set_ylabel('Frequency', fontsize=12)
    axes[0, 1].set_title('AF Residuals Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # MF residuals scatter
    axes[1, 0].scatter(y_pred[:, 1], residuals_mf, alpha=0.5, s=30, color='green')
    axes[1, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Predicted MF (°C)', fontsize=12)
    axes[1, 0].set_ylabel('Residuals (°C)', fontsize=12)
    axes[1, 0].set_title('MF Residuals', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # MF residuals histogram
    axes[1, 1].hist(residuals_mf, bins=30, edgecolor='black', alpha=0.7, color='green')
    axes[1, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Residuals (°C)', fontsize=12)
    axes[1, 1].set_ylabel('Frequency', fontsize=12)
    axes[1, 1].set_title('MF Residuals Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Residuals plot saved to: {save_path}")
    plt.close()


def main():
    """Main training pipeline"""
    print("="*80)
    print("MLP Neural Network for Shape Memory Alloy Temperature Prediction")
    print("="*80)

    # Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    X, y, feature_names = load_and_preprocess_data('dataset/Combined_SMA_Dataset_Filled.csv')

    # Split data: 80% train, 20% test
    print("\n2. Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Further split train into train/val (80% train, 20% val of training set)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Normalize features
    print("\n3. Normalizing features with StandardScaler...")
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)

    # Build model
    print("\n4. Building MLP model...")
    mlp = MLPShapeMemoryAlloy(input_dim=X_train_scaled.shape[1])
    model = mlp.build_model()

    print("\nModel Architecture:")
    model.summary()

    # Train model
    print("\n5. Training model...")
    print(f"Configuration:")
    print(f"  - Epochs: 200 (with early stopping, patience=20)")
    print(f"  - Batch size: 64")
    print(f"  - Learning rate: 0.001")
    print(f"  - Loss: Huber Loss")
    print(f"  - Optimizer: Adam")
    print(f"  - Regularization: Dropout(0.3) + L2(0.001)")

    history = mlp.train(
        X_train_scaled, y_train,
        X_val_scaled, y_val,
        epochs=200,
        batch_size=64,
        patience=20
    )

    # Evaluate on test set
    print("\n6. Evaluating on test set...")
    test_metrics = mlp.evaluate(X_test_scaled, y_test)

    print("\nTest Set Performance:")
    print(f"{'='*60}")
    print(f"AF (Austenite Finish):")
    print(f"  MAE:  {test_metrics['AF']['MAE']:.2f}°C")
    print(f"  R²:   {test_metrics['AF']['R2']:.4f}")
    print(f"  RMSE: {test_metrics['AF']['RMSE']:.2f}°C")
    print(f"\nMF (Martensite Finish):")
    print(f"  MAE:  {test_metrics['MF']['MAE']:.2f}°C")
    print(f"  R²:   {test_metrics['MF']['R2']:.4f}")
    print(f"  RMSE: {test_metrics['MF']['RMSE']:.2f}°C")
    print(f"\nOverall:")
    print(f"  Average MAE: {test_metrics['Overall']['MAE']:.2f}°C")
    print(f"  Average R²:  {test_metrics['Overall']['R2']:.4f}")
    print(f"{'='*60}")

    # Generate visualizations
    print("\n7. Generating visualizations...")
    y_pred_test = mlp.predict(X_test_scaled)

    plot_training_history(history, 'mlp_training_history.png')
    plot_predictions(y_test, y_pred_test, 'mlp_predictions.png')
    plot_residuals(y_test, y_pred_test, 'mlp_residuals.png')

    # Cross-validation
    print("\n8. Performing cross-validation...")
    cv_results = cross_validation_evaluation(X, y, n_splits=5)

    # Save model
    print("\n9. Saving model...")
    model.save('mlp_sma_model.keras')
    print("Model saved to: mlp_sma_model.keras")

    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)

    # Final summary
    print("\nFinal Summary:")
    print(f"  Expected Performance: R² = 0.80-0.88, MAE = 8-15°C")
    print(f"  Achieved Performance: R² = {test_metrics['Overall']['R2']:.4f}, MAE = {test_metrics['Overall']['MAE']:.2f}°C")

    if test_metrics['Overall']['R2'] >= 0.80:
        print("  Status: ✓ Meets expected performance!")
    else:
        print("  Status: ⚠ Below expected performance - consider more training or tuning")


if __name__ == "__main__":
    main()

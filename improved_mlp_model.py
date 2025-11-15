"""
IMPROVED MLP Neural Network for Shape Memory Alloy (SMA) Dataset
Major improvements:
1. Uses ALL available features including AS, MS temperatures
2. Feature engineering with polynomial features
3. Improved architecture with Batch Normalization
4. Better regularization strategy
5. Ensemble predictions with multiple models

Target: Predict AF (Austenite Finish) and MF (Martensite Finish) temperatures
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class ImprovedMLPShapeMemoryAlloy:
    """Improved MLP Model for SMA Temperature Prediction"""

    def __init__(self, input_dim, learning_rate=0.001):
        """
        Initialize improved MLP model

        Args:
            input_dim: Number of input features
            learning_rate: Learning rate for Adam optimizer
        """
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.model = None
        self.scaler_X = RobustScaler()  # More robust to outliers than StandardScaler
        self.history = None

    def build_model(self, use_batch_norm=True):
        """
        Build improved MLP architecture with Batch Normalization:
        Input → 256 (BN, ReLU, Dropout 0.3, L2)
             → 128 (BN, ReLU, Dropout 0.3, L2)
             → 64 (BN, ReLU, Dropout 0.2, L2)
             → 32 (BN, ReLU, Dropout 0.2, L2)
             → 2 (AF, MF outputs)
        """
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=(self.input_dim,)),

            # Hidden Layer 1: 256 neurons
            layers.Dense(
                256,
                kernel_regularizer=regularizers.l2(0.001),
                kernel_initializer='he_normal',
                name='hidden_layer_1'
            ),
            layers.BatchNormalization(name='bn_1') if use_batch_norm else layers.Lambda(lambda x: x),
            layers.Activation('relu'),
            layers.Dropout(0.3, name='dropout_1'),

            # Hidden Layer 2: 128 neurons
            layers.Dense(
                128,
                kernel_regularizer=regularizers.l2(0.001),
                kernel_initializer='he_normal',
                name='hidden_layer_2'
            ),
            layers.BatchNormalization(name='bn_2') if use_batch_norm else layers.Lambda(lambda x: x),
            layers.Activation('relu'),
            layers.Dropout(0.3, name='dropout_2'),

            # Hidden Layer 3: 64 neurons
            layers.Dense(
                64,
                kernel_regularizer=regularizers.l2(0.001),
                kernel_initializer='he_normal',
                name='hidden_layer_3'
            ),
            layers.BatchNormalization(name='bn_3') if use_batch_norm else layers.Lambda(lambda x: x),
            layers.Activation('relu'),
            layers.Dropout(0.2, name='dropout_3'),

            # Hidden Layer 4: 32 neurons
            layers.Dense(
                32,
                kernel_regularizer=regularizers.l2(0.001),
                kernel_initializer='he_normal',
                name='hidden_layer_4'
            ),
            layers.BatchNormalization(name='bn_4') if use_batch_norm else layers.Lambda(lambda x: x),
            layers.Activation('relu'),
            layers.Dropout(0.2, name='dropout_4'),

            # Output layer: 2 neurons (AF, MF)
            layers.Dense(2, kernel_initializer='glorot_normal', name='output_layer')
        ])

        # Compile with Huber loss and Adam optimizer
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=keras.losses.Huber(delta=1.0),
            metrics=['mae', 'mse']
        )

        self.model = model
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=300, batch_size=32, patience=30):
        """
        Train the improved MLP model with enhanced callbacks

        Args:
            X_train: Training features
            y_train: Training targets (AF, MF)
            X_val: Validation features
            y_val: Validation targets
            epochs: Maximum number of epochs
            batch_size: Batch size for training (smaller for better generalization)
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
            patience=15,
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


def create_engineered_features(df):
    """
    Create engineered features to improve model performance

    Args:
        df: DataFrame with raw features

    Returns:
        DataFrame with additional engineered features
    """
    df_eng = df.copy()

    # Element composition features
    element_cols = [
        'Ag (at.%)', 'Al (at.%)', 'Au (at.%)', 'Cd (at.%)', 'Co (at.%)',
        'Cu (at.%)', 'Fe (at.%)', 'Hf (at.%)', 'Mn (at.%)', 'Nb (at.%)',
        'Ni (at.%)', 'Pd (at.%)', 'Pt (at.%)', 'Ru (at.%)', 'Si (at.%)',
        'Ta (at.%)', 'Ti (at.%)', 'Zn (at.%)', 'Zr (at.%)'
    ]

    # Key interaction features (based on common SMA systems)
    # NiTi is the most important SMA system
    if 'Ni (at.%)' in df_eng.columns and 'Ti (at.%)' in df_eng.columns:
        df_eng['NiTi_ratio'] = df_eng['Ni (at.%)'] / (df_eng['Ti (at.%)'] + 1e-6)
        df_eng['NiTi_product'] = df_eng['Ni (at.%)'] * df_eng['Ti (at.%)']
        df_eng['NiTi_sum'] = df_eng['Ni (at.%)'] + df_eng['Ti (at.%)']

    # Cu-based SMAs
    if 'Cu (at.%)' in df_eng.columns and 'Al (at.%)' in df_eng.columns:
        df_eng['CuAl_product'] = df_eng['Cu (at.%)'] * df_eng['Al (at.%)']

    # Process parameter interactions
    if 'Cooling Rate (°C/min)' in df_eng.columns and 'Heating Rate (°C/min)' in df_eng.columns:
        df_eng['Rate_ratio'] = df_eng['Cooling Rate (°C/min)'] / (df_eng['Heating Rate (°C/min)'] + 1e-6)
        df_eng['Rate_avg'] = (df_eng['Cooling Rate (°C/min)'] + df_eng['Heating Rate (°C/min)']) / 2

    # Number of elements present (compositional complexity)
    element_present = (df_eng[element_cols] > 0).sum(axis=1)
    df_eng['Num_elements'] = element_present

    # Dominant element percentage
    df_eng['Max_element_pct'] = df_eng[element_cols].max(axis=1)

    return df_eng


def load_and_preprocess_data_improved(file_path):
    """
    Load and preprocess the SMA dataset with ALL features and feature engineering

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

    # Define ALL available features
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

    # IMPORTANT: Include AS and MS as features (they are highly correlated with AF and MF)
    additional_features = [
        'Austenite Start Temperature - AS - (°C)',
        'Martensite Start Temperature - MS - (°C)',
    ]

    # Target columns
    target_cols = [
        'Austenite Finish Temperature - AF - (°C)',
        'Martensite Finish Temperature - MF - (°C)'
    ]

    # Create engineered features
    print("\nCreating engineered features...")
    df_eng = create_engineered_features(df)

    # Get all engineered feature names
    engineered_cols = [col for col in df_eng.columns if col not in df.columns]
    print(f"Created {len(engineered_cols)} engineered features: {engineered_cols}")

    # Combine all features
    feature_cols = element_features + process_features + additional_features + engineered_cols

    # Check if all columns exist
    available_features = [col for col in feature_cols if col in df_eng.columns]
    missing_cols = [col for col in feature_cols if col not in df_eng.columns]

    if missing_cols:
        print(f"\nWarning: Missing columns: {missing_cols}")

    print(f"\nUsing {len(available_features)} features total")

    # Extract features and targets
    X = df_eng[available_features].values
    y = df_eng[target_cols].values

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

    return X, y, available_features


def train_ensemble(X_train, y_train, X_val, y_val, n_models=3):
    """
    Train an ensemble of models for more robust predictions

    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        n_models: Number of models in ensemble

    Returns:
        List of trained models
    """
    models = []

    print(f"\n{'='*60}")
    print(f"Training Ensemble of {n_models} Models")
    print(f"{'='*60}")

    for i in range(n_models):
        print(f"\nTraining Model {i+1}/{n_models}")
        print("-" * 40)

        # Use different random seeds for each model
        tf.random.set_seed(42 + i)
        np.random.seed(42 + i)

        # Build and train model
        mlp = ImprovedMLPShapeMemoryAlloy(input_dim=X_train.shape[1], learning_rate=0.001)
        mlp.build_model(use_batch_norm=True)

        # Train with reduced verbosity for ensemble
        mlp.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=300,
            batch_size=32,
            callbacks=[
                callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=0),
                callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-7, verbose=0)
            ],
            verbose=0
        )

        models.append(mlp)

        # Evaluate individual model
        metrics = mlp.evaluate(X_val, y_val)
        print(f"Model {i+1} - AF R²: {metrics['AF']['R2']:.4f}, MF R²: {metrics['MF']['R2']:.4f}")

    return models


def ensemble_predict(models, X):
    """
    Make predictions using ensemble of models

    Args:
        models: List of trained models
        X: Input features

    Returns:
        Average predictions from all models
    """
    predictions = np.array([model.predict(X) for model in models])
    return np.mean(predictions, axis=0)


def evaluate_ensemble(models, X, y):
    """Evaluate ensemble model performance"""
    y_pred = ensemble_predict(models, X)

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


def plot_training_history(history, save_path='improved_mlp_training_history.png'):
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


def plot_predictions(y_true, y_pred, save_path='improved_mlp_predictions.png'):
    """Plot predicted vs actual values for AF and MF"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # AF predictions
    axes[0].scatter(y_true[:, 0], y_pred[:, 0], alpha=0.6, s=30)
    axes[0].plot([y_true[:, 0].min(), y_true[:, 0].max()],
                 [y_true[:, 0].min(), y_true[:, 0].max()],
                 'r--', linewidth=2, label='Perfect Prediction')

    # Add R² to plot
    r2_af = r2_score(y_true[:, 0], y_pred[:, 0])
    mae_af = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
    axes[0].text(0.05, 0.95, f'R² = {r2_af:.4f}\nMAE = {mae_af:.2f}°C',
                 transform=axes[0].transAxes, fontsize=12, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    axes[0].set_xlabel('Actual AF (°C)', fontsize=12)
    axes[0].set_ylabel('Predicted AF (°C)', fontsize=12)
    axes[0].set_title('AF: Predicted vs Actual', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # MF predictions
    axes[1].scatter(y_true[:, 1], y_pred[:, 1], alpha=0.6, s=30, color='green')
    axes[1].plot([y_true[:, 1].min(), y_true[:, 1].max()],
                 [y_true[:, 1].min(), y_true[:, 1].max()],
                 'r--', linewidth=2, label='Perfect Prediction')

    # Add R² to plot
    r2_mf = r2_score(y_true[:, 1], y_pred[:, 1])
    mae_mf = mean_absolute_error(y_true[:, 1], y_pred[:, 1])
    axes[1].text(0.05, 0.95, f'R² = {r2_mf:.4f}\nMAE = {mae_mf:.2f}°C',
                 transform=axes[1].transAxes, fontsize=12, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    axes[1].set_xlabel('Actual MF (°C)', fontsize=12)
    axes[1].set_ylabel('Predicted MF (°C)', fontsize=12)
    axes[1].set_title('MF: Predicted vs Actual', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Predictions plot saved to: {save_path}")
    plt.close()


def plot_residuals(y_true, y_pred, save_path='improved_mlp_residuals.png'):
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
    axes[0, 1].set_title(f'AF Residuals Distribution (μ={residuals_af.mean():.2f}°C)', fontsize=14, fontweight='bold')
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
    axes[1, 1].set_title(f'MF Residuals Distribution (μ={residuals_mf.mean():.2f}°C)', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Residuals plot saved to: {save_path}")
    plt.close()


def main():
    """Main training pipeline with improvements"""
    print("="*80)
    print("IMPROVED MLP Neural Network for Shape Memory Alloy Temperature Prediction")
    print("="*80)
    print("\nKey Improvements:")
    print("  ✓ Using ALL available features (AS, MS temperatures)")
    print("  ✓ Advanced feature engineering (interactions, ratios)")
    print("  ✓ Deeper architecture with Batch Normalization")
    print("  ✓ RobustScaler for better handling of outliers")
    print("  ✓ Ensemble of 3 models for robust predictions")
    print("="*80)

    # Load and preprocess data with improvements
    print("\n1. Loading and preprocessing data with feature engineering...")
    X, y, feature_names = load_and_preprocess_data_improved('dataset/Combined_SMA_Dataset_Filled.csv')

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

    # Normalize features with RobustScaler (better for outliers)
    print("\n3. Normalizing features with RobustScaler...")
    scaler_X = RobustScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)

    # Build and train single model first
    print("\n4. Building improved MLP model...")
    mlp = ImprovedMLPShapeMemoryAlloy(input_dim=X_train_scaled.shape[1])
    model = mlp.build_model(use_batch_norm=True)

    print("\nModel Architecture:")
    model.summary()

    # Train model
    print("\n5. Training improved model...")
    print(f"Configuration:")
    print(f"  - Architecture: 256 → 128 → 64 → 32 → 2")
    print(f"  - Batch Normalization: Enabled")
    print(f"  - Epochs: 300 (with early stopping, patience=30)")
    print(f"  - Batch size: 32")
    print(f"  - Learning rate: 0.001")
    print(f"  - Loss: Huber Loss")
    print(f"  - Optimizer: Adam")
    print(f"  - Regularization: Dropout(0.3/0.2) + L2(0.001) + BatchNorm")

    history = mlp.train(
        X_train_scaled, y_train,
        X_val_scaled, y_val,
        epochs=300,
        batch_size=32,
        patience=30
    )

    # Evaluate single model
    print("\n6. Evaluating single model on test set...")
    test_metrics = mlp.evaluate(X_test_scaled, y_test)

    print("\nSingle Model Test Performance:")
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

    # Train ensemble
    print("\n7. Training ensemble of models...")
    ensemble_models = train_ensemble(X_train_scaled, y_train, X_val_scaled, y_val, n_models=3)

    # Evaluate ensemble
    print("\n8. Evaluating ensemble on test set...")
    ensemble_metrics = evaluate_ensemble(ensemble_models, X_test_scaled, y_test)

    print("\nEnsemble Model Test Performance:")
    print(f"{'='*60}")
    print(f"AF (Austenite Finish):")
    print(f"  MAE:  {ensemble_metrics['AF']['MAE']:.2f}°C")
    print(f"  R²:   {ensemble_metrics['AF']['R2']:.4f}")
    print(f"  RMSE: {ensemble_metrics['AF']['RMSE']:.2f}°C")
    print(f"\nMF (Martensite Finish):")
    print(f"  MAE:  {ensemble_metrics['MF']['MAE']:.2f}°C")
    print(f"  R²:   {ensemble_metrics['MF']['R2']:.4f}")
    print(f"  RMSE: {ensemble_metrics['MF']['RMSE']:.2f}°C")
    print(f"\nOverall:")
    print(f"  Average MAE: {ensemble_metrics['Overall']['MAE']:.2f}°C")
    print(f"  Average R²:  {ensemble_metrics['Overall']['R2']:.4f}")
    print(f"{'='*60}")

    # Generate visualizations (using ensemble predictions)
    print("\n9. Generating visualizations...")
    y_pred_test = ensemble_predict(ensemble_models, X_test_scaled)

    plot_training_history(history, 'improved_mlp_training_history.png')
    plot_predictions(y_test, y_pred_test, 'improved_mlp_predictions.png')
    plot_residuals(y_test, y_pred_test, 'improved_mlp_residuals.png')

    # Save best single model
    print("\n10. Saving best model...")
    model.save('improved_mlp_sma_model.keras')
    print("Model saved to: improved_mlp_sma_model.keras")

    # Save ensemble models
    for i, ens_model in enumerate(ensemble_models):
        ens_model.model.save(f'improved_mlp_sma_ensemble_{i+1}.keras')
    print(f"Ensemble models saved (3 models)")

    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)

    # Final comparison
    print("\nPerformance Comparison:")
    print(f"  Original Expected: R² = 0.80-0.88, MAE = 8-15°C")
    print(f"  Single Model:      R² = {test_metrics['Overall']['R2']:.4f}, MAE = {test_metrics['Overall']['MAE']:.2f}°C")
    print(f"  Ensemble Model:    R² = {ensemble_metrics['Overall']['R2']:.4f}, MAE = {ensemble_metrics['Overall']['MAE']:.2f}°C")

    improvement = (ensemble_metrics['Overall']['R2'] - test_metrics['Overall']['R2']) * 100
    print(f"\nEnsemble Improvement: +{improvement:.2f}% R² score")

    if ensemble_metrics['Overall']['R2'] >= 0.80:
        print("  Status: ✓ Meets expected performance!")
    elif ensemble_metrics['Overall']['R2'] >= 0.70:
        print("  Status: ⚠ Close to expected performance")
    else:
        print("  Status: ⚠ Below expected performance")

    print("\nNote: If performance is still below expectations, consider:")
    print("  - Removing outliers from the dataset")
    print("  - Trying different loss functions (MSE, MAE)")
    print("  - Hyperparameter tuning with larger search space")
    print("  - Using gradient boosting (XGBoost) instead of neural networks")


if __name__ == "__main__":
    main()

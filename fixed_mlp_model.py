"""
FIXED MLP Model for SMA Temperature Prediction

Key fixes:
1. Use StandardScaler (proven to work)
2. Add AS/MS features (high correlation 0.84+)
3. Keep moderate architecture (not over-complex)
4. Use MSE loss (simpler than Huber)
5. Train separate models for AF and MF for better accuracy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)


def load_and_preprocess_data(file_path):
    """Load SMA dataset with ALL available features"""
    df = pd.read_csv(file_path)

    print(f"Dataset shape: {df.shape}")

    # Element compositions
    element_features = [
        'Ag (at.%)', 'Al (at.%)', 'Au (at.%)', 'Cd (at.%)', 'Co (at.%)',
        'Cu (at.%)', 'Fe (at.%)', 'Hf (at.%)', 'Mn (at.%)', 'Nb (at.%)',
        'Ni (at.%)', 'Pd (at.%)', 'Pt (at.%)', 'Ru (at.%)', 'Si (at.%)',
        'Ta (at.%)', 'Ti (at.%)', 'Zn (at.%)', 'Zr (at.%)'
    ]

    # Process parameters
    process_features = [
        'Cooling Rate (°C/min)',
        'Heating Rate (°C/min)',
        'Calculated Density (g/cm^3)'
    ]

    # Additional temperature features (high correlation!)
    temp_features = [
        'Austenite Start Temperature - AS - (°C)',
        'Martensite Start Temperature - MS - (°C)',
    ]

    # Targets
    target_af = 'Austenite Finish Temperature - AF - (°C)'
    target_mf = 'Martensite Finish Temperature - MF - (°C)'

    # Combine features
    all_features = element_features + process_features + temp_features

    # Extract data
    X = df[all_features].values
    y_af = df[target_af].values
    y_mf = df[target_mf].values

    # Remove NaN (should be none based on diagnostics)
    valid_idx = ~(np.isnan(X).any(axis=1) | np.isnan(y_af) | np.isnan(y_mf))
    X = X[valid_idx]
    y_af = y_af[valid_idx]
    y_mf = y_mf[valid_idx]

    print(f"\nSamples: {X.shape[0]}")
    print(f"Features: {X.shape[1]} ({len(element_features)} elements + {len(process_features)} process + {len(temp_features)} temps)")
    print(f"\nTargets:")
    print(f"  AF: mean={y_af.mean():.1f}°C, std={y_af.std():.1f}°C")
    print(f"  MF: mean={y_mf.mean():.1f}°C, std={y_mf.std():.1f}°C")

    return X, y_af, y_mf, all_features


def create_model(input_dim, output_name='AF'):
    """
    Create a simple, proven architecture
    Similar to baseline but slightly improved
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),

        # Layer 1: 128 neurons
        layers.Dense(
            128,
            activation='relu',
            kernel_regularizer=regularizers.l2(0.001),
            kernel_initializer='he_normal'
        ),
        layers.Dropout(0.3),

        # Layer 2: 64 neurons
        layers.Dense(
            64,
            activation='relu',
            kernel_regularizer=regularizers.l2(0.001),
            kernel_initializer='he_normal'
        ),
        layers.Dropout(0.3),

        # Layer 3: 32 neurons
        layers.Dense(
            32,
            activation='relu',
            kernel_regularizer=regularizers.l2(0.001),
            kernel_initializer='he_normal'
        ),
        layers.Dropout(0.2),

        # Output: single value
        layers.Dense(1, name=f'output_{output_name}')
    ], name=f'MLP_{output_name}')

    # Use MSE loss (simpler than Huber, proven to work)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    return model


def train_model(model, X_train, y_train, X_val, y_val, model_name='AF'):
    """Train model with early stopping"""
    print(f"\nTraining {model_name} model...")

    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=25,
            restore_best_weights=True,
            verbose=0
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=0
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=64,
        callbacks=callbacks_list,
        verbose=0
    )

    print(f"  Training completed: {len(history.history['loss'])} epochs")
    print(f"  Best val_loss: {min(history.history['val_loss']):.2f}")

    return history


def evaluate_model(model, X, y_true, model_name='AF'):
    """Evaluate single-output model"""
    y_pred = model.predict(X, verbose=0).flatten()

    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return {'MAE': mae, 'R2': r2, 'RMSE': rmse}, y_pred


def plot_results(y_true_af, y_pred_af, y_true_mf, y_pred_mf,
                 metrics_af, metrics_mf, save_path='fixed_mlp_results.png'):
    """Plot prediction results"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # AF plot
    axes[0].scatter(y_true_af, y_pred_af, alpha=0.5, s=30)
    axes[0].plot([y_true_af.min(), y_true_af.max()],
                 [y_true_af.min(), y_true_af.max()],
                 'r--', linewidth=2, label='Perfect Prediction')
    axes[0].text(0.05, 0.95,
                 f"R² = {metrics_af['R2']:.4f}\nMAE = {metrics_af['MAE']:.2f}°C\nRMSE = {metrics_af['RMSE']:.2f}°C",
                 transform=axes[0].transAxes, fontsize=11,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    axes[0].set_xlabel('Actual AF (°C)', fontsize=12)
    axes[0].set_ylabel('Predicted AF (°C)', fontsize=12)
    axes[0].set_title('AF: Predicted vs Actual', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # MF plot
    axes[1].scatter(y_true_mf, y_pred_mf, alpha=0.5, s=30, color='green')
    axes[1].plot([y_true_mf.min(), y_true_mf.max()],
                 [y_true_mf.min(), y_true_mf.max()],
                 'r--', linewidth=2, label='Perfect Prediction')
    axes[1].text(0.05, 0.95,
                 f"R² = {metrics_mf['R2']:.4f}\nMAE = {metrics_mf['MAE']:.2f}°C\nRMSE = {metrics_mf['RMSE']:.2f}°C",
                 transform=axes[1].transAxes, fontsize=11,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    axes[1].set_xlabel('Actual MF (°C)', fontsize=12)
    axes[1].set_ylabel('Predicted MF (°C)', fontsize=12)
    axes[1].set_title('MF: Predicted vs Actual', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nResults plot saved: {save_path}")
    plt.close()


def main():
    print("="*80)
    print("FIXED MLP Model for SMA Temperature Prediction")
    print("="*80)
    print("\nKey Fixes:")
    print("  ✓ StandardScaler (proven to work)")
    print("  ✓ AS/MS features included (0.84+ correlation)")
    print("  ✓ Moderate architecture: 128-64-32")
    print("  ✓ Separate models for AF and MF")
    print("  ✓ MSE loss (simpler, proven)")
    print("="*80)

    # Load data
    print("\n1. Loading data...")
    X, y_af, y_mf, feature_names = load_and_preprocess_data('dataset/Combined_SMA_Dataset_Filled.csv')

    # Split data
    print("\n2. Splitting data (80% train, 20% test)...")
    X_train, X_test, y_af_train, y_af_test, y_mf_train, y_mf_test = train_test_split(
        X, y_af, y_mf, test_size=0.2, random_state=42
    )

    # Further split for validation
    X_train, X_val, y_af_train, y_af_val, y_mf_train, y_mf_val = train_test_split(
        X_train, y_af_train, y_mf_train, test_size=0.2, random_state=42
    )

    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Val:   {X_val.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples")

    # Scale features (StandardScaler - proven to work!)
    print("\n3. Scaling features with StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Train AF model
    print("\n4. Training separate models for AF and MF...")
    model_af = create_model(X_train_scaled.shape[1], 'AF')
    history_af = train_model(model_af, X_train_scaled, y_af_train,
                              X_val_scaled, y_af_val, 'AF')

    # Train MF model
    model_mf = create_model(X_train_scaled.shape[1], 'MF')
    history_mf = train_model(model_mf, X_train_scaled, y_mf_train,
                              X_val_scaled, y_mf_val, 'MF')

    # Evaluate on test set
    print("\n5. Evaluating on test set...")
    metrics_af, y_pred_af = evaluate_model(model_af, X_test_scaled, y_af_test, 'AF')
    metrics_mf, y_pred_mf = evaluate_model(model_mf, X_test_scaled, y_mf_test, 'MF')

    # Print results
    print("\n" + "="*80)
    print("TEST SET PERFORMANCE")
    print("="*80)
    print(f"\nAF (Austenite Finish):")
    print(f"  MAE:  {metrics_af['MAE']:.2f}°C")
    print(f"  R²:   {metrics_af['R2']:.4f}")
    print(f"  RMSE: {metrics_af['RMSE']:.2f}°C")

    print(f"\nMF (Martensite Finish):")
    print(f"  MAE:  {metrics_mf['MAE']:.2f}°C")
    print(f"  R²:   {metrics_mf['R2']:.4f}")
    print(f"  RMSE: {metrics_mf['RMSE']:.2f}°C")

    avg_r2 = (metrics_af['R2'] + metrics_mf['R2']) / 2
    avg_mae = (metrics_af['MAE'] + metrics_mf['MAE']) / 2

    print(f"\nOverall Average:")
    print(f"  MAE:  {avg_mae:.2f}°C")
    print(f"  R²:   {avg_r2:.4f}")
    print("="*80)

    # Plot results
    print("\n6. Generating visualizations...")
    plot_results(y_af_test, y_pred_af, y_mf_test, y_pred_mf,
                 metrics_af, metrics_mf)

    # Save models
    print("\n7. Saving models...")
    model_af.save('fixed_mlp_af_model.keras')
    model_mf.save('fixed_mlp_mf_model.keras')
    print("  Models saved: fixed_mlp_af_model.keras, fixed_mlp_mf_model.keras")

    # Final summary
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    print(f"  Baseline (no AS/MS):      R² = 0.5696, MAE = 52.90°C")
    print(f"  'Improved' (broken):      R² = -0.0329, MAE = 77.10°C  ❌")
    print(f"  Fixed (with AS/MS):       R² = {avg_r2:.4f}, MAE = {avg_mae:.2f}°C")
    print(f"  Target:                   R² = 0.80-0.88, MAE = 8-15°C")

    if avg_r2 > 0.80:
        print(f"\n  Status: ✅ MEETS TARGET!")
    elif avg_r2 > 0.70:
        print(f"\n  Status: ⚠ Close to target")
    elif avg_r2 > 0.57:
        print(f"\n  Status: ✓ Better than baseline (was R²=0.57)")
    else:
        print(f"\n  Status: ❌ Needs more work")

    improvement = ((avg_r2 - 0.5696) / 0.5696) * 100
    if improvement > 0:
        print(f"  Improvement over baseline: +{improvement:.1f}%")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()

"""
Performance Boosting Strategies for SMA Temperature Prediction

Current: R² = 0.7560, MAE = 28.49°C
Target:  R² = 0.80-0.88, MAE = 8-15°C

Strategies to try:
1. Ensemble of multiple models
2. Hyperparameter tuning
3. Outlier removal
4. Separate models by alloy family
5. Advanced architectures
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)


def load_data():
    """Load SMA dataset"""
    df = pd.read_csv('dataset/Combined_SMA_Dataset_Filled.csv')

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

    temp_features = [
        'Austenite Start Temperature - AS - (°C)',
        'Martensite Start Temperature - MS - (°C)',
    ]

    all_features = element_features + process_features + temp_features

    target_af = 'Austenite Finish Temperature - AF - (°C)'
    target_mf = 'Martensite Finish Temperature - MF - (°C)'

    X = df[all_features].values
    y_af = df[target_af].values
    y_mf = df[target_mf].values

    valid_idx = ~(np.isnan(X).any(axis=1) | np.isnan(y_af) | np.isnan(y_mf))
    X = X[valid_idx]
    y_af = y_af[valid_idx]
    y_mf = y_mf[valid_idx]

    return X, y_af, y_mf, df[valid_idx], all_features


# ============================================================================
# Strategy 1: Outlier Removal
# ============================================================================

def remove_outliers(X, y_af, y_mf, df, n_std=3):
    """Remove extreme outliers based on temperature ranges"""
    print(f"\n{'='*80}")
    print("Strategy 1: Outlier Removal")
    print(f"{'='*80}")

    # Calculate z-scores for targets
    af_mean, af_std = y_af.mean(), y_af.std()
    mf_mean, mf_std = y_mf.mean(), y_mf.std()

    af_z = np.abs((y_af - af_mean) / af_std)
    mf_z = np.abs((y_mf - mf_mean) / mf_std)

    # Keep samples within n_std standard deviations
    mask = (af_z < n_std) & (mf_z < n_std)

    removed = len(y_af) - mask.sum()
    print(f"Samples before: {len(y_af)}")
    print(f"Outliers removed (>{n_std}σ): {removed} ({removed/len(y_af)*100:.1f}%)")
    print(f"Samples after: {mask.sum()}")

    print(f"\nAF range before: [{y_af.min():.1f}, {y_af.max():.1f}]°C")
    print(f"AF range after:  [{y_af[mask].min():.1f}, {y_af[mask].max():.1f}]°C")
    print(f"MF range before: [{y_mf.min():.1f}, {y_mf.max():.1f}]°C")
    print(f"MF range after:  [{y_mf[mask].min():.1f}, {y_mf[mask].max():.1f}]°C")

    return X[mask], y_af[mask], y_mf[mask], df[mask]


# ============================================================================
# Strategy 2: Improved Architecture with Residual Connections
# ============================================================================

def create_residual_model(input_dim, output_name='AF'):
    """Create model with residual connections for better gradient flow"""
    inputs = layers.Input(shape=(input_dim,))

    # First block
    x = layers.Dense(128, kernel_regularizer=regularizers.l2(0.001), kernel_initializer='he_normal')(inputs)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)

    # Second block with residual
    residual = x
    x = layers.Dense(128, kernel_regularizer=regularizers.l2(0.001), kernel_initializer='he_normal')(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Add()([x, residual])  # Residual connection

    # Third block
    x = layers.Dense(64, kernel_regularizer=regularizers.l2(0.001), kernel_initializer='he_normal')(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)

    # Output
    outputs = layers.Dense(1, name=f'output_{output_name}')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=f'Residual_{output_name}')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    return model


# ============================================================================
# Strategy 3: Ensemble of Multiple Models
# ============================================================================

def train_ensemble(X_train, y_train, X_val, y_val, model_name='AF', n_models=5, use_residual=False):
    """Train ensemble of models with different initializations"""
    print(f"\nTraining {n_models}-model ensemble for {model_name}...")

    models = []
    for i in range(n_models):
        # Set different seed for each model
        tf.random.set_seed(42 + i)

        if use_residual:
            model = create_residual_model(X_train.shape[1], model_name)
        else:
            # Standard architecture
            model = keras.Sequential([
                layers.Input(shape=(X_train.shape[1],)),
                layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001), kernel_initializer='he_normal'),
                layers.Dropout(0.3),
                layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001), kernel_initializer='he_normal'),
                layers.Dropout(0.3),
                layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001), kernel_initializer='he_normal'),
                layers.Dropout(0.2),
                layers.Dense(1)
            ])
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

        # Train
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=200,
            batch_size=64,
            callbacks=[
                callbacks.EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=0),
                callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7, verbose=0)
            ],
            verbose=0
        )

        models.append(model)
        print(f"  Model {i+1}/{n_models} trained")

    return models


def ensemble_predict(models, X):
    """Average predictions from ensemble"""
    predictions = np.array([model.predict(X, verbose=0).flatten() for model in models])
    return np.mean(predictions, axis=0)


# ============================================================================
# Strategy 4: Alloy-Family-Specific Models
# ============================================================================

def identify_alloy_families(df):
    """Identify major alloy families in dataset"""
    # NiTi family
    niti_mask = (df['Ni (at.%)'] > 40) & (df['Ti (at.%)'] > 40)

    # Cu-based family
    cu_mask = (df['Cu (at.%)'] > 20) & ~niti_mask

    # Others
    other_mask = ~(niti_mask | cu_mask)

    print(f"\n{'='*80}")
    print("Strategy 4: Alloy Family Analysis")
    print(f"{'='*80}")
    print(f"NiTi alloys: {niti_mask.sum()} ({niti_mask.sum()/len(df)*100:.1f}%)")
    print(f"Cu-based alloys: {cu_mask.sum()} ({cu_mask.sum()/len(df)*100:.1f}%)")
    print(f"Other alloys: {other_mask.sum()} ({other_mask.sum()/len(df)*100:.1f}%)")

    return niti_mask, cu_mask, other_mask


# ============================================================================
# Main Boosting Pipeline
# ============================================================================

def boost_performance():
    print("="*80)
    print("PERFORMANCE BOOSTING STRATEGIES")
    print("="*80)
    print(f"Current:  R² = 0.7560, MAE = 28.49°C")
    print(f"Target:   R² = 0.80-0.88, MAE = 8-15°C")
    print(f"Gap:      Need +5-10% R², reduce MAE by ~50%")
    print("="*80)

    # Load data
    print("\nLoading data...")
    X, y_af, y_mf, df, feature_names = load_data()
    print(f"Dataset: {len(X)} samples, {X.shape[1]} features")

    # Strategy 1: Remove outliers
    X_clean, y_af_clean, y_mf_clean, df_clean = remove_outliers(X, y_af, y_mf, df, n_std=3)

    # Split data
    print(f"\n{'='*80}")
    print("Splitting data...")
    print(f"{'='*80}")
    X_train, X_test, y_af_train, y_af_test, y_mf_train, y_mf_test = train_test_split(
        X_clean, y_af_clean, y_mf_clean, test_size=0.2, random_state=42
    )
    X_train, X_val, y_af_train, y_af_val, y_mf_train, y_mf_val = train_test_split(
        X_train, y_af_train, y_mf_train, test_size=0.2, random_state=42
    )

    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Strategy 2 & 3: Train ensemble with residual connections
    print(f"\n{'='*80}")
    print("Strategy 2 & 3: Residual Architecture + Ensemble")
    print(f"{'='*80}")

    ensemble_af = train_ensemble(X_train_scaled, y_af_train, X_val_scaled, y_af_val,
                                  'AF', n_models=5, use_residual=True)
    ensemble_mf = train_ensemble(X_train_scaled, y_mf_train, X_val_scaled, y_mf_val,
                                  'MF', n_models=5, use_residual=True)

    # Evaluate
    print(f"\n{'='*80}")
    print("RESULTS - Boosted Model")
    print(f"{'='*80}")

    y_pred_af = ensemble_predict(ensemble_af, X_test_scaled)
    y_pred_mf = ensemble_predict(ensemble_mf, X_test_scaled)

    mae_af = mean_absolute_error(y_af_test, y_pred_af)
    r2_af = r2_score(y_af_test, y_pred_af)
    rmse_af = np.sqrt(mean_squared_error(y_af_test, y_pred_af))

    mae_mf = mean_absolute_error(y_mf_test, y_pred_mf)
    r2_mf = r2_score(y_mf_test, y_pred_mf)
    rmse_mf = np.sqrt(mean_squared_error(y_mf_test, y_pred_mf))

    avg_r2 = (r2_af + r2_mf) / 2
    avg_mae = (mae_af + mae_mf) / 2

    print(f"\nAF (Austenite Finish):")
    print(f"  MAE:  {mae_af:.2f}°C")
    print(f"  R²:   {r2_af:.4f}")
    print(f"  RMSE: {rmse_af:.2f}°C")

    print(f"\nMF (Martensite Finish):")
    print(f"  MAE:  {mae_mf:.2f}°C")
    print(f"  R²:   {r2_mf:.4f}")
    print(f"  RMSE: {rmse_mf:.2f}°C")

    print(f"\nOverall Average:")
    print(f"  MAE:  {avg_mae:.2f}°C")
    print(f"  R²:   {avg_r2:.4f}")

    print(f"\n{'='*80}")
    print("COMPARISON")
    print(f"{'='*80}")
    print(f"  Baseline (no AS/MS):       R² = 0.5696, MAE = 52.90°C")
    print(f"  Fixed (with AS/MS):        R² = 0.7560, MAE = 28.49°C")
    print(f"  Boosted (outliers+ensemble): R² = {avg_r2:.4f}, MAE = {avg_mae:.2f}°C")
    print(f"  Target:                    R² = 0.80-0.88, MAE = 8-15°C")

    improvement_vs_fixed = ((avg_r2 - 0.7560) / 0.7560) * 100
    improvement_vs_baseline = ((avg_r2 - 0.5696) / 0.5696) * 100

    print(f"\n  Improvement vs Fixed:    {improvement_vs_fixed:+.1f}%")
    print(f"  Improvement vs Baseline: {improvement_vs_baseline:+.1f}%")

    if avg_r2 >= 0.80:
        print(f"\n  Status: ✅ MEETS OR EXCEEDS TARGET!")
    elif avg_r2 >= 0.78:
        print(f"\n  Status: ⚠ Very close to target")
    elif avg_r2 > 0.7560:
        print(f"\n  Status: ✓ Better than fixed model")
    else:
        print(f"\n  Status: ⚠ Same or worse than fixed model")

    # Plot results
    print(f"\n{'='*80}")
    print("Generating visualizations...")
    print(f"{'='*80}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # AF
    axes[0].scatter(y_af_test, y_pred_af, alpha=0.5, s=30)
    axes[0].plot([y_af_test.min(), y_af_test.max()],
                 [y_af_test.min(), y_af_test.max()],
                 'r--', linewidth=2, label='Perfect Prediction')
    axes[0].text(0.05, 0.95,
                 f"R² = {r2_af:.4f}\nMAE = {mae_af:.2f}°C",
                 transform=axes[0].transAxes, fontsize=12,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    axes[0].set_xlabel('Actual AF (°C)', fontsize=12)
    axes[0].set_ylabel('Predicted AF (°C)', fontsize=12)
    axes[0].set_title('Boosted Model - AF Predictions', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # MF
    axes[1].scatter(y_mf_test, y_pred_mf, alpha=0.5, s=30, color='green')
    axes[1].plot([y_mf_test.min(), y_mf_test.max()],
                 [y_mf_test.min(), y_mf_test.max()],
                 'r--', linewidth=2, label='Perfect Prediction')
    axes[1].text(0.05, 0.95,
                 f"R² = {r2_mf:.4f}\nMAE = {mae_mf:.2f}°C",
                 transform=axes[1].transAxes, fontsize=12,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    axes[1].set_xlabel('Actual MF (°C)', fontsize=12)
    axes[1].set_ylabel('Predicted MF (°C)', fontsize=12)
    axes[1].set_title('Boosted Model - MF Predictions', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('boosted_model_results.png', dpi=300, bbox_inches='tight')
    print("Results saved: boosted_model_results.png")
    plt.close()

    # Save models
    print("\nSaving ensemble models...")
    for i, model in enumerate(ensemble_af):
        model.save(f'boosted_af_model_{i+1}.keras')
    for i, model in enumerate(ensemble_mf):
        model.save(f'boosted_mf_model_{i+1}.keras')
    print(f"Saved 10 models (5 AF + 5 MF)")

    print("\n" + "="*80)
    print("BOOSTING COMPLETE!")
    print("="*80)

    # Additional analysis
    identify_alloy_families(df_clean)


if __name__ == "__main__":
    boost_performance()

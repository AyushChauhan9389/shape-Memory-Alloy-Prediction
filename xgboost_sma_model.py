"""
XGBoost Model for Shape Memory Alloy (SMA) Temperature Prediction
Predicts Austenite Finish (AF) and Martensite Finish (MF) temperatures
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class SMAXGBoostModel:
    """
    XGBoost-based predictor for Shape Memory Alloy transformation temperatures
    """

    def __init__(self):
        self.model_af = None
        self.model_mf = None
        self.scaler = StandardScaler()
        self.feature_columns = None

    def load_data(self, filepath):
        """Load and prepare the SMA dataset"""
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"Dataset shape: {df.shape}")
        print(f"\nColumns: {df.columns.tolist()}")
        return df

    def prepare_features(self, df):
        """
        Prepare features for model training
        Selects relevant features and creates engineered features
        """
        print("\nPreparing features...")

        # Define element composition columns
        element_cols = [
            'Ag (at.%)', 'Al (at.%)', 'Au (at.%)', 'Cd (at.%)', 'Co (at.%)',
            'Cu (at.%)', 'Fe (at.%)', 'Hf (at.%)', 'Mn (at.%)', 'Nb (at.%)',
            'Ni (at.%)', 'Pd (at.%)', 'Pt (at.%)', 'Ru (at.%)', 'Si (at.%)',
            'Ta (at.%)', 'Ti (at.%)', 'Zn (at.%)', 'Zr (at.%)'
        ]

        # Additional features
        additional_features = [
            'Austenite Start Temperature - AS - (°C)',
            'Martensite Start Temperature - MS - (°C)',
            'Cooling Rate (°C/min)',
            'Heating Rate (°C/min)',
            'Calculated Density (g/cm^3)'
        ]

        # Note: Not using TSPAN as it's derived from AF-MF (target leakage)

        # Combine all features
        self.feature_columns = element_cols + additional_features

        # Create feature matrix
        X = df[self.feature_columns].copy()

        # Feature Engineering: Create element ratios and interactions
        # Ni/Ti ratio (important for NiTi alloys)
        X['Ni_Ti_ratio'] = X['Ni (at.%)'] / (X['Ti (at.%)'] + 1e-6)

        # Total alloying elements (excluding Ni and Ti)
        alloying_elements = [col for col in element_cols if col not in ['Ni (at.%)', 'Ti (at.%)']]
        X['Total_Alloying'] = X[alloying_elements].sum(axis=1)

        # Ni * Ti interaction
        X['Ni_Ti_interaction'] = X['Ni (at.%)'] * X['Ti (at.%)']

        # Rate ratio
        X['Cooling_Heating_ratio'] = X['Cooling Rate (°C/min)'] / (X['Heating Rate (°C/min)'] + 1e-6)

        print(f"Feature matrix shape: {X.shape}")
        print(f"Number of features: {X.shape[1]}")

        return X

    def prepare_targets(self, df):
        """Prepare target variables (AF and MF)"""
        y_af = df['Austenite Finish Temperature - AF - (°C)']
        y_mf = df['Martensite Finish Temperature - MF - (°C)']

        print(f"\nTarget AF shape: {y_af.shape}")
        print(f"Target MF shape: {y_mf.shape}")
        print(f"\nAF range: [{y_af.min():.2f}, {y_af.max():.2f}]")
        print(f"MF range: [{y_mf.min():.2f}, {y_mf.max():.2f}]")

        return y_af, y_mf

    def train(self, X, y_af, y_mf):
        """
        Train separate XGBoost models for AF and MF
        Using separate models allows independent optimization
        """
        print("\n" + "="*60)
        print("TRAINING XGBOOST MODELS")
        print("="*60)

        # Split data
        X_train, X_test, y_af_train, y_af_test, y_mf_train, y_mf_test = train_test_split(
            X, y_af, y_mf, test_size=0.2, random_state=42
        )

        print(f"\nTraining set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # XGBoost parameters - optimized for tabular data
        xgb_params = {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.05,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'reg_alpha': 0.1,  # L1 regularization
            'reg_lambda': 1.0,  # L2 regularization
            'random_state': 42,
            'n_jobs': -1,
            'early_stopping_rounds': 50,
            'eval_metric': 'rmse'
        }

        # Train model for AF (Austenite Finish)
        print("\n" + "-"*60)
        print("Training XGBoost for AF (Austenite Finish Temperature)")
        print("-"*60)
        self.model_af = xgb.XGBRegressor(**xgb_params)
        self.model_af.fit(
            X_train_scaled, y_af_train,
            eval_set=[(X_test_scaled, y_af_test)],
            verbose=False
        )

        # Train model for MF (Martensite Finish)
        print("\nTraining XGBoost for MF (Martensite Finish Temperature)")
        print("-"*60)
        self.model_mf = xgb.XGBRegressor(**xgb_params)
        self.model_mf.fit(
            X_train_scaled, y_mf_train,
            eval_set=[(X_test_scaled, y_mf_test)],
            verbose=False
        )

        print("\nTraining completed!")

        return X_train_scaled, X_test_scaled, y_af_train, y_af_test, y_mf_train, y_mf_test

    def evaluate(self, X_train, X_test, y_af_train, y_af_test, y_mf_train, y_mf_test):
        """Evaluate model performance"""
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)

        # Predictions
        y_af_train_pred = self.model_af.predict(X_train)
        y_af_test_pred = self.model_af.predict(X_test)
        y_mf_train_pred = self.model_mf.predict(X_train)
        y_mf_test_pred = self.model_mf.predict(X_test)

        # AF Model Metrics
        print("\n" + "-"*60)
        print("AF (Austenite Finish) Model Performance:")
        print("-"*60)
        print(f"Training R² Score:   {r2_score(y_af_train, y_af_train_pred):.4f}")
        print(f"Test R² Score:       {r2_score(y_af_test, y_af_test_pred):.4f}")
        print(f"Training MAE:        {mean_absolute_error(y_af_train, y_af_train_pred):.2f} °C")
        print(f"Test MAE:            {mean_absolute_error(y_af_test, y_af_test_pred):.2f} °C")
        print(f"Training RMSE:       {np.sqrt(mean_squared_error(y_af_train, y_af_train_pred)):.2f} °C")
        print(f"Test RMSE:           {np.sqrt(mean_squared_error(y_af_test, y_af_test_pred)):.2f} °C")

        # MF Model Metrics
        print("\n" + "-"*60)
        print("MF (Martensite Finish) Model Performance:")
        print("-"*60)
        print(f"Training R² Score:   {r2_score(y_mf_train, y_mf_train_pred):.4f}")
        print(f"Test R² Score:       {r2_score(y_mf_test, y_mf_test_pred):.4f}")
        print(f"Training MAE:        {mean_absolute_error(y_mf_train, y_mf_train_pred):.2f} °C")
        print(f"Test MAE:            {mean_absolute_error(y_mf_test, y_mf_test_pred):.2f} °C")
        print(f"Training RMSE:       {np.sqrt(mean_squared_error(y_mf_train, y_mf_train_pred)):.2f} °C")
        print(f"Test RMSE:           {np.sqrt(mean_squared_error(y_mf_test, y_mf_test_pred)):.2f} °C")

        return {
            'af': {
                'train_pred': y_af_train_pred,
                'test_pred': y_af_test_pred,
                'train_r2': r2_score(y_af_train, y_af_train_pred),
                'test_r2': r2_score(y_af_test, y_af_test_pred),
                'test_mae': mean_absolute_error(y_af_test, y_af_test_pred),
                'test_rmse': np.sqrt(mean_squared_error(y_af_test, y_af_test_pred))
            },
            'mf': {
                'train_pred': y_mf_train_pred,
                'test_pred': y_mf_test_pred,
                'train_r2': r2_score(y_mf_train, y_mf_train_pred),
                'test_r2': r2_score(y_mf_test, y_mf_test_pred),
                'test_mae': mean_absolute_error(y_mf_test, y_mf_test_pred),
                'test_rmse': np.sqrt(mean_squared_error(y_mf_test, y_mf_test_pred))
            }
        }

    def plot_feature_importance(self, top_n=20):
        """Plot feature importance for both models"""
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*60)

        # Get feature names (including engineered features)
        feature_names = (
            self.feature_columns +
            ['Ni_Ti_ratio', 'Total_Alloying', 'Ni_Ti_interaction', 'Cooling_Heating_ratio']
        )

        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # AF Feature Importance
        af_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model_af.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)

        axes[0].barh(range(len(af_importance)), af_importance['importance'])
        axes[0].set_yticks(range(len(af_importance)))
        axes[0].set_yticklabels(af_importance['feature'])
        axes[0].invert_yaxis()
        axes[0].set_xlabel('Importance')
        axes[0].set_title(f'Top {top_n} Features - AF Model')
        axes[0].grid(axis='x', alpha=0.3)

        # MF Feature Importance
        mf_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model_mf.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)

        axes[1].barh(range(len(mf_importance)), mf_importance['importance'])
        axes[1].set_yticks(range(len(mf_importance)))
        axes[1].set_yticklabels(mf_importance['feature'])
        axes[1].invert_yaxis()
        axes[1].set_xlabel('Importance')
        axes[1].set_title(f'Top {top_n} Features - MF Model')
        axes[1].grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        print("\nFeature importance plot saved as 'feature_importance.png'")

        # Print top features
        print("\nTop 10 Important Features for AF:")
        for idx, row in af_importance.head(10).iterrows():
            print(f"  {row['feature']:40s} {row['importance']:.4f}")

        print("\nTop 10 Important Features for MF:")
        for idx, row in mf_importance.head(10).iterrows():
            print(f"  {row['feature']:40s} {row['importance']:.4f}")

    def plot_predictions(self, y_train, y_test, y_train_pred, y_test_pred, target_name):
        """Plot predicted vs actual values"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Training set
        axes[0].scatter(y_train, y_train_pred, alpha=0.5, s=20)
        axes[0].plot([y_train.min(), y_train.max()],
                     [y_train.min(), y_train.max()], 'r--', lw=2)
        axes[0].set_xlabel(f'Actual {target_name} (°C)')
        axes[0].set_ylabel(f'Predicted {target_name} (°C)')
        axes[0].set_title(f'{target_name} - Training Set')
        axes[0].grid(alpha=0.3)

        # Test set
        axes[1].scatter(y_test, y_test_pred, alpha=0.5, s=20, color='green')
        axes[1].plot([y_test.min(), y_test.max()],
                     [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[1].set_xlabel(f'Actual {target_name} (°C)')
        axes[1].set_ylabel(f'Predicted {target_name} (°C)')
        axes[1].set_title(f'{target_name} - Test Set')
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        return fig

    def save_models(self):
        """Save trained models and scaler"""
        print("\n" + "="*60)
        print("SAVING MODELS")
        print("="*60)
        joblib.dump(self.model_af, 'xgboost_af_model.pkl')
        joblib.dump(self.model_mf, 'xgboost_mf_model.pkl')
        joblib.dump(self.scaler, 'feature_scaler.pkl')
        print("Models saved:")
        print("  - xgboost_af_model.pkl")
        print("  - xgboost_mf_model.pkl")
        print("  - feature_scaler.pkl")

    def load_models(self):
        """Load pre-trained models"""
        self.model_af = joblib.load('xgboost_af_model.pkl')
        self.model_mf = joblib.load('xgboost_mf_model.pkl')
        self.scaler = joblib.load('feature_scaler.pkl')
        print("Models loaded successfully!")


def main():
    """Main execution function"""
    print("="*60)
    print("XGBOOST MODEL FOR SHAPE MEMORY ALLOY PREDICTION")
    print("="*60)

    # Initialize model
    sma_model = SMAXGBoostModel()

    # Load data
    df = sma_model.load_data('dataset/Combined_SMA_Dataset_Filled.csv')

    # Prepare features and targets
    X = sma_model.prepare_features(df)
    y_af, y_mf = sma_model.prepare_targets(df)

    # Train models
    X_train, X_test, y_af_train, y_af_test, y_mf_train, y_mf_test = sma_model.train(X, y_af, y_mf)

    # Evaluate models
    results = sma_model.evaluate(X_train, X_test, y_af_train, y_af_test, y_mf_train, y_mf_test)

    # Feature importance analysis
    sma_model.plot_feature_importance(top_n=20)

    # Plot predictions
    print("\n" + "="*60)
    print("GENERATING VISUALIZATION PLOTS")
    print("="*60)

    fig_af = sma_model.plot_predictions(
        y_af_train, y_af_test,
        results['af']['train_pred'], results['af']['test_pred'],
        'AF'
    )
    fig_af.savefig('af_predictions.png', dpi=300, bbox_inches='tight')
    print("\nAF predictions plot saved as 'af_predictions.png'")

    fig_mf = sma_model.plot_predictions(
        y_mf_train, y_mf_test,
        results['mf']['train_pred'], results['mf']['test_pred'],
        'MF'
    )
    fig_mf.savefig('mf_predictions.png', dpi=300, bbox_inches='tight')
    print("MF predictions plot saved as 'mf_predictions.png'")

    # Save models
    sma_model.save_models()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nAF Model - Test R²: {results['af']['test_r2']:.4f}, MAE: {results['af']['test_mae']:.2f}°C")
    print(f"MF Model - Test R²: {results['mf']['test_r2']:.4f}, MAE: {results['mf']['test_mae']:.2f}°C")
    print("\n" + "="*60)
    print("ALL TASKS COMPLETED SUCCESSFULLY!")
    print("="*60)


if __name__ == "__main__":
    main()

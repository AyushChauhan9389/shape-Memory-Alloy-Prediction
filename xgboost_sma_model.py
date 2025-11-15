"""
XGBoost Model for Shape Memory Alloy (SMA) Temperature Prediction
Predicts Austenite Finish (AF) and Martensite Finish (MF) temperatures
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class SMAXGBoostModel:
    """
    XGBoost-based predictor for Shape Memory Alloy transformation temperatures
    """

    def __init__(self, use_tuning=True):
        self.model_af = None
        self.model_mf = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.use_tuning = use_tuning
        self.best_params_af = None
        self.best_params_mf = None
        self.use_gpu = self._check_gpu_available()

    def _check_gpu_available(self):
        """Check if GPU is available for XGBoost"""
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=2)
            gpu_available = result.returncode == 0
            if gpu_available:
                print("🚀 GPU detected - will use GPU acceleration for training!")
            else:
                print("💻 No GPU detected - using CPU (consider using GPU for 10-50x speedup)")
            return gpu_available
        except:
            print("💻 No GPU detected - using CPU")
            return False

    def load_data(self, filepath):
        """Load and prepare the SMA dataset"""
        print(f"Loading data from {filepath}...")
        with tqdm(total=100, desc="Loading CSV", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
            df = pd.read_csv(filepath)
            pbar.update(100)
        print(f"✓ Dataset loaded successfully: {df.shape[0]} samples, {df.shape[1]} features")
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

        # Combine all features
        self.feature_columns = element_cols + additional_features

        # Create feature matrix with progress
        with tqdm(total=5, desc="Feature Engineering", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
            X = df[self.feature_columns].copy()
            pbar.update(1)

            # Ni/Ti ratio (important for NiTi alloys)
            X['Ni_Ti_ratio'] = X['Ni (at.%)'] / (X['Ti (at.%)'] + 1e-6)
            pbar.update(1)

            # Total alloying elements (excluding Ni and Ti)
            alloying_elements = [col for col in element_cols if col not in ['Ni (at.%)', 'Ti (at.%)']]
            X['Total_Alloying'] = X[alloying_elements].sum(axis=1)
            pbar.update(1)

            # Ni * Ti interaction
            X['Ni_Ti_interaction'] = X['Ni (at.%)'] * X['Ti (at.%)']
            pbar.update(1)

            # Rate ratio
            X['Cooling_Heating_ratio'] = X['Cooling Rate (°C/min)'] / (X['Heating Rate (°C/min)'] + 1e-6)
            pbar.update(1)

        print(f"✓ Features ready: {X.shape[1]} total features (24 base + 4 engineered)")

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
        print("TRAINING XGBOOST MODELS WITH OPTIMIZATION")
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

        # Improved base parameters - REDUCED OVERFITTING
        base_params = {
            'random_state': 42,
            'eval_metric': 'rmse'
        }

        # Add GPU support if available
        if self.use_gpu:
            base_params['tree_method'] = 'gpu_hist'
            base_params['device'] = 'cuda'
            print("⚡ Using GPU acceleration (tree_method='gpu_hist')")
        else:
            base_params['n_jobs'] = -1
            base_params['tree_method'] = 'hist'  # Fast CPU method
            print("⚙️  Using CPU with fast histogram method")

        # Better default parameters to reduce overfitting
        improved_params = {
            'n_estimators': 300,          # Reduced from 500
            'max_depth': 4,                # Reduced from 6 (prevents overfitting)
            'learning_rate': 0.01,         # Reduced from 0.05 (slower, more stable)
            'min_child_weight': 5,         # Increased from 3 (more conservative)
            'subsample': 0.7,              # Reduced from 0.8
            'colsample_bytree': 0.7,       # Reduced from 0.8
            'gamma': 0.2,                  # Increased from 0.1 (more pruning)
            'reg_alpha': 1.0,              # Increased L1 from 0.1
            'reg_lambda': 10.0,            # Increased L2 from 1.0
            **base_params
        }

        if self.use_tuning:
            # Hyperparameter tuning grid
            param_grid = {
                'n_estimators': [200, 300, 500],
                'max_depth': [3, 4, 5],
                'learning_rate': [0.01, 0.05, 0.1],
                'min_child_weight': [3, 5, 7],
                'subsample': [0.6, 0.7, 0.8],
                'colsample_bytree': [0.6, 0.7, 0.8],
                'gamma': [0.1, 0.2, 0.5],
                'reg_alpha': [0.1, 1.0, 5.0],
                'reg_lambda': [1.0, 5.0, 10.0]
            }

            xgb_params = improved_params
        else:
            xgb_params = improved_params

        # Train model for AF (Austenite Finish)
        print("\n" + "-"*60)
        print("Training XGBoost for AF (Austenite Finish Temperature)")
        print("-"*60)

        if self.use_tuning:
            # Use RandomizedSearchCV for faster tuning (especially on CPU)
            # Tests 100 random combinations instead of all 19,683
            n_iter = 100 if not self.use_gpu else 200  # More iterations on GPU
            print(f"⚙️  Running RandomizedSearchCV for hyperparameter tuning...")
            print(f"   Testing {n_iter} random parameter combinations with 3-fold CV")
            print(f"   (Much faster than GridSearch's {3*3*3*3*3*3*3*3*3} combinations)")

            grid_search_af = RandomizedSearchCV(
                estimator=xgb.XGBRegressor(**base_params),
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=3,
                scoring='r2',
                n_jobs=-1 if not self.use_gpu else 1,
                verbose=1,
                random_state=42
            )

            with tqdm(total=100, desc="GridSearch AF", bar_format='{l_bar}{bar}| {n_fmt}%') as pbar:
                import threading

                def tune_af():
                    grid_search_af.fit(X_train_scaled, y_af_train)

                tune_thread = threading.Thread(target=tune_af)
                tune_thread.start()

                while tune_thread.is_alive():
                    if pbar.n < 95:
                        pbar.update(5)
                    time.sleep(2)

                tune_thread.join()
                pbar.update(100 - pbar.n)

            self.model_af = grid_search_af.best_estimator_
            self.best_params_af = grid_search_af.best_params_
            print(f"✓ Best AF parameters found:")
            for param, value in self.best_params_af.items():
                print(f"   {param}: {value}")
            print(f"✓ Best CV R² score: {grid_search_af.best_score_:.4f}")
        else:
            self.model_af = xgb.XGBRegressor(**xgb_params)

            with tqdm(total=100, desc="Training AF Model", bar_format='{l_bar}{bar}| {n_fmt}%') as pbar:
                import threading

                def train_af():
                    self.model_af.fit(
                        X_train_scaled, y_af_train,
                        eval_set=[(X_test_scaled, y_af_test)],
                        verbose=False
                    )

                train_thread = threading.Thread(target=train_af)
                train_thread.start()

                while train_thread.is_alive():
                    if pbar.n < 95:
                        pbar.update(5)
                    time.sleep(0.5)

                train_thread.join()
                pbar.update(100 - pbar.n)

            print("✓ AF model training completed")

        # Train model for MF (Martensite Finish)
        print("\n" + "-"*60)
        print("Training XGBoost for MF (Martensite Finish Temperature)")
        print("-"*60)

        if self.use_tuning:
            n_iter = 100 if not self.use_gpu else 200
            print(f"⚙️  Running RandomizedSearchCV for hyperparameter tuning...")
            print(f"   Testing {n_iter} random parameter combinations with 3-fold CV")
            print(f"   (Much faster than GridSearch's {3*3*3*3*3*3*3*3*3} combinations)")

            grid_search_mf = RandomizedSearchCV(
                estimator=xgb.XGBRegressor(**base_params),
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=3,
                scoring='r2',
                n_jobs=-1 if not self.use_gpu else 1,
                verbose=1,
                random_state=42
            )

            with tqdm(total=100, desc="GridSearch MF", bar_format='{l_bar}{bar}| {n_fmt}%') as pbar:
                import threading

                def tune_mf():
                    grid_search_mf.fit(X_train_scaled, y_mf_train)

                tune_thread = threading.Thread(target=tune_mf)
                tune_thread.start()

                while tune_thread.is_alive():
                    if pbar.n < 95:
                        pbar.update(5)
                    time.sleep(2)

                tune_thread.join()
                pbar.update(100 - pbar.n)

            self.model_mf = grid_search_mf.best_estimator_
            self.best_params_mf = grid_search_mf.best_params_
            print(f"✓ Best MF parameters found:")
            for param, value in self.best_params_mf.items():
                print(f"   {param}: {value}")
            print(f"✓ Best CV R² score: {grid_search_mf.best_score_:.4f}")
        else:
            self.model_mf = xgb.XGBRegressor(**xgb_params)

            with tqdm(total=100, desc="Training MF Model", bar_format='{l_bar}{bar}| {n_fmt}%') as pbar:
                import threading

                def train_mf():
                    self.model_mf.fit(
                        X_train_scaled, y_mf_train,
                        eval_set=[(X_test_scaled, y_mf_test)],
                        verbose=False
                    )

                train_thread = threading.Thread(target=train_mf)
                train_thread.start()

                while train_thread.is_alive():
                    if pbar.n < 95:
                        pbar.update(5)
                    time.sleep(0.5)

                train_thread.join()
                pbar.update(100 - pbar.n)

            print("✓ MF model training completed")

        return X_train_scaled, X_test_scaled, y_af_train, y_af_test, y_mf_train, y_mf_test

    def evaluate(self, X_train, X_test, y_af_train, y_af_test, y_mf_train, y_mf_test):
        """Evaluate model performance"""
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)

        # Predictions with progress bar
        print("\nGenerating predictions...")
        with tqdm(total=4, desc="Making Predictions", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
            y_af_train_pred = self.model_af.predict(X_train)
            pbar.update(1)
            y_af_test_pred = self.model_af.predict(X_test)
            pbar.update(1)
            y_mf_train_pred = self.model_mf.predict(X_train)
            pbar.update(1)
            y_mf_test_pred = self.model_mf.predict(X_test)
            pbar.update(1)

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
        with tqdm(total=3, desc="Saving Models", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
            joblib.dump(self.model_af, 'xgboost_af_model.pkl')
            pbar.update(1)
            joblib.dump(self.model_mf, 'xgboost_mf_model.pkl')
            pbar.update(1)
            joblib.dump(self.scaler, 'feature_scaler.pkl')
            pbar.update(1)
        print("✓ Models saved successfully:")
        print("  - xgboost_af_model.pkl")
        print("  - xgboost_mf_model.pkl")
        print("  - feature_scaler.pkl")

    def load_models(self):
        """Load pre-trained models"""
        self.model_af = joblib.load('xgboost_af_model.pkl')
        self.model_mf = joblib.load('xgboost_mf_model.pkl')
        self.scaler = joblib.load('feature_scaler.pkl')
        print("Models loaded successfully!")


def main(use_tuning=False):
    """Main execution function"""
    print("="*60)
    print("XGBOOST MODEL FOR SHAPE MEMORY ALLOY PREDICTION")
    if use_tuning:
        print("WITH HYPERPARAMETER TUNING (GridSearchCV)")
    else:
        print("WITH IMPROVED REGULARIZATION PARAMETERS")
    print("="*60)

    # Initialize model
    sma_model = SMAXGBoostModel(use_tuning=use_tuning)

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

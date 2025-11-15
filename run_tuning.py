"""
Run XGBoost Model with RandomizedSearchCV Hyperparameter Tuning
- Uses GPU if available (10-50x faster)
- Tests 100 random combinations (much faster than full GridSearch)
- Estimated time: 5-15 minutes on CPU, <2 minutes on GPU
"""

from xgboost_sma_model import main

if __name__ == "__main__":
    print("Starting hyperparameter tuning with RandomizedSearchCV...")
    print("- GPU: Automatically enabled if available")
    print("- Testing 100 random parameter combinations (vs 19,683 for full grid)")
    print("- Much faster while still finding near-optimal parameters\n")

    main(use_tuning=True)

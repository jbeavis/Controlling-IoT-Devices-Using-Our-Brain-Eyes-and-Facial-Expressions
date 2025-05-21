from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV

import pandas as pd
import pickle

def trainModel(df_features, totalEpochs):
    # Separate features from markers for machine learning
    X = df_features.drop(columns=["Timestamp", "Event Type"])
    y = df_features["Event Type"]  # Labels

    feature_names = X.columns  # Save the column names

    # Undersample -- I blink far more often than I do anything else during recordings (I can't help it) but this skews things
    undersampler = RandomUnderSampler(random_state=42)    
    X, y = undersampler.fit_resample(X, y)

    # Normalize feature values (important for consistency)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # Save scaler (for real time prediction laters)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Split into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42) # Random seed

    print(f"Training samples: {len(X_train)}, testing samples: {len(X_test)}")

    # Initialize model
    rf_model = RandomForestClassifier(min_samples_leaf = 1, n_estimators=500, bootstrap = False, random_state=42, max_features="log2", max_depth=None, min_samples_split=2, class_weight={-1:20, 1:10, 2:10, 3:10, 4:15, 5:10, 6:10, 7:10, 8:5, 9:10}) # More estimators = more accurate? but slower
    rf_model.fit(X_train, y_train)
    # Train the model
    print("Model training complete! :)\n")
    
    # # Get feature importances
    # importances = rf_model.feature_importances_
    # importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    # importance_df = importance_df.sort_values(by="Importance", ascending=False)
    # pd.set_option('display.max_rows', None)
    # print(importance_df)

    rf_model = CalibratedClassifierCV(rf_model, method='sigmoid')  # Platt Scaling TODO check why
    rf_model.fit(X_train, y_train)

    return rf_model, X_test, y_test

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

param_grid = {
    'n_estimators': [200, 400, 500],  # More trees for stability
    'max_depth': [10, 20, 30, None],  # More depth options
    'min_samples_split': [2, 4, 6, 8, 10],  # More granularity
    'min_samples_leaf': [1, 2, 4, 6],  # Prevents overfitting
    'bootstrap': [True, False],  # Bootstrapping or using all data
}

def hyperparameterTuning(df_features):
    print("\nEntering hyperparameter tuning...\n")
    # Separate features from markers for machine learning
    X = df_features.drop(columns=["Timestamp", "Event Type"])
    y = df_features["Event Type"]  # Labels

    # Undersample -- I blink far more often than I do anything else during recordings (I can't help it) but this skews things
    #sampling_strategy={1: totalEpochs//9}, 
    undersampler = RandomUnderSampler(random_state=42) 
    X, y = undersampler.fit_resample(X, y)

    # Normalize feature values (important for consistency)
    with open("scaler.pkl", "rb") as f: # Must use the same scaler as the one used for creating the model
        scaler = pickle.load(f)
    X_scaled = scaler.fit_transform(X)

    # Split into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42) # Random seed

    print(f"Training samples: {len(X_train)}, testing samples: {len(X_test)}")

    # Initialize model
    rf_model = RandomForestClassifier(n_estimators=200, random_state=42, max_features="log2", max_depth=10, min_samples_split=2, class_weight={-1:5,1:1,2:1,3:1,4:1,5:1,6:1,7:1,8:1,9:1}) # More estimators = more accurate? but slower

    # Perform Grid Search with 5-fold cross-validation
    grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

    # Fit the model on your training data
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters
    print("Best Hyperparameters:", grid_search.best_params_)

    return grid_search.best_params_

from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split, cross_val_score, KFold

import numpy as np

import pickle

import loadData
import filterData
import generateEpochs
import featureExtraction
import trainModel
# import hyperparameterTuning
# import Main.AdditionalScripts.plotEXG as plotEXG

# Load the data
df = loadData.loadData()

# Filter Data
df_cleaned = filterData.applyFilters(df)

# Extract event epochs
event_epochs = generateEpochs.generate_event_epochs(df_cleaned)
print(f"Created {len(event_epochs)} epochs with markers.")

# Extract idle epochs
idle_epochs = generateEpochs.generate_idle_epochs(df_cleaned, len(event_epochs))
print(f"Created {len(idle_epochs)} idle epochs")

# Combine epochs
all_epochs = {**event_epochs, **idle_epochs}
print(f"Created {len(all_epochs)} total epochs\n")

# Extract Features From epochs
df_features = featureExtraction.extract_features(all_epochs)

# Train Multiclass Model
# rf_model, X_test, y_test = trainModel.trainModel(df_features[df_features["Event Type"] != -1], len(all_epochs))
rf_model, X_test, y_test = trainModel.trainModel(df_features, len(all_epochs))

# Predict on test set
y_pred = rf_model.predict(X_test)

# Print accuracy and detailed report
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# Hyperparameter tuning:
# hyperparameterTuning.hyperparameterTuning(df_features)

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=set(y_test), yticklabels=set(y_test))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(rf_model, f)

# Plot EXG
# plotEXG.plotEXG(all_epochs)

X = df_features.drop(columns=["Timestamp", "Event Type"])
y = df_features["Event Type"]  # Labels

# Undersample -- I blink far more often than I do anything else during recordings (I can't help it) but this skews things
undersampler = RandomUnderSampler(random_state=42) 
X, y = undersampler.fit_resample(X, y)

# Normalize feature values (important for consistency)
with open("scaler.pkl", "rb") as f: # Must use the same scaler as the one used for creating the model
    scaler = pickle.load(f)
X_scaled = scaler.fit_transform(X)

# Evaluate 
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(rf_model, X_scaled, y, cv=kfold, scoring='accuracy')
print(f"Average accuracy (k fold): {scores.mean():.4f}")
print(f"Standard deviation: {np.std(scores)}")
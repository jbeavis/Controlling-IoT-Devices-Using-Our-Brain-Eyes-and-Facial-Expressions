import pandas as pd
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import welch, find_peaks

import numpy as np

def extract_features(epochs, fs=200):
    features = []

    for (timestamp, event_type), epoch_df in epochs.items():
        feature_dict = {"Timestamp": timestamp, "Event Type": event_type}
        
        for ch in [0,1,2,3]: # Extract features for each channel
            signal = epoch_df[f"EXG Channel {ch}"].values
            
            # Time domain features
            feature_dict[f"Ch{ch}_Mean"] = np.mean(signal)
            feature_dict[f"Ch{ch}_Variance"] = np.var(signal)
            feature_dict[f"Ch{ch}_Skewness"] = skew(signal)
            feature_dict[f"Ch{ch}_Kurtosis"] = kurtosis(signal)
            feature_dict[f"Ch{ch}_RMS"] = np.sqrt(np.mean(signal ** 2))
            
            # For idle 
            feature_dict[f"Ch{ch}_Entropy"] = entropy(np.abs(signal)) # Captures how unpredictable a signal is - idle was misclassified before this
            feature_dict[f"Ch{ch}_ZeroCrossingRate"] = np.sum(np.diff(np.sign(signal)) != 0) / len(signal)
            feature_dict[f"Ch{ch}_RollingStd"] = np.std(pd.Series(signal).rolling(window=10).mean())

            # Hjorth Parameters, for left/right confusion
            first_deriv = np.diff(signal)
            second_deriv = np.diff(first_deriv)
            
            var_zero = np.var(signal)
            var_d1 = np.var(first_deriv)
            var_d2 = np.var(second_deriv)
            
            mobility = np.sqrt(var_d1 / var_zero)
            complexity = np.sqrt(var_d2 / var_d1) / mobility
                

            feature_dict[f"Ch{ch}_Mobility"] = mobility
            feature_dict[f"Ch{ch}_Complexity"] = complexity

            # Frustration vs Jaw (frustration is no longer used but these features are still useful)
            feature_dict[f"Ch{ch}_PeakToPeak"] = np.ptp(signal)
            feature_dict[f"Ch{ch}_Energy"] = np.sum(signal ** 2)
            feature_dict["Ch3_High_Activity"] = (epoch_df["EXG Channel 3"].abs() > 10).sum()
            feature_dict["Ch3_ZeroCrossings"] = ((epoch_df["EXG Channel 3"][:-1] * epoch_df["EXG Channel 3"][1:]) < 0).sum()
            feature_dict["Ch3_Max_Derivative"] = np.max(np.abs(np.diff(epoch_df["EXG Channel 3"]))) # for steepness of slope

            # Frequency domain features
            nperseg = min(200, len(signal))  # This must be variable because the live predictor won't have 200 samples to begin with
            freqs, psd = welch(signal, fs=fs, nperseg= nperseg)
            feature_dict[f"Ch{ch}_Delta"] = np.sum(psd[(freqs >= 0.5) & (freqs < 4)])
            feature_dict[f"Ch{ch}_Theta"] = np.sum(psd[(freqs >= 4) & (freqs < 8)])
            feature_dict[f"Ch{ch}_Alpha"] = np.sum(psd[(freqs >= 8) & (freqs < 13)])
            feature_dict[f"Ch{ch}_Beta"] = np.sum(psd[(freqs >= 13) & (freqs < 30)])
        features.append(feature_dict)
    return pd.DataFrame(features)

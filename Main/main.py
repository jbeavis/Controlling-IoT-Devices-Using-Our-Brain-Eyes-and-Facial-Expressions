from pylsl import StreamInlet, resolve_byprop
import pandas as pd
import threading
import queue
import filterData
import featureExtraction
import pickle
from collections import deque 
from sklearn.preprocessing import RobustScaler
import time
import numpy as np

import matplotlib.pyplot as plt

from musicPlayer import MusicPlayer

# Future work: Smoothing, only classify if the past few were the same?

start_time = time.time()  # Store start time

markers = {-1:"Idle", 1: "Blink", 2: "Left", 3: "Right", 4: "Jaw", 5: "Up", 6:"Down", 7: "Happy", 8: "Frustrated", 9: "Alpha"}

# Load models and scaler
with open("model.pkl", "rb") as f:
    rf_model = pickle.load(f) # Random forest classification model made earlier using the preProcessing.py 
with open("scaler.pkl", "rb") as f: # Must use the same scaler as the one used for creating the model
    scaler = pickle.load(f)

df_columns = ["EXG Channel 0", "EXG Channel 1","EXG Channel 2", "EXG Channel 3", "Timestamp"]

# first resolve an EEG stream on the lab network
print("looking for an EEG stream...")
streams = resolve_byprop("type", "EEG")

# create a new inlet to read from the stream
inlet = StreamInlet(streams[0])

# Define parameters
window_size = 400  
step_size = 90   # overlap
buffer = deque(maxlen=window_size)  # Rolling buffer, maxlen means when a new thing is added, the oldest item is popped if it's over the max len

df_columns = ["EXG Channel 0", "EXG Channel 1", "EXG Channel 2", "EXG Channel 3", "Timestamp"]

# For sharing between threads
dataQueue = queue.Queue(maxsize=0)
predictionQueue = queue.Queue(maxsize=0) 

thresholds = {
    -1: 0.5, # Idle
    1: 0.4,  # Blink
    2: 0.4, # Left
    3: 0.35, # Right
    4: 0.35,  # Jaw
    5: 0.4, # Up
    6: 0.4, # Down
    7: 0.3,  # Happy
    8: 0.5,  # Frustrated
    9: 0.008  # alpha
}

# !Read Data Thread
def readData():
    # firstWindow = True
    # Streaming loop
    while True:
        # Collect new samples
        for x in range(step_size):  # Only collect step_size samples at a time
            sample, timestamp = inlet.pull_sample()
            buffer.append(sample + [timestamp])  # Append data with timestamp

        # Convert buffer to DataFrame
        sample_df = pd.DataFrame(buffer, columns=df_columns)
        dataQueue.put(sample_df)


# !Prediction Thread
def makePredictions():
    bigQueueFlag = False
    while True:
        try:
            queueSize = dataQueue.qsize()
            if queueSize > 2:
                print(f"Queue size for predictions: {dataQueue.qsize()}")
                bigQueueFlag = True
            elif bigQueueFlag == True:
                print("Queue empty")
                bigQueueFlag = False
            sample_df = dataQueue.get(timeout=0.1)

            # Clean data
            cleaned_df = filterData.applyFilters(sample_df)
            baseline = np.mean(cleaned_df.iloc[:, :-1])  # Compute mean per channel
            cleaned_df.iloc[:, :-1] -= baseline  # Center data around 0

            # Prepare format for feature extraction
            formatForExtraction = {(sample_df["Timestamp"].iloc[0], -1): cleaned_df}
            features = featureExtraction.extract_features(formatForExtraction)

            # Preprocess & predict
            features = features.drop(columns=["Timestamp", "Event Type"], errors="ignore")
            features = scaler.transform(features)

            # Test for idleness first:
            prediction = rf_model.predict(features)
            proba = rf_model.predict_proba(features)[0]
            maxProba = max(proba)
            # print(proba[7])
            if proba[0] < 0.5: 
                # print(markers[prediction[0]], proba)
                if maxProba > thresholds[prediction[0]] and markers[prediction[0]] != "Blink": # Must be confident, thresholding
                    if markers[prediction[0]] == "Alpha":
                        pass
                    elif markers[prediction[0]] == "Jaw" and proba[7] > thresholds[9]:
                        prediction = [9]
                    predictionQueue.put(prediction) # Add to threadsafe queue
        except queue.Empty:
            continue

# !Music Control Thread
def controlMusic():
    lastEventTime = time.time()
    musicPlayer = MusicPlayer()
    musicPlayer.play()
    lastPrediction=0
    while True:
        currentTime = time.time()
        try:
            prediction = predictionQueue.get(timeout=0.1)  
            if currentTime - lastEventTime >= 2.2 or (currentTime - lastEventTime >= 1 and lastPrediction != prediction) : # Only report predicted marker if there hasn't been another one recently
                lastEventTime = currentTime
                print()
                match markers[prediction[0]]:
                    case "Left":
                        # Previous track
                        print("Previous track")
                        musicPlayer.prev_track()
                    case "Right":
                        # Next track
                        print("Next track")
                        musicPlayer.next_track()
                    case "Jaw":
                        # Pause/ unpause track
                        musicPlayer.pause()
                    case "Up":
                        # Volume up
                        print("Volume up")
                        musicPlayer.increase_volume()
                        pass
                    case "Down":
                        # Volume down
                        print("Volume down")
                        musicPlayer.decrease_volume()
                        pass
                    case "Happy":
                        # Switch playlist to happy (if sad)
                        print("Happy playlist (wip)")
                        musicPlayer.emotion = "Happy"
                        pass
                    case "Frustrated":
                        # Switch playlist to sad (if happy)
                        print("Sad Playlist (wip)")
                        musicPlayer.emotion = "Sad"
                        pass
                    case "Alpha":
                        print("Alpha Waves Detected, switching playlist...")
                        if musicPlayer.emotion == "Happy":
                            musicPlayer.emotion = "Sad"
                        else:
                            musicPlayer.emotion == "Happy"
                        pass
            lastPrediction=prediction
        except queue.Empty:
            continue

if __name__ == "__main__":
    # Start streaming thread
    streamingThread = threading.Thread(target=readData, daemon=True)
    streamingThread.start()

    # Start prediction thread
    prediction_thread = threading.Thread(target=makePredictions, daemon=True)
    prediction_thread.start()

    # Start music control thread
    music_thread = threading.Thread(target=controlMusic, daemon=True)
    music_thread.start()

    # Keep main thread alive
    while True:
        time.sleep(1)

import random 

def generate_event_epochs(df, window=(-1, 1.0), fs=200):
    epochs = {}
    samples_before = int(abs(window[0]) * fs)  # Number of samples before event
    samples_after = int(window[1] * fs)        # Number of samples after event
    total_samples = samples_before + samples_after

    # Find event timestamps and corresponding marker values
    # The 'isin' defines which events to make epochs for. 
    events = df[df["Marker Channel"].isin([1,2,3,4,5,6,9])][["Timestamp", "Marker Channel"]].values  # Get time + marker value

    print(f"Number of events: {len(events)}")

    for event_time, marker_value in events:
        # Define the epoch time range
        start_time = event_time + window[0]
        end_time = event_time + window[1]

        # Extract data within this time window
        epoch_df = df[(df["Timestamp"] >= start_time) & (df["Timestamp"] <= end_time)].copy()
        
        # Ensure consistent(ish) length of epoch (there will be some packet loss as recording over ble, but they may still be useful)
        if len(epoch_df) >= total_samples * 0.9825:
            # Add marker column to indicate event type
            epoch_df["Event Type"] = marker_value  
            epochs[(event_time, marker_value)] = epoch_df
    return epochs

def generate_idle_epochs(df, totalEpochsWithMarkers, window=(-1, 1.0), fs=200):
    import numpy as np

    idle_epochs = {}
    samples_before = int(abs(window[0]) * fs)
    samples_after = int(window[1] * fs)
    total_samples = samples_before + samples_after

    # Define a buffer zone around each event marker
    event_markers = df[df["Marker Channel"] != 0]["Timestamp"].values
    event_ranges = []
    for event_time in event_markers:
        start = event_time + window[0]  # window 
        end = event_time + window[1]    # and end 1s after the event
        event_ranges.append((start, end))

    event_ranges = np.array(event_ranges)  # Shape (n, 2)

    # Possible timestamps: areas with no events
    candidate_df = df[df["Marker Channel"] == 0]
    timestamps = candidate_df["Timestamp"].values

    i = 0
    while i < len(timestamps) - total_samples:
        if len(idle_epochs) >= (round(totalEpochsWithMarkers / 8)): # Don't extract too many, wastes time
            break

        center_time = timestamps[i]
        start_time = center_time + window[0]
        end_time = center_time + window[1]

        # skip if possible idle window overlaps with any event window
        if np.any((start_time < event_ranges[:, 1]) & (end_time > event_ranges[:, 0])):
            i += 1
            continue

        # if it's clean, extract the idle epoch
        epoch_df = df[(df["Timestamp"] >= start_time) & (df["Timestamp"] <= end_time)].copy()

        if len(epoch_df) >= total_samples * 0.9825:
            epoch_df["Event Type"] = -1
            idle_epochs[(center_time, -1)] = epoch_df
            i += total_samples  # Skip ahead by full epoch
        else:
            i += 1  # if too short short, skip forward

    return idle_epochs
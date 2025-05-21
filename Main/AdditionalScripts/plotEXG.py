import pandas as pd
import numpy as np
import random 
import matplotlib.pyplot as plt


def plotEXG(all_epochs):
    random.seed(42)
    # Repeat for all events
    for selected_event_type in [-1,1,2,3,4,5,6,7,8]:
        # Convert selected event type to match stored format
        selected_event_type = np.float64(selected_event_type)

        # Filter epochs that match the selected event type
        matching_epochs = {key: value for key, value in all_epochs.items() if key[1] == selected_event_type}

        # Extract only the timestamps
        matching_keys_list = list(matching_epochs.keys())

        if len(matching_keys_list) > 0:
            # Select one random timestamp
            selected_key = random.choice(matching_keys_list)  
            epoch_data = matching_epochs[selected_key]  # Extract Exg data
            # print(len(epoch_data))

            # Filter only EXG channels
            exg_channels = [col for col in epoch_data.columns if "EXG" in col]

            # Extract data (EXG channels only)
            eeg_data = epoch_data[exg_channels].values.T /10 # Transpose for plotting

            # Create a figure for plotting
            plt.figure(figsize=(12, 5))

            # Plot each EXG channel with a label
            for i, channel_data in enumerate(eeg_data):
                plt.plot(channel_data, label=exg_channels[i], alpha=0.8)

            # Formatting
            plt.title(f"EEG Signal - Event Type {selected_event_type}", fontsize=14)
            plt.xlabel("Time (samples)", fontsize=12)
            plt.ylabel("Amplitude (µV)", fontsize=12)  
            plt.ylim(-200, 200)
            plt.grid(True)
            plt.legend(loc="upper right", fontsize=10)  # Add legend to differentiate channels
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)

            # Show the plot
            plt.show()

        else:
            print(f"No epochs found for event type: {selected_event_type}")
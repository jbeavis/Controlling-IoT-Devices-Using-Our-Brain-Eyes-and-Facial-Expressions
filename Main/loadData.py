import pandas as pd
import os
# Make txt
# Add header
# Go through each txt in folder
# Skip first x lines
# Add all subsequent lines to new txt
    

def loadData():
    # Define the header (must only once in the final file)
    header = """%OpenBCI Raw EXG Data
%Number of channels = 4
%Sample Rate = 200 Hz
%Board = OpenBCI_GUI$BoardGanglionNative
Sample Index, EXG Channel 0, EXG Channel 1, EXG Channel 2, EXG Channel 3, Accel Channel 0, Accel Channel 1, Accel Channel 2, Other, Other, Other, Other, Other, Timestamp, Marker Channel, Timestamp (Formatted)"""

    output_file = "collectedData.txt"
    recordings_dir = "Recordings"

    # Collect all text files
    recordings = []
    print("Loading sessions... ")
    for session in os.listdir(recordings_dir):
        print(session)
        session_path = os.path.join(recordings_dir, session)
        for file in os.listdir(session_path):
            if file.endswith(".txt"):
                recordings.append(os.path.join(session_path, file))
    print("")

    with open(output_file, "w") as out_file:
        out_file.write(header+"\n")  # Write the header once
        
        for file_path in recordings:
            with open(file_path, "r") as f:
                lines = f.readlines()

            # Write everything after the repeated header 
            out_file.writelines(lines[5:])  

    with open(output_file, "r") as f:
        lines = f.readlines()
    start_idx = next(i for i, line in enumerate(lines) if not line.startswith('%')) # AI generated line

    dataframe = pd.read_csv(output_file, skiprows=start_idx) # Put data into pandas dataframe

    dataframe.rename(columns={' EXG Channel 0': 'EXG Channel 0', ' EXG Channel 1': 'EXG Channel 1',' EXG Channel 2': 'EXG Channel 2',' EXG Channel 3': 'EXG Channel 3',' Timestamp': 'Timestamp', ' Marker Channel': 'Marker Channel',' Timestamp (Formatted)': 'Timestamp (Formatted)',}, inplace=True) # Rename columns as they have spaces in them

    # Drop unneeded columns
    columns_to_drop = [" Other", " Other.1", " Other.2", " Other.3", " Other.4", " Accel Channel 0", " Accel Channel 1", " Accel Channel 2"]
    df_cleaned = dataframe.drop(columns=[col for col in columns_to_drop])

    # Convert data types (keeping Timestamp (Formatted) and Marker Channel)
    df_cleaned = df_cleaned.astype({
        "Sample Index": int,
        "EXG Channel 0": float,
        "EXG Channel 1": float,
        "EXG Channel 2": float,
        "EXG Channel 3": float,
        "Timestamp": float,
        "Marker Channel": float
    })
    return df_cleaned

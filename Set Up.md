# How to set up device

Before running, ensure the device isn't sitting on a laptop or other machines. Turn off Bluetooth on nearby mobile phones and other devices.

Unless stated otherwise, use top row of pins. Use relative directions (eg. YOUR right, not right as someone is looking at you).
- Connect black electrodes to D_G and REF pins, stick to **earlobes** (or behind ear).
- Connect first red electrode to +1 pin, stick to **right temple**.
- Connect blue electrode to +2 pin, stick just above **right eyebrow**, slightly to the left.
- Connect other blue electrode to +3 pin, stick just above **left eyebrow**, slightly to the right.
- Connect other red electrode to +4 pin, stick just above and to the right of your **lips**, not below nose.

See the demonstration video for visual reference.

# How to record data
The OpenBCI GUI uses a specific port, 12345. The sendMarker.py code should use this port, but if something isn't working, check this. Or check whether the port the GUI is using has been changed.
- After electrodes are set up, turn on the bluetooth
- Open the OpenBCI GUI, choose ganglion, native bluetooth, start streaming
- Run sendMarker.py
- Start streaming
- When you do an event, press the corresponding button in the script. The markers won't do anything unless streaming
- Data will be recorded in Documents

# How to classify live
- Ensure the model has been trained and saved. If it hasn't, run Main/preProcessing.py
- In the OpenBCI GUI, turn on networking over LSL
- Run Main/main.py
- Start streaming
- main.py will classify data.
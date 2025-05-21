For the 17.8% of people in England and Wales who are disabled, using conventional means of controlling technology such as keyboards, mice and touch screens may be difficult or even impossible. Accommodations in the form of alternative means of control exist, but they often come with caveats. This calls for a new accommodation for these individuals, particularly those with motor impairments.

This study explores the use of biosignals, specifically EEG, EOG and EMG, as an alternative means of controlling technology. Signals from the brain, eyes and facial muscles are captured and filtered, and useful time and frequency domain features are extracted from the raw data. These features are used to train a random forest machine learning model. The resulting classifier is used to predict a user’s actions in real time.

The model is designed to identify when the user is looking up, down, left and right, clenching their jaw, and generating alpha waves. The system’s ability to predict the user’s actions is demonstrated through the scenario of controlling the playback of music; the person’s actions are mapped to controls.

5-Fold cross-validation showed that the ML model has an average accuracy of 95.71%. The system represents a promising proof-of-concept for biosignal-driven assistive interfaces, though further validation on a broader user base is required.

A demonstration of the system in use can be found here: https://www.youtube.com/watch?v=WC70dQBcvec 

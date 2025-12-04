# LSTM_Autoencoder_Fall_Detection_Analysis
Deep Learning project using an LSTM Autoencoder for Fall Detection in sensor data. Optimized architecture (Deep LSTM, Tanh, LR=0.00001) to stabilize Nan errors. Final analysis yielded a low Recall of 2.75%, demonstrating the limitation of the unsupervised Autoencoder approach for high-noise time-series anomaly detection.

Project Overview

This project focuses on applying Deep Learning techniques to address the challenge of Anomaly Detection in human activity sensor data, specifically targeting Fall Detection. The primary goal was to train a Recurrent Neural Network (RNN) architecture, the LSTM Autoencoder, to distinguish between normal human movement and highly critical fall events. The entire project was developed using open-source data from Kaggle.com.

Technical Stack and Libraries

The entire workflow, from data preprocessing to model evaluation, was implemented in Python, utilizing the following core libraries:

Deep Learning Framework: TensorFlow and Keras (for building and training the Autoencoder architecture).

Data Manipulation: Pandas and NumPy (for sequence creation, data splitting, and numerical array operations).

Evaluation: Scikit-learn (sklearn) (for calculating performance metrics like Confusion Matrix, Recall, and F1 Score).

Visualization: Matplotlib (for plotting the training loss and analyzing results).

Methodology and Deep Learning Architecture

The project employed an unsupervised Machine Learning approach, where the model was exclusively trained on normal human activity data to learn its underlying patterns. The core of the methodology is based on Reconstruction Error: high reconstruction error signifies an anomaly (a fall).

1. Data Preprocessing and Sequence Generation

Goal: To prepare the raw, continuous time-series data for the sequential nature of the LSTM model.

Process: The raw sensor readings (accelerometer, gyroscope, etc., totaling 7 features) were standardized (scaling) and then segmented into time windows. We iteratively increased the window size from WS=50 to WS=100 to provide the model with a richer temporal context.

Result: The final dataset consisted of numerous sequences of length 100, each having 7 features.

2. Model Architecture: Deep LSTM Autoencoder

The final stable architecture was a Deep LSTM Autoencoder designed for sequence-to-sequence reconstruction:

Layer Type	Units	Activation	Purpose
Encoder	128 → 64 (LSTM)	Tanh	Compresses the input sequence into a compact latent representation.
Bottleneck	RepeatVector	N/A	Repeats the latent vector 100 times (sequence length).
Decoder	64 → 128 (LSTM)	Tanh	Reconstructs the original sequence from the latent vector.
Output	TimeDistributed(Dense(7))	N/A	Maps the sequence back to the original 7 features.

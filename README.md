# LSTM_Autoencoder_Fall_Detection_Analysis
Deep Learning project using an LSTM Autoencoder for Fall Detection in sensor data. Optimized architecture (Deep LSTM, Tanh, LR=0.00001) to stabilize Nan errors. Final analysis yielded a low Recall of 2.75%, demonstrating the limitation of the unsupervised Autoencoder approach for high-noise time-series anomaly detection.

## 📝 README.md: LSTM Autoencoder for Fall Detection Analysis

### 🎯 Project Overview

This project focuses on applying **Deep Learning** techniques to address the challenge of **Anomaly Detection** in human activity sensor data, specifically targeting **Fall Detection**. The primary goal was to train a **Recurrent Neural Network (RNN)** architecture, the **LSTM Autoencoder**, to distinguish between normal human movement and highly critical fall events. The entire project was developed using open-source data from **Kaggle.com**.

---

### 🛠️ Technical Stack and Libraries

The entire workflow, from data preprocessing to model evaluation, was implemented in **Python**, utilizing the following core libraries:

* **Deep Learning Framework:** **TensorFlow** and **Keras** (for building and training the Autoencoder architecture).
* **Data Manipulation:** **Pandas** and **NumPy** (for sequence creation, data splitting, and numerical array operations).
* **Evaluation:** **Scikit-learn (sklearn)** (for calculating performance metrics like Confusion Matrix, Recall, and F1 Score).
* **Visualization:** **Matplotlib** (for plotting the training loss and analyzing results).

---

###  Methodology and Deep Learning Architecture

The project employed an **unsupervised Machine Learning** approach, where the model was exclusively trained on **normal human activity data** to learn its underlying patterns. The core of the methodology is based on Reconstruction Error: high reconstruction error signifies an anomaly (a fall). 

#### **1. Data Preprocessing and Sequence Generation**

* **Goal:** To prepare the raw, continuous time-series data for the sequential nature of the LSTM model.
* **Process:** The raw sensor readings (accelerometer, gyroscope, etc., totaling **7 features**) were **standardized** (scaling) and then segmented into time windows. The final sequences had a length of $\text{100}$.

#### **2. Model Architecture: Deep LSTM Autoencoder**

The final stable architecture was a **Deep LSTM Autoencoder** designed for sequence-to-sequence reconstruction, consisting of a two-layer **Encoder** and a two-layer **Decoder** separated by a **RepeatVector** bottleneck.

---

### Critical Challenges and Solutions

The project required extensive hyperparameter tuning and architecture modification to achieve training stability:

* **Initial Challenge (Nan Loss / Exploding Gradients):** The early model versions suffered from severe instability, primarily due to the high learning rate and the use of the ReLU activation function in deep recurrent layers.
* **Solution:** Stability was finally achieved by making two critical changes:
    1.  Switching the activation function from ReLU to **Tanh**.
    2.  Setting the **Ultra-Low Learning Rate** to **$\mathbf{0.00001}$** for the Adam optimizer.
* **Performance Tuning:** The model was upgraded to a **Deep (Two-Layer) LSTM Autoencoder** architecture, and the **Window Size was increased to $\text{100}$**, to improve its ability to capture subtle temporal patterns.

---

###  Final Conclusion and Results

Despite achieving high stability (zero Nan errors) through comprehensive optimization, the model's final performance remained critically low, pointing to a fundamental limitation of the chosen approach for this specific dataset.

* **New Anomaly Threshold:** The final threshold (95th Percentile of Normal Loss) was $\mathbf{0.110690}$.
* **Recall (Sensitivity) Score:** The model achieved a low $\text{Recall}$ of **$\mathbf{0.0275}$ ($\approx 2.75\%$)**.
* **Conclusion:** The score demonstrates that the **LSTM Autoencoder is an unsuitable unsupervised method** for reliably distinguishing the fall event from high-noise, complex time-series data. The reconstruction error distributions for normal activities and fall events overlapped too heavily for the simple thresholding method to be effective. This project serves as a valuable case study highlighting the limitations of the Autoencoder approach for specific time-series anomaly detection tasks.

 

# Signal-Processing-and-Isolation-Forest-for-System-Load-Anomaly-Detection
The code implements a hybrid anomaly detection pipeline that applies a Moving Average (MA) filter to detrend non-stationary CPU utilisation time-series data, followed by Isolation Forest (IF) for unsupervised anomaly detection. A baseline Isolation Forest model on raw data and a rolling mean + 3σ statistical thresholding method are also implemented for comparative evaluation.

## The implemented pipeline evaluates:
* Baseline Isolation Forest on raw signal
* Hybrid Moving Average + Isolation Forest (MA + IF)
* Rolling mean + 3σ statistical baseline

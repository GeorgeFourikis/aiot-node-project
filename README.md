# AIoT Node Project - Fan Anomaly Detection

This project detects whether a small USB computer fan is running normally or with an artificial anomaly.

The anomaly is created by adding a small weight to one fan blade. A second identical fan is kept unmodified and is used as the normal baseline.

The Arduino Nano 33 BLE Sense Rev2 records audio using its onboard microphone. The collected audio data is used to train a small machine learning model. The model is then converted to TensorFlow Lite and deployed back to the Nano for live inference.

---

## Goal

Classify fan audio into two classes:
- **normal**
- **anomaly**


---

## Hardware
- Arduino Nano 33 BLE Sense Rev2
- 2 small USB computer fans
- Laptop for data collection and training
- USB cable

Fan setup:
- Normal fan: Unmodified fan from the factory
- Anomaly fan: Fan with a small weight added to one blade

Arduino setup:
- Board: Arduino Nano 33 BLE Sense Rev2
- Library: ArduTFLite 1.0.2
- PDM microphone library

---

## Project Structure

```text
.
├── requirements.txt
│
├── nano_code/
│   ├── nano_code.ino
│   └── fan_anomaly_inference/
│       ├── fan_anomaly_inference.ino
│       ├── fan_anomaly_model.h
│       └── feature_scaler.h
│
└── python_code/
    ├── data/
    │   ├── normal/
    │   │   ├── normal_001.csv
    │   │   └── ...
    │   └── anomaly/
    │       ├── anomaly_001.csv
    │       └── ...
    │
    ├── record_samples.py
    ├── train_baseline.py
    ├── train_tflite_features.py
    └── train_tflite_deployable.py
```

---

## Dataset

The dataset contains microphone recordings from the two fans.

Each recording is:
- 1 second long
- 16000 audio samples
- 16000 Hz sample rate


Current dataset:
- 50 normal recordings
- 50 anomaly recordings


The recordings are stored as CSV files:

- `python_code/data/normal/`
- `python_code/data/anomaly/`

---

## Python Setup

Go to the Python folder:

```bash
cd python_code
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r ..\requirements.txt
```

---

## Record New Samples

First, upload the data collection sketch to the Arduino Nano:

`nano_code/nano_code.ino`

Before recording, check `record_samples.py`.
The serial port is currently set to `COM7`, but this can change depending on the computer, USB port, and Arduino setup.
Also change `LABEL` to `normal` or `anomaly` depending on which fan is being recorded.

Then run:

```bash
python record_samples.py
```

The script records 1 second of audio from the Nano microphone and saves it as a CSV file.

Normal fan recordings go here:
`data/normal/`

Anomaly fan recordings go here:
`data/anomaly/`

---

## Train Baseline Model

Run:

```bash
python train_baseline.py
```

This trains a Random Forest model.

This model is only used to check whether the data is useful and separable.

Current baseline result:
Accuracy: 92%

---

## Train Deployable TinyML Model

Run:

```bash
python train_tflite_deployable.py
```

This trains a small neural network using simple audio features that can also be calculated on the Arduino.

The model uses features such as:
- RMS
- standard deviation
- peak-to-peak value
- zero crossings
- chunk RMS values
- chunk zero crossing values


The script creates the files needed by the Arduino sketch:
`fan_anomaly_model.h`
`feature_scaler.h`

If the model is retrained, copy the generated header files into:
`nano_code/fan_anomaly_inference/`

Current deployable model result:
Accuracy: 80%
Anomaly recall: 92%
TFLite model size: about 7 KB


---

## Arduino Deployment

Open this sketch in Arduino IDE:

`nano_code/fan_anomaly_inference/fan_anomaly_inference.ino`

Make sure these files are in the same folder:

`fan_anomaly_inference.ino`
`fan_anomaly_model.h`
`feature_scaler.h`

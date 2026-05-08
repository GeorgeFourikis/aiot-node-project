from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


DATA_DIR = Path("data")
MODEL_DIR = Path("model")
ARDUINO_DIR = Path("arduino")

MODEL_DIR.mkdir(exist_ok=True)
ARDUINO_DIR.mkdir(exist_ok=True)

SAMPLE_RATE = 16000


def load_csv_signal(path: Path) -> np.ndarray:
    values = np.loadtxt(path, delimiter=",", dtype=np.float32)

    if values.ndim > 1:
        values = values.flatten()

    return values


def extract_features(signal: np.ndarray) -> dict:
    signal = signal.astype(np.float32)

    # Remove DC offset
    signal = signal - np.mean(signal)

    rms = np.sqrt(np.mean(signal ** 2))
    std = np.std(signal)
    peak_to_peak = np.max(signal) - np.min(signal)
    zero_crossings = np.mean(np.abs(np.diff(np.signbit(signal))))

    fft = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / SAMPLE_RATE)

    fft_sum = np.sum(fft) + 1e-9

    dominant_freq = freqs[np.argmax(fft)]
    spectral_centroid = np.sum(freqs * fft) / fft_sum

    def band_energy(low, high):
        mask = (freqs >= low) & (freqs < high)
        return np.sum(fft[mask]) / fft_sum

    return {
        "rms": rms,
        "std": std,
        "peak_to_peak": peak_to_peak,
        "zero_crossings": zero_crossings,
        "dominant_freq": dominant_freq,
        "spectral_centroid": spectral_centroid,
        "energy_0_500": band_energy(0, 500),
        "energy_500_1000": band_energy(500, 1000),
        "energy_1000_2000": band_energy(1000, 2000),
        "energy_2000_4000": band_energy(2000, 4000),
        "energy_4000_8000": band_energy(4000, 8000),
    }


rows = []

for label_name, label_id in [("normal", 0), ("anomaly", 1)]:
    folder = DATA_DIR / label_name

    if not folder.exists():
        raise RuntimeError(f"Missing folder: {folder}")

    for csv_path in folder.glob("*.csv"):
        signal = load_csv_signal(csv_path)
        features = extract_features(signal)

        rows.append({
            "file": str(csv_path),
            "label_name": label_name,
            "label": label_id,
            "sample_count": len(signal),
            **features,
        })


df = pd.DataFrame(rows)

print("\nLoaded recordings:")
print(df[["file", "label_name", "sample_count"]])

print("\nCounts:")
print(df["label_name"].value_counts())

feature_columns = [
    "rms",
    "std",
    "peak_to_peak",
    "zero_crossings",
    "dominant_freq",
    "spectral_centroid",
    "energy_0_500",
    "energy_500_1000",
    "energy_1000_2000",
    "energy_2000_4000",
    "energy_4000_8000",
]

X = df[feature_columns].values.astype(np.float32)
y = df["label"].values.astype(np.int64)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
X_test_scaled = scaler.transform(X_test).astype(np.float32)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(len(feature_columns),)),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(2, activation="softmax"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(
    X_train_scaled,
    y_train,
    epochs=80,
    batch_size=8,
    validation_split=0.2,
    verbose=1,
)

loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)

pred_probs = model.predict(X_test_scaled)
predictions = np.argmax(pred_probs, axis=1)

print("\nTest accuracy:")
print(accuracy)

print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["normal", "anomaly"]
))

# Save normal Keras model
model.save(MODEL_DIR / "fan_anomaly_model.keras")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

tflite_path = MODEL_DIR / "fan_anomaly_model.tflite"
tflite_path.write_bytes(tflite_model)

print(f"\nSaved TFLite model: {tflite_path}")
print(f"TFLite model size: {len(tflite_model)} bytes")

# Save feature metadata
metadata = pd.DataFrame({
    "feature": feature_columns,
    "mean": scaler.mean_,
    "scale": scaler.scale_,
})

metadata_path = MODEL_DIR / "feature_scaler.csv"
metadata.to_csv(metadata_path, index=False)

print(f"Saved scaler metadata: {metadata_path}")

# Generate model header
header_path = ARDUINO_DIR / "fan_anomaly_model.h"

with open(header_path, "w") as f:
    f.write("#ifndef FAN_ANOMALY_MODEL_H\n")
    f.write("#define FAN_ANOMALY_MODEL_H\n\n")

    f.write("const unsigned char fan_anomaly_model[] = {\n")

    for i, byte in enumerate(tflite_model):
        if i % 12 == 0:
            f.write("  ")
        f.write(f"0x{byte:02x}")
        if i < len(tflite_model) - 1:
            f.write(", ")
        if i % 12 == 11:
            f.write("\n")

    f.write("\n};\n\n")
    f.write(f"const int fan_anomaly_model_len = {len(tflite_model)};\n\n")
    f.write("#endif\n")

print(f"Saved Arduino model header: {header_path}")

# Generate scaler header
scaler_header_path = ARDUINO_DIR / "feature_scaler.h"

with open(scaler_header_path, "w") as f:
    f.write("#ifndef FEATURE_SCALER_H\n")
    f.write("#define FEATURE_SCALER_H\n\n")

    f.write(f"const int FEATURE_COUNT = {len(feature_columns)};\n\n")

    f.write("const float feature_means[] = {\n  ")
    f.write(", ".join([f"{v:.8f}f" for v in scaler.mean_]))
    f.write("\n};\n\n")

    f.write("const float feature_scales[] = {\n  ")
    f.write(", ".join([f"{v:.8f}f" for v in scaler.scale_]))
    f.write("\n};\n\n")

    f.write("#endif\n")

print(f"Saved Arduino scaler header: {scaler_header_path}")

# Save features for report/debugging
df.to_csv(MODEL_DIR / "recording_features.csv", index=False)

print("\nDone.")
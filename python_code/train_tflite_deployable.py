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

CHUNK_COUNT = 16


def load_csv_signal(path: Path) -> np.ndarray:
    values = np.loadtxt(path, delimiter=",", dtype=np.float32)

    if values.ndim > 1:
        values = values.flatten()

    return values


def zero_crossing_rate(signal: np.ndarray) -> float:
    return float(np.mean(np.abs(np.diff(np.signbit(signal)))))


def extract_features(signal: np.ndarray) -> dict:
    signal = signal.astype(np.float32)
    signal = signal - np.mean(signal)

    features = {}

    features["rms"] = float(np.sqrt(np.mean(signal ** 2)))
    features["std"] = float(np.std(signal))
    features["peak_to_peak"] = float(np.max(signal) - np.min(signal))
    features["zero_crossings"] = zero_crossing_rate(signal)

    chunks = np.array_split(signal, CHUNK_COUNT)

    for i, chunk in enumerate(chunks):
        features[f"chunk_{i}_rms"] = float(np.sqrt(np.mean(chunk ** 2)))
        features[f"chunk_{i}_zero_crossings"] = zero_crossing_rate(chunk)

    return features


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
]

for i in range(CHUNK_COUNT):
    feature_columns.append(f"chunk_{i}_rms")
    feature_columns.append(f"chunk_{i}_zero_crossings")

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
    tf.keras.layers.Dense(24, activation="relu"),
    tf.keras.layers.Dense(12, activation="relu"),
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
    epochs=100,
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

model.save(MODEL_DIR / "fan_anomaly_deployable_model.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

tflite_path = MODEL_DIR / "fan_anomaly_deployable_model.tflite"
tflite_path.write_bytes(tflite_model)

print(f"\nSaved TFLite model: {tflite_path}")
print(f"TFLite model size: {len(tflite_model)} bytes")

metadata = pd.DataFrame({
    "feature": feature_columns,
    "mean": scaler.mean_,
    "scale": scaler.scale_,
})

metadata_path = MODEL_DIR / "deployable_feature_scaler.csv"
metadata.to_csv(metadata_path, index=False)

print(f"Saved scaler metadata: {metadata_path}")

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

scaler_header_path = ARDUINO_DIR / "feature_scaler.h"

with open(scaler_header_path, "w") as f:
    f.write("#ifndef FEATURE_SCALER_H\n")
    f.write("#define FEATURE_SCALER_H\n\n")

    f.write(f"const int FEATURE_COUNT = {len(feature_columns)};\n")
    f.write(f"const int CHUNK_COUNT = {CHUNK_COUNT};\n\n")

    f.write("const float feature_means[] = {\n  ")
    f.write(", ".join([f"{v:.8f}f" for v in scaler.mean_]))
    f.write("\n};\n\n")

    f.write("const float feature_scales[] = {\n  ")
    f.write(", ".join([f"{v:.8f}f" for v in scaler.scale_]))
    f.write("\n};\n\n")

    f.write("#endif\n")

print(f"Saved Arduino scaler header: {scaler_header_path}")

df.to_csv(MODEL_DIR / "deployable_recording_features.csv", index=False)

print("\nDone.")
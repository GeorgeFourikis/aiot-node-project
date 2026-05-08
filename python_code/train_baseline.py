from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib


DATA_DIR = Path("data")
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

for label in ["normal", "anomaly"]:
    folder = DATA_DIR / label

    if not folder.exists():
        raise RuntimeError(f"Missing folder: {folder}")

    for csv_path in folder.glob("*.csv"):
        signal = load_csv_signal(csv_path)
        features = extract_features(signal)

        row = {
            "file": str(csv_path),
            "label": label,
            "sample_count": len(signal),
            **features,
        }

        rows.append(row)


df = pd.DataFrame(rows)

print("\nLoaded recordings:")
print(df[["file", "label", "sample_count"]])

print("\nCounts:")
print(df["label"].value_counts())

if len(df) < 20:
    raise RuntimeError("Too few recordings. Collect more samples.")

if df["sample_count"].nunique() != 1:
    print("\nWARNING: Not all recordings have the same length:")
    print(df["sample_count"].value_counts())


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

X = df[feature_columns]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=4,
        )),
    ]
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nAccuracy:")
print(accuracy_score(y_test, predictions))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions, labels=["normal", "anomaly"]))

print("\nClassification report:")
print(classification_report(y_test, predictions))

joblib.dump(model, "fan_baseline_model.joblib")
df.to_csv("recording_features.csv", index=False)

print("\nSaved:")
print("fan_baseline_model.joblib")
print("recording_features.csv")
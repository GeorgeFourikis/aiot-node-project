import serial
import serial.tools.list_ports
import time
import numpy as np
from pathlib import Path

PORT = "COM7"
BAUD = 921600

# Use "anomaly" when recording the faulty fan and "normal" when the healthy fan is used.
LABEL = "normal"  
COUNT = 50



output_dir = Path("data") / LABEL
output_dir.mkdir(parents=True, exist_ok=True)


def show_ports():
    print("Available serial ports:")
    for port in serial.tools.list_ports.comports():
        print(f"  {port.device} - {port.description}")


def record_one(ser, index: int):
    print("Sending record command...")

    ser.reset_input_buffer()
    time.sleep(0.2)

    ser.write(b"r")
    ser.flush()

    values_line = None

    while True:
        line = ser.readline().decode(errors="ignore").strip()

        if not line:
            continue

        print(f"Nano: {line[:80]}")

        if line == "ERROR_PDM_BEGIN_FAILED":
            raise RuntimeError("Nano could not start the PDM microphone.")

        if line == "BEGIN_SAMPLE":
            values_line = ser.readline().decode(errors="ignore").strip()
            print("Received sample data.")

        elif line == "END_SAMPLE":
            break

    if values_line is None:
        raise RuntimeError("No sample data received.")

    values = np.array(
        [int(x) for x in values_line.split(",") if x.strip()],
        dtype=np.int16
    )

    filename = output_dir / f"{LABEL}_{index:03d}.csv"
    np.savetxt(filename, values, fmt="%d", delimiter=",")

    print(f"Saved {filename} with {len(values)} values")


show_ports()

print(f"Opening {PORT} at {BAUD} baud...")

ser = serial.Serial(PORT, BAUD, timeout=10)
time.sleep(2)

try:
    for i in range(1, COUNT + 1):
        input(f"Prepare {LABEL} sample {i}. Press Enter to record...")
        record_one(ser, i)

finally:
    ser.close()
    print("Serial port closed.")
#include <PDM.h>
#include <ArduTFLite.h>

#include "fan_anomaly_model.h"
#include "feature_scaler.h"

const int SAMPLE_RATE = 16000;
const int SAMPLE_COUNT = 16000;
const int MODEL_INPUT_SIZE = FEATURE_COUNT;

short pdmBuffer[512];
int16_t recording[SAMPLE_COUNT];

volatile int recordIndex = 0;
volatile bool isRecording = false;
volatile bool sampleReady = false;

unsigned long recordingStartTime = 0;

float features[FEATURE_COUNT];

constexpr int tensorArenaSize = 32 * 1024;
alignas(16) uint8_t tensorArena[tensorArenaSize];

void onPDMdata() {
  int bytesAvailable = PDM.available();

  if (bytesAvailable <= 0) {
    return;
  }

  PDM.read(pdmBuffer, bytesAvailable);

  int samplesRead = bytesAvailable / 2;

  if (isRecording) {
    for (int i = 0; i < samplesRead; i++) {
      if (recordIndex < SAMPLE_COUNT) {
        recording[recordIndex] = pdmBuffer[i];
        recordIndex++;
      }

      if (recordIndex >= SAMPLE_COUNT) {
        isRecording = false;
        sampleReady = true;
        break;
      }
    }
  }
}

float computeMean(const int16_t* data, int start, int count) {
  float sum = 0.0f;

  for (int i = 0; i < count; i++) {
    sum += data[start + i];
  }

  return sum / count;
}

float computeRmsCentered(const int16_t* data, int start, int count, float mean) {
  float sumSquares = 0.0f;

  for (int i = 0; i < count; i++) {
    float centered = data[start + i] - mean;
    sumSquares += centered * centered;
  }

  return sqrtf(sumSquares / count);
}

float computePeakToPeak(const int16_t* data, int start, int count) {
  int16_t minValue = data[start];
  int16_t maxValue = data[start];

  for (int i = 1; i < count; i++) {
    int16_t value = data[start + i];

    if (value < minValue) {
      minValue = value;
    }

    if (value > maxValue) {
      maxValue = value;
    }
  }

  return float(maxValue - minValue);
}

float computeZeroCrossingsCentered(const int16_t* data, int start, int count, float mean) {
  int crossings = 0;

  bool previousSign = (data[start] - mean) < 0.0f;

  for (int i = 1; i < count; i++) {
    bool currentSign = (data[start + i] - mean) < 0.0f;

    if (currentSign != previousSign) {
      crossings++;
    }

    previousSign = currentSign;
  }

  return float(crossings) / float(count - 1);
}

void extractFeatures() {
  const int chunkCount = 16;
  const int chunkSize = SAMPLE_COUNT / chunkCount;

  float globalMean = computeMean(recording, 0, SAMPLE_COUNT);

  int featureIndex = 0;

  features[featureIndex++] = computeRmsCentered(recording, 0, SAMPLE_COUNT, globalMean);
  features[featureIndex++] = computeRmsCentered(recording, 0, SAMPLE_COUNT, globalMean);
  features[featureIndex++] = computePeakToPeak(recording, 0, SAMPLE_COUNT);
  features[featureIndex++] = computeZeroCrossingsCentered(recording, 0, SAMPLE_COUNT, globalMean);

  for (int chunk = 0; chunk < chunkCount; chunk++) {
    int start = chunk * chunkSize;
    float chunkMean = computeMean(recording, start, chunkSize);

    features[featureIndex++] = computeRmsCentered(recording, start, chunkSize, chunkMean);
    features[featureIndex++] = computeZeroCrossingsCentered(recording, start, chunkSize, chunkMean);
  }

  for (int i = 0; i < MODEL_INPUT_SIZE; i++) {
    features[i] = (features[i] - feature_means[i]) / feature_scales[i];
  }
}

void runInference() {
  extractFeatures();

  for (int i = 0; i < MODEL_INPUT_SIZE; i++) {
    modelSetInput(features[i], i);
  }

  if (!modelRunInference()) {
    Serial.println("ERROR: modelRunInference failed");
    return;
  }

  float normalScore = modelGetOutput(0);
  float anomalyScore = modelGetOutput(1);

  Serial.print("normal=");
  Serial.print(normalScore, 4);

  Serial.print(" anomaly=");
  Serial.print(anomalyScore, 4);

  Serial.print(" prediction=");

  if (anomalyScore > normalScore) {
    Serial.println("ANOMALY");
    digitalWrite(LED_BUILTIN, HIGH);
  } else {
    Serial.println("NORMAL");
    digitalWrite(LED_BUILTIN, LOW);
  }
}

void startRecording() {
  recordIndex = 0;
  sampleReady = false;
  isRecording = true;
  recordingStartTime = millis();

  Serial.println("Recording 1 second...");
}

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);

  Serial.begin(115200);
  while (!Serial);

  Serial.println("Fan anomaly inference starting...");

  if (!modelInit(fan_anomaly_model, tensorArena, tensorArenaSize)) {
    Serial.println("ERROR: modelInit failed");
    while (true);
  }

  Serial.println("Model initialized.");

  /*
    Important:
    Register the microphone callback BEFORE starting PDM.
  */
  PDM.onReceive(onPDMdata);

  if (!PDM.begin(1, SAMPLE_RATE)) {
    Serial.println("ERROR: Failed to start PDM microphone");
    while (true);
  }

  Serial.println("PDM microphone started.");
  Serial.println("Ready.");
  Serial.println("Send r to record and classify.");
}

void loop() {
  if (Serial.available()) {
    char command = Serial.read();

    if (command == 'r' && !isRecording) {
      startRecording();
    }
  }

  if (isRecording && millis() - recordingStartTime > 3000) {
    isRecording = false;

    Serial.print("ERROR: Recording timeout. Samples captured: ");
    Serial.println(recordIndex);

    Serial.println("Send r to try again.");
  }

  if (sampleReady) {
    sampleReady = false;

    Serial.print("Sample ready. Samples captured: ");
    Serial.println(recordIndex);

    Serial.println("Running inference...");
    runInference();

    Serial.println("Send r to run again.");
  }
}
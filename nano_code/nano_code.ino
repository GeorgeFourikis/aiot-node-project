#include <PDM.h>

const int SAMPLE_RATE = 16000;
const int SAMPLE_COUNT = 16000;

short pdmBuffer[512];

int16_t recording[SAMPLE_COUNT];
volatile int recordIndex = 0;
volatile bool isRecording = false;

bool sampleReady = false;

unsigned long lastReadyPrint = 0;

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

void setup() {
  Serial.begin(921600);

  // Do NOT block forever waiting for Serial.
  delay(1500);

  PDM.onReceive(onPDMdata);

  if (!PDM.begin(1, SAMPLE_RATE)) {
    while (true) {
      Serial.println("ERROR_PDM_BEGIN_FAILED");
      delay(1000);
    }
  }

  Serial.println("READY");
}

void loop() {
  if (!isRecording && !sampleReady && millis() - lastReadyPrint > 2000) {
    Serial.println("READY");
    lastReadyPrint = millis();
  }

  if (Serial.available()) {
    char command = Serial.read();

    if (command == 'r' && !isRecording && !sampleReady) {
      recordIndex = 0;
      sampleReady = false;
      isRecording = true;
      Serial.println("RECORDING");
    }
  }

  if (sampleReady) {
    sampleReady = false;

    Serial.println("BEGIN_SAMPLE");

    for (int i = 0; i < SAMPLE_COUNT; i++) {
      Serial.print(recording[i]);

      if (i < SAMPLE_COUNT - 1) {
        Serial.print(",");
      }
    }

    Serial.println();
    Serial.println("END_SAMPLE");
    Serial.println("READY");
  }
}
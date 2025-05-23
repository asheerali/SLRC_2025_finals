// TCS3200 (HW-067) required pin definitions
#define S0 44  // Frequency scaling
#define S2 47  // Filter select
#define S3 46  // Filter select
#define OUT 49 // Frequency output


unsigned long redFreq, greenFreq, blueFreq;

void setup() {
  pinMode(S0, OUTPUT);
  pinMode(S2, OUTPUT);
  pinMode(S3, OUTPUT);
  pinMode(OUT, INPUT);

  digitalWrite(S0, LOW);  // 2% scaling for wider range

  Serial.begin(9600);
  Serial.println("HW-067 White/Yellow Detection");
}

void readColors() {
  unsigned long redSum = 0, greenSum = 0, blueSum = 0;
  const int samples = 5;

  for (int i = 0; i < samples; i++) {
    digitalWrite(S2, LOW);
    digitalWrite(S3, LOW);
    redSum += pulseIn(OUT, LOW);

    digitalWrite(S2, HIGH);
    digitalWrite(S3, HIGH);
    greenSum += pulseIn(OUT, LOW);

    digitalWrite(S2, LOW);
    digitalWrite(S3, HIGH);
    blueSum += pulseIn(OUT, LOW);

    delay(10);
  }

  redFreq = redSum / samples;
  greenFreq = greenSum / samples;
  blueFreq = blueSum / samples;
}

void loop() {
  readColors();

  Serial.print("Red: ");
  Serial.print(redFreq);
  Serial.print(" Green: ");
  Serial.print(greenFreq);
  Serial.print(" Blue: ");
  Serial.println(blueFreq);

  // Adjusted thresholds (example for 2% scaling)
  if (redFreq < 350 && greenFreq < 350 && blueFreq < 350) {
    Serial.println("Detected: White");
  } else if (redFreq < 400 && greenFreq < 350 && blueFreq > 500) {
    Serial.println("Detected: Yellow");
  } else {
    Serial.println("Unknown Color");
  }

  delay(1000);
}
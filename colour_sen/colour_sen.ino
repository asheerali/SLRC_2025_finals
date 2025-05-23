#define S0 44
#define S2 47
#define S3 46
#define OUT 49

unsigned long redFreq, greenFreq, blueFreq;

void setup() {
  pinMode(S0, OUTPUT);
  pinMode(S2, OUTPUT);
  pinMode(S3, OUTPUT);
  pinMode(OUT, INPUT);

  digitalWrite(S0, HIGH);  // 20% scaling (S1 assumed LOW)

  Serial.begin(9600);
  Serial.println("HW-067 Ready for White/Yellow Detection");
}

void readColors() {
  digitalWrite(S2, LOW);
  digitalWrite(S3, LOW);
  redFreq = pulseIn(OUT, LOW);

  digitalWrite(S2, HIGH);
  digitalWrite(S3, HIGH);
  greenFreq = pulseIn(OUT, LOW);

  digitalWrite(S2, LOW);
  digitalWrite(S3, HIGH);
  blueFreq = pulseIn(OUT, LOW);
}

void loop() {
  readColors();

  Serial.print("Red: ");
  Serial.print(redFreq);
  Serial.print(" Green: ");
  Serial.print(greenFreq);
  Serial.print(" Blue: ");
  Serial.println(blueFreq);


  if (redFreq>15 && blueFreq>15){
    Serial.println("Yellow");

  }
  else {
    Serial.println("White");

  }

  // Calibrated detection
 

  delay(1000);
}
/*
 * single_servo_test.ino
 *
 * Minimal test: drives ONE servo directly from an ESP32 GPIO pin (no PCA9685).
 *
 * WARNING: Do NOT power the servo from the ESP32 5V pin!
 * MG996R servos draw up to 2.5A stall current, which will damage the ESP32.
 * Use a separate 5-6V power supply for the servo. Share GND only.
 *
 * Wiring:
 *   Servo signal (orange) -> ESP32 GPIO 13
 *   Servo VCC (red)       -> External 5-6V PSU (+)
 *   Servo GND (brown)     -> External PSU (-) AND ESP32 GND (common ground)
 *
 * On boot: sweeps the servo to verify it works, then listens for serial commands.
 *
 * Serial protocol (same as full firmware, but only joint index 0 is actuated):
 *   J:<a0>,<a1>,...,<a7>   -> drives servo to a0 degrees (ignores a1-a7)
 *   S:0:<angle>            -> drives servo to <angle> degrees
 *   H                      -> move to 0 degrees
 *   Q                      -> report current angle
 *   E:<0|1>                -> enable/disable
 *   A:<angle>              -> shortcut: set angle directly (degrees)
 *
 * Baud: 115200
 */

// ==================== CONFIG ====================

#define SERIAL_BAUD 115200
#define SERVO_PIN   13

// PWM config
#define PWM_FREQ       50    // 50 Hz standard servo
#define PWM_RESOLUTION 16    // 16-bit -> 0..65535

// Servo pulse range (microseconds) — adjust if your servo differs
#define SERVO_MIN_US 500
#define SERVO_MAX_US 2500

// Angle limits (degrees)
#define ANGLE_MIN -90.0
#define ANGLE_MAX 135.0

// ==================== GLOBALS ====================

float currentAngle = 0.0;
bool enabled = true;
String inputBuffer = "";

// ==================== HELPERS ====================

float mapFloat(float x, float inMin, float inMax, float outMin, float outMax) {
  return (x - inMin) * (outMax - outMin) / (inMax - inMin) + outMin;
}

void setServoAngle(float angle) {
  if (!enabled) return;

  // Clamp
  if (angle < ANGLE_MIN) angle = ANGLE_MIN;
  if (angle > ANGLE_MAX) angle = ANGLE_MAX;

  currentAngle = angle;

  // Angle -> pulse width (us) -> LEDC duty
  float pulseUs = mapFloat(angle, ANGLE_MIN, ANGLE_MAX, SERVO_MIN_US, SERVO_MAX_US);
  // 16-bit at 50 Hz: period = 20000us, full scale = 65536
  uint32_t duty = (uint32_t)(pulseUs / 20000.0 * 65536.0);

  ledcWrite(SERVO_PIN, duty);
}

// ==================== STARTUP SWEEP ====================

void sweepTest() {
  Serial.println("Sweep test: -45 -> +90 -> 0");

  // Go to -45
  setServoAngle(-45.0);
  delay(600);

  // Sweep to +90
  for (float a = -45.0; a <= 90.0; a += 2.0) {
    setServoAngle(a);
    delay(15);
  }
  delay(400);

  // Back to 0
  for (float a = 90.0; a >= 0.0; a -= 2.0) {
    setServoAngle(a);
    delay(15);
  }
  delay(200);

  Serial.println("Sweep done");
}

// ==================== COMMAND HANDLER ====================

void handleCommand(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) return;

  char type = cmd.charAt(0);

  switch (type) {
    case 'J': {
      // J:<a0>,<a1>,...  — use only a0
      if (cmd.length() < 3 || cmd.charAt(1) != ':') {
        Serial.println("ERR:1:Bad format");
        return;
      }
      String data = cmd.substring(2);
      int comma = data.indexOf(',');
      String first = (comma > 0) ? data.substring(0, comma) : data;
      float angle = first.toFloat();
      if (angle < ANGLE_MIN || angle > ANGLE_MAX) {
        Serial.println("ERR:2:Angle out of range");
        return;
      }
      setServoAngle(angle);
      Serial.println("OK");
      break;
    }

    case 'S': {
      // S:<joint>:<angle> — only accept joint 0
      if (cmd.length() < 5 || cmd.charAt(1) != ':') {
        Serial.println("ERR:1:Bad format");
        return;
      }
      String data = cmd.substring(2);
      int colon = data.indexOf(':');
      if (colon < 0) {
        Serial.println("ERR:1:Bad format");
        return;
      }
      int joint = data.substring(0, colon).toInt();
      if (joint != 0) {
        Serial.println("OK");  // Silently ignore other joints
        return;
      }
      float angle = data.substring(colon + 1).toFloat();
      if (angle < ANGLE_MIN || angle > ANGLE_MAX) {
        Serial.println("ERR:2:Angle out of range");
        return;
      }
      setServoAngle(angle);
      Serial.println("OK");
      break;
    }

    case 'H':
      setServoAngle(0.0);
      Serial.println("OK");
      break;

    case 'Q': {
      // Report: P:<a0>,0.0,0.0,0.0,0.0,0.0,0.0,0.0
      Serial.print("P:");
      Serial.print(currentAngle, 1);
      for (int i = 1; i < 8; i++) {
        Serial.print(",0.0");
      }
      Serial.println();
      break;
    }

    case 'E':
      if (cmd.length() >= 3) {
        enabled = (cmd.charAt(2) == '1');
        if (!enabled) {
          ledcWrite(SERVO_PIN, 0);  // Stop PWM signal
        }
      }
      Serial.println("OK");
      break;

    case 'A': {
      // A:<angle>  — shortcut for quick manual testing
      if (cmd.length() < 3 || cmd.charAt(1) != ':') {
        Serial.println("ERR:1:Bad format");
        return;
      }
      float angle = cmd.substring(2).toFloat();
      if (angle < ANGLE_MIN || angle > ANGLE_MAX) {
        Serial.println("ERR:2:Angle out of range");
        return;
      }
      setServoAngle(angle);
      Serial.print("OK angle=");
      Serial.println(angle, 1);
      break;
    }

    default:
      Serial.println("ERR:1:Unknown command");
  }
}

// ==================== SETUP & LOOP ====================

void setup() {
  Serial.begin(SERIAL_BAUD);

  // Configure LEDC PWM (ESP32 Arduino core 3.x API)
  ledcAttach(SERVO_PIN, PWM_FREQ, PWM_RESOLUTION);

  delay(100);

  sweepTest();

  Serial.println("READY");
  Serial.println("Commands: J:<angles> | S:0:<angle> | H | Q | E:<0|1> | A:<angle>");
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      handleCommand(inputBuffer);
      inputBuffer = "";
    } else {
      inputBuffer += c;
    }
  }
}

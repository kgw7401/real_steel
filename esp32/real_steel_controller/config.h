#pragma once

// ==================== Serial ====================
#define SERIAL_BAUD 115200
#define INPUT_BUFFER_SIZE 128

// ==================== I2C / PCA9685 ====================
#define I2C_SDA 21
#define I2C_SCL 22
#define PCA9685_ADDR 0x40
#define PWM_FREQ 50  // 50 Hz standard servo

// ==================== Servos ====================
#define NUM_SERVOS 8

// Joint order:
//   0 = L_SHOULDER_ROLL    4 = R_SHOULDER_ROLL
//   1 = L_SHOULDER_TILT    5 = R_SHOULDER_TILT
//   2 = L_SHOULDER_PAN     6 = R_SHOULDER_PAN
//   3 = L_ELBOW            7 = R_ELBOW

// PCA9685 channel mapping (servo index -> PWM channel)
// Change these if your wiring order differs.
const uint8_t SERVO_CHANNEL[NUM_SERVOS] = {0, 1, 2, 3, 4, 5, 6, 7};

// Servo pulse width limits (microseconds).
// CALIBRATE THESE per servo — defaults are conservative.
const int SERVO_MIN_US[NUM_SERVOS] = {500, 500, 500, 500, 500, 500, 500, 500};
const int SERVO_MAX_US[NUM_SERVOS] = {2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500};

// Angle limits (degrees) — must match motion_mapper joint limits.
//                              roll   tilt   pan   elbow  roll   tilt   pan   elbow
const float ANGLE_MIN[NUM_SERVOS] = {-20,  -90,  -90,    0,  -20,  -90,  -90,    0};
const float ANGLE_MAX[NUM_SERVOS] = {135,   90,   90,  135,  135,   90,   90,  135};

// Home positions (degrees)
const float HOME_ANGLES[NUM_SERVOS] = {0, 0, 0, 0, 0, 0, 0, 0};

// ==================== Safety ====================
// If no serial command received within this period, home and disable servos.
#define WATCHDOG_TIMEOUT_MS 2000

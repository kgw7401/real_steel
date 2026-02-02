#pragma once

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include "config.h"

// ==================== State ====================

static Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(PCA9685_ADDR);
static float currentAngles[NUM_SERVOS] = {0};
static bool servosEnabled = false;

// ==================== Helpers ====================

static float mapFloat(float x, float inMin, float inMax, float outMin, float outMax) {
    return (x - inMin) * (outMax - outMin) / (inMax - inMin) + outMin;
}

// ==================== Public API ====================

void servo_init() {
    Wire.begin(I2C_SDA, I2C_SCL);
    pwm.begin();
    pwm.setPWMFreq(PWM_FREQ);
    delay(10);
}

void servo_set_angle(int servo, float angle) {
    if (servo < 0 || servo >= NUM_SERVOS) return;
    if (!servosEnabled) return;

    // Clamp to limits
    if (angle < ANGLE_MIN[servo]) angle = ANGLE_MIN[servo];
    if (angle > ANGLE_MAX[servo]) angle = ANGLE_MAX[servo];

    currentAngles[servo] = angle;

    // Map angle to pulse width (microseconds)
    float pulseUs = mapFloat(angle, ANGLE_MIN[servo], ANGLE_MAX[servo],
                             SERVO_MIN_US[servo], SERVO_MAX_US[servo]);

    // Convert microseconds to PCA9685 tick (4096 ticks per 20ms cycle)
    uint16_t tick = (uint16_t)((pulseUs * 4096.0) / 20000.0);

    pwm.setPWM(SERVO_CHANNEL[servo], 0, tick);
}

void servo_home() {
    for (int i = 0; i < NUM_SERVOS; i++) {
        servo_set_angle(i, HOME_ANGLES[i]);
    }
}

void servo_enable(bool enabled) {
    servosEnabled = enabled;
    if (!enabled) {
        // Turn off all PWM outputs (servos go limp)
        for (int i = 0; i < NUM_SERVOS; i++) {
            pwm.setPWM(SERVO_CHANNEL[i], 0, 0);
        }
    }
}

bool servo_is_enabled() {
    return servosEnabled;
}

float servo_get_angle(int servo) {
    if (servo < 0 || servo >= NUM_SERVOS) return 0.0;
    return currentAngles[servo];
}

/*
 * real_steel_controller.ino
 *
 * Main firmware for Real Steel shadow boxing robot.
 * ESP32 + PCA9685 PWM driver controlling 8 MG996R servos.
 *
 * Serial protocol (115200 baud):
 *   J:<a0>,<a1>,...,<a7>  — Set all 8 joints (degrees)
 *   S:<joint>:<angle>     — Set single joint
 *   H                     — Home (all joints to 0)
 *   Q                     — Query current positions
 *   E:<0|1>               — Enable/disable servos
 *
 * Safety:
 *   - Servos start DISABLED (must send E:1 to enable)
 *   - Watchdog: homes and disables after 2s of no commands
 *   - All angles clamped to config limits before actuation
 *
 * Wiring:
 *   ESP32 GPIO21 (SDA) -> PCA9685 SDA
 *   ESP32 GPIO22 (SCL) -> PCA9685 SCL
 *   ESP32 3V3          -> PCA9685 VCC
 *   ESP32 GND          -> PCA9685 GND
 *   External 6V PSU    -> PCA9685 V+ (servo power)
 *   External PSU GND   -> PCA9685 GND (shared with ESP32 GND)
 *
 * Dependencies:
 *   - Adafruit PWM Servo Driver Library (install via Arduino Library Manager)
 */

#include "config.h"
#include "servo_controller.h"
#include "serial_handler.h"

// Shared with serial_handler.h for watchdog
unsigned long lastCommandTime = 0;
bool watchdogTriggered = false;

void setup() {
    serial_init();
    servo_init();

    // Servos start disabled for safety.
    // Python side sends E:0 -> H -> E:1 on connect.
    servo_enable(false);

    serial_send_ready();
    lastCommandTime = millis();
}

void loop() {
    serial_poll();

    // Watchdog: if no commands for WATCHDOG_TIMEOUT_MS, home and disable
    unsigned long elapsed = millis() - lastCommandTime;
    if (elapsed > WATCHDOG_TIMEOUT_MS) {
        if (servo_is_enabled()) {
            servo_home();
            servo_enable(false);
            Serial.println("WATCHDOG:servos disabled");
        }
        if (!watchdogTriggered) {
            watchdogTriggered = true;
        }
    } else {
        watchdogTriggered = false;
    }
}

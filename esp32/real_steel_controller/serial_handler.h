#pragma once

#include "config.h"
#include "servo_controller.h"

// ==================== State ====================

static char inputBuffer[INPUT_BUFFER_SIZE];
static int bufferPos = 0;

// Timestamp of last valid command (for watchdog)
extern unsigned long lastCommandTime;

// ==================== Response helpers ====================

static void serial_send_ok() {
    Serial.println("OK");
}

static void serial_send_error(int code, const char* msg) {
    Serial.print("ERR:");
    Serial.print(code);
    Serial.print(":");
    Serial.println(msg);
}

static void serial_send_positions() {
    Serial.print("P:");
    for (int i = 0; i < NUM_SERVOS; i++) {
        Serial.print(servo_get_angle(i), 1);
        if (i < NUM_SERVOS - 1) Serial.print(",");
    }
    Serial.println();
}

void serial_send_ready() {
    Serial.println("READY");
}

// ==================== Command handlers ====================

// J:<a0>,<a1>,<a2>,<a3>,<a4>,<a5>,<a6>,<a7>
static void handle_joint_command(const char* data) {
    float angles[NUM_SERVOS];
    int count = 0;

    // Parse comma-separated floats
    char buf[INPUT_BUFFER_SIZE];
    strncpy(buf, data, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';

    char* token = strtok(buf, ",");
    while (token != NULL && count < NUM_SERVOS) {
        angles[count] = atof(token);
        count++;
        token = strtok(NULL, ",");
    }

    if (count != NUM_SERVOS) {
        serial_send_error(1, "Expected 8 angles");
        return;
    }

    // Validate ranges
    for (int i = 0; i < NUM_SERVOS; i++) {
        if (angles[i] < ANGLE_MIN[i] || angles[i] > ANGLE_MAX[i]) {
            char msg[40];
            snprintf(msg, sizeof(msg), "Joint %d out of range", i);
            serial_send_error(2, msg);
            return;
        }
    }

    // Apply
    for (int i = 0; i < NUM_SERVOS; i++) {
        servo_set_angle(i, angles[i]);
    }
    serial_send_ok();
}

// S:<joint>:<angle>
static void handle_single_joint(const char* data) {
    // Find the colon separator
    const char* colon = strchr(data, ':');
    if (colon == NULL) {
        serial_send_error(1, "Bad format, expected S:<joint>:<angle>");
        return;
    }

    int joint = atoi(data);
    float angle = atof(colon + 1);

    if (joint < 0 || joint >= NUM_SERVOS) {
        serial_send_error(3, "Invalid joint index");
        return;
    }

    if (angle < ANGLE_MIN[joint] || angle > ANGLE_MAX[joint]) {
        serial_send_error(2, "Angle out of range");
        return;
    }

    servo_set_angle(joint, angle);
    serial_send_ok();
}

// E:<0|1>
static void handle_enable(const char* data) {
    if (data[0] == '1') {
        servo_enable(true);
    } else if (data[0] == '0') {
        servo_enable(false);
    } else {
        serial_send_error(1, "Expected E:0 or E:1");
        return;
    }
    serial_send_ok();
}

// ==================== Command dispatch ====================

static void handle_command(const char* cmd) {
    if (cmd[0] == '\0') return;

    lastCommandTime = millis();

    switch (cmd[0]) {
        case 'J':
            if (cmd[1] == ':') {
                handle_joint_command(cmd + 2);
            } else {
                serial_send_error(1, "Bad format, expected J:<angles>");
            }
            break;

        case 'S':
            if (cmd[1] == ':') {
                handle_single_joint(cmd + 2);
            } else {
                serial_send_error(1, "Bad format, expected S:<joint>:<angle>");
            }
            break;

        case 'H':
            servo_home();
            serial_send_ok();
            break;

        case 'Q':
            serial_send_positions();
            break;

        case 'E':
            if (cmd[1] == ':') {
                handle_enable(cmd + 2);
            } else {
                serial_send_error(1, "Bad format, expected E:<0|1>");
            }
            break;

        default:
            serial_send_error(1, "Unknown command");
            break;
    }
}

// ==================== Public API ====================

void serial_init() {
    Serial.begin(SERIAL_BAUD);
    bufferPos = 0;
}

void serial_poll() {
    while (Serial.available()) {
        char c = Serial.read();

        if (c == '\n' || c == '\r') {
            if (bufferPos > 0) {
                inputBuffer[bufferPos] = '\0';

                // Trim trailing whitespace
                while (bufferPos > 0 && (inputBuffer[bufferPos - 1] == ' ' ||
                       inputBuffer[bufferPos - 1] == '\r')) {
                    inputBuffer[--bufferPos] = '\0';
                }

                handle_command(inputBuffer);
                bufferPos = 0;
            }
        } else if (bufferPos < INPUT_BUFFER_SIZE - 1) {
            inputBuffer[bufferPos++] = c;
        }
        // Silently drop chars if buffer is full (prevents overflow)
    }
}

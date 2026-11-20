/* Edge Impulse ingestion SDK
 * Copyright (c) 2022 EdgeImpulse Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

/* Includes ---------------------------------------------------------------- */
#include <Tarea3_P1_inferencing.h>
#include <Arduino_LSM9DS1.h> //Click here to get the library: https://www.arduino.cc/reference/en/libraries/arduino_lsm9ds1/

/* Constant defines -------------------------------------------------------- */
#define CONVERT_G_TO_MS2    9.80665f
#define MAX_ACCEPTED_RANGE  2.0f

static bool debug_nn = false;

// 1. CORREGIDO: Sintaxis de resta (255 - r) para encender el LED correctamente
void setColor(int r, int g, int b) {
  analogWrite(LEDR, 255 - r);
  analogWrite(LEDG, 255 - g);
  analogWrite(LEDB, 255 - b);
}

void setup()
{
    Serial.begin(115200);
    while (!Serial);
    Serial.println("Edge Impulse Inferencing Demo - Acelerómetro + Giroscopio");

    if (!IMU.begin()) {
        ei_printf("Failed to initialize IMU!\r\n");
    }
    else {
        ei_printf("IMU initialized\r\n");
    }

    // 2. CORREGIDO: Ahora son 6 ejes (3 de acelerómetro + 3 de giroscopio)
    if (EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME != 6) {
        ei_printf("ERR: EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME deberia ser 6. Asegurate de marcar acc y gyr en Edge Impulse.\n");
        return;
    }
    
    pinMode(LEDR, OUTPUT);
    pinMode(LEDG, OUTPUT);
    pinMode(LEDB, OUTPUT);
    setColor(0, 0, 0); // Apagar al inicio
}

float ei_get_sign(float number) {
    return (number >= 0.0) ? 1.0 : -1.0;
}

void loop()
{
    ei_printf("\nStarting inferencing in 2 seconds...\n");
    delay(2000);
    ei_printf("Sampling...\n");

    float buffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE] = { 0 };

    // 3. CORREGIDO: El bucle ahora avanza de 6 en 6 (ix += 6)
    for (size_t ix = 0; ix < EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE; ix += 6) {
        uint64_t next_tick = micros() + (EI_CLASSIFIER_INTERVAL_MS * 1000);

        // Leer AMBOS sensores y guardarlos en el buffer consecutivamente
        IMU.readAcceleration(buffer[ix], buffer[ix + 1], buffer[ix + 2]);
        IMU.readGyroscope(buffer[ix + 3], buffer[ix + 4], buffer[ix + 5]);

        // Aplicar el límite y la conversión SOLO a los 3 primeros datos (Acelerómetro)
        for (int i = 0; i < 3; i++) {
            if (fabs(buffer[ix + i]) > MAX_ACCEPTED_RANGE) {
                // 4. CORREGIDO: Faltaba el asterisco (*) de multiplicación
                buffer[ix + i] = ei_get_sign(buffer[ix + i]) * MAX_ACCEPTED_RANGE;
            }
            buffer[ix + i] *= CONVERT_G_TO_MS2;
        }
        // Los datos del giroscopio (índices +3, +4, +5) se dejan intactos tal cual salen del sensor

        delayMicroseconds(next_tick - micros());
    }

    // Convertir el buffer en señal para Edge Impulse
    signal_t signal;
    int err = numpy::signal_from_buffer(buffer, EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, &signal);
    if (err != 0) {
        ei_printf("Failed to create signal from buffer (%d)\n", err);
        return;
    }

    // Ejecutar el clasificador
    ei_impulse_result_t result = { 0 };
    err = run_classifier(&signal, &result, debug_nn);
    if (err != EI_IMPULSE_OK) {
        ei_printf("ERR: Failed to run classifier (%d)\n", err);
        return;
    }

    // Imprimir predicciones
    ei_printf("Predictions ");
    ei_printf("(DSP: %d ms., Classification: %d ms., Anomaly: %d ms.): \n",
        result.timing.dsp, result.timing.classification, result.timing.anomaly);
        
    float max_val = 0;
    int best_class = -1;
    for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
        ei_printf("    %s: %.5f\n", result.classification[ix].label, result.classification[ix].value);
        if (result.classification[ix].value > max_val) {
            max_val = result.classification[ix].value;
            best_class = ix;
        }
    }
    
    // Si la confianza es mayor al 70%, encendemos el LED
    if (max_val > 0.70) {
        String etiqueta = result.classification[best_class].label;
        
        if (etiqueta == "Roll_Pos") {
            setColor(255, 0, 0); // Rojo
        } else if (etiqueta == "Roll_Neg") {
            setColor(255, 105, 180); // Rosa
        } else if (etiqueta == "Pitch_Pos") {
            setColor(0, 255, 0); // Verde
        } else if (etiqueta == "Pitch_Neg") {
            setColor(255, 255, 0); // Amarillo
        } else if (etiqueta == "Yaw_Pos") {
            setColor(0, 0, 255); // Azul
        } else if (etiqueta == "Yaw_Neg") {
            setColor(128, 0, 128); // Morado
        } else {
            setColor(0, 0, 0); // Apagado
        }
    } else {
        setColor(0, 0, 0); // Apagar si ninguna llega al 70%
    }
}

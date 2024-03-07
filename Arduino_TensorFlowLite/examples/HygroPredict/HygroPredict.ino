/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "DHT.h"
#include <TensorFlowLite.h>

#include "main_functions.h"

#include "my_test_model.h"
#include "tensorflow/lite/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// 센서 핀 정의
#define low_dt1 15
#define low_dt2 28
#define low_dt3 27
#define low_dt4 26
#define high_dt 22

DHT low_Dht1(low_dt1, DHT11);
DHT low_Dht2(low_dt2, DHT11);
DHT low_Dht3(low_dt3, DHT11);
DHT low_Dht4(low_dt4, DHT11);
DHT high_Dht(high_dt, DHT22);

namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;
  constexpr int kTensorArenaSize = 8 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
}


void setup() {
  Serial.begin(9600);
  low_Dht1.begin();
  low_Dht2.begin();
  low_Dht3.begin();
  low_Dht4.begin();
  high_Dht.begin();

  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  model = tflite::GetModel(g_hygropredict_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter, "Model schema version mismatch!");
    return;
  }

  static tflite::ops::micro::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("Failed to allocate tensors!");
    return;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);
}


// The name of this function is important for Arduino compatibility.
void loop() {
  // 센서값 읽기
  float low1_h = low_Dht1.readHumidity();
  float low1_t = low_Dht1.readTemperature();
  float low2_h = low_Dht2.readHumidity();
  float low2_t = low_Dht2.readTemperature();
  float low3_h = low_Dht3.readHumidity();
  float low3_t = low_Dht3.readTemperature();
  float low4_h = low_Dht4.readHumidity();
  float low4_t = low_Dht4.readTemperature();
  float high_h = high_Dht.readHumidity();
  float high_t = high_Dht.readTemperature();

    if (isnan(low1_h) || isnan(low1_t)|| isnan(low2_h) || isnan(low2_t)|| isnan(low3_h) 
      || isnan(low3_t)|| isnan(low4_h) || isnan(low4_t)|| isnan(high_h) || isnan(high_t) ) {
      //값 읽기 실패시 시리얼 모니터 출력
      Serial.println("Failed to read from DHT");
    }else {
    // 입력 텐서에 센서 데이터 설정
    input->data.f[0] = low1_h;
    input->data.f[1] = low1_t;
    input->data.f[2] = low2_h;
    input->data.f[3] = low2_t;
    input->data.f[4] = low3_h;
    input->data.f[5] = low3_t;
    input->data.f[6] = low4_h;
    input->data.f[7] = low4_t;

    // 모델 실행
    if (interpreter->Invoke() != kTfLiteOk) {
      Serial.println("Failed to invoke tflite!");
      return;
    }

    // 출력 텐서에서 예측값 읽기
    float predicted_high_h = output->data.f[0];
    float predicted_high_t = output->data.f[1];

    // 결과 출력
    Serial.println("Predicted:[" + String(predicted_high_h) + ", " + String(predicted_high_t) + 
               "]   Real:[" + String(high_h) + ", " + String(high_t) + "]");

  }
  delay(1500);
}

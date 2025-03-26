#include <iostream>
#include <vector>
#include <memory> // smart pointers
#include <limits> // std::numeric_limits
#include <fstream>
#include <chrono>

#include "csv.h"
#include "model/xgboost.h"

int main() {

  constexpr std::size_t num_features = 23;
  io::CSVReader<num_features> features("X_train3.csv");
  // features.read_header(io::ignore_extra_column, "age", "hypertension", "heart_disease", "bmi", "HbA1c_level", 
  //                      "blood_glucose_level", "gender_Female", "gender_Male", "smoking_history_current", 
  //                      "smoking_history_ever", "smoking_history_former", "smoking_history_never", "smoking_history_not_current");
  features.read_header(io::ignore_extra_column, "Age", "Flight Distance", "Inflight Wifi Service", 
                       "Departure/Arrival Time Convenient", "Ease Of Online Booking", "Gate Location", 
                       "Food And Drink", "Online Boarding", "Seat Comfort", "Inflight Entertainment", 
                       "On-Board Service", "Leg Room Service", "Baggage Handling", "Checkin Service", 
                       "Inflight Service", "Cleanliness", "Departure Delay In Minutes", "Arrival Delay In Minutes", 
                       "Gender Male", "Customer Type Disloyal Customer", "Type Of Travel Personal Travel", 
                       "Class Eco", "Class Eco Plus");

  std::shared_ptr<Data> data = std::make_shared<Data>(num_features);
  std::vector<float> row_values(num_features, std::numeric_limits<float>::quiet_NaN());

  while (features.read_row(row_values[0], row_values[1], row_values[2], row_values[3], row_values[4], 
                           row_values[5], row_values[6], row_values[7], row_values[8], row_values[9],
                           row_values[10], row_values[11], row_values[12], row_values[13], row_values[14],
                           row_values[15], row_values[16], row_values[17], row_values[18], row_values[19],
                           row_values[20], row_values[21], row_values[22])) {

    data->AddRow(row_values);
    // reset row_values values
    std::fill(row_values.begin(), row_values.end(), std::numeric_limits<float>::quiet_NaN());

  }

  constexpr std::size_t num_responders = 1;
  io::CSVReader<num_responders> responder("y_train3.csv");
  responder.read_header(io::ignore_extra_column, "Satisfaction Satisfied");

  std::vector<int> responses;
  responses.reserve(data->num_rows);
  int result;

  while (responder.read_row(result)) {
    responses.push_back(result);
  }

  data->SortFeatureBlocks();
  std::shared_ptr<TrainingDataRowInformation> data_row_information = std::make_shared<TrainingDataRowInformation>(data->num_rows);

  std::size_t max_depth = 6;
  int wqs_splits = 256;
  float lambda = 1;
  float learning_rate = 0.2;
  float gamma = 1;
  std::size_t num_rounds = 100;
  std::unique_ptr<XGBoost> xgboost = std::make_unique<XGBoost>(
    max_depth, data, data_row_information, responses, wqs_splits, lambda, learning_rate, gamma, num_rounds
  );

  auto start = std::chrono::high_resolution_clock::now();
  xgboost->Train();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  std::cout << "Time taken: " << duration.count() << " seconds!\n";
  // xgboost->PrintModel();

  std::ofstream weights_file;
  weights_file.open("weights3.bin", std::ios::out | std::ios::binary); // don't actually need std::ios::out as it's ofstream
  xgboost->Save(weights_file);

}

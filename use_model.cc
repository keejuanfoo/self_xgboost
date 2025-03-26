#include <iostream>
#include <fstream>
#include <memory>

#include "model/xgboost.h"
#include "csv.h"

int main() {

  std::ifstream weights_file("weights3.bin");
  std::unique_ptr<XGBoost> xgboost = std::make_unique<XGBoost>();
  xgboost->Load(weights_file);
  
  // constexpr std::size_t num_features = 13;
  // io::CSVReader<num_features> features("X_test.csv");
  // features.read_header(io::ignore_extra_column, "age", "hypertension", "heart_disease", "bmi", "HbA1c_level", 
  //                      "blood_glucose_level", "gender_Female", "gender_Male", "smoking_history_current", 
  //                      "smoking_history_ever", "smoking_history_former", "smoking_history_never", "smoking_history_not_current");

  constexpr std::size_t num_features = 23;
  io::CSVReader<num_features> features("X_test3.csv");
  features.read_header(io::ignore_extra_column, "Age", "Flight Distance", "Inflight Wifi Service", 
                       "Departure/Arrival Time Convenient", "Ease Of Online Booking", "Gate Location", 
                       "Food And Drink", "Online Boarding", "Seat Comfort", "Inflight Entertainment", 
                       "On-Board Service", "Leg Room Service", "Baggage Handling", "Checkin Service", 
                       "Inflight Service", "Cleanliness", "Departure Delay In Minutes", "Arrival Delay In Minutes", 
                       "Gender Male", "Customer Type Disloyal Customer", "Type Of Travel Personal Travel", 
                       "Class Eco", "Class Eco Plus");

  std::vector<std::vector<float>> features_list;
  features_list.resize(num_features);
  std::vector<float> row_values(num_features, std::numeric_limits<float>::quiet_NaN());
  std::size_t num_rows = 0;

  while (features.read_row(row_values[0], row_values[1], row_values[2], row_values[3], row_values[4], 
                           row_values[5], row_values[6], row_values[7], row_values[8], row_values[9],
                           row_values[10], row_values[11], row_values[12], row_values[13], row_values[14],
                           row_values[15], row_values[16], row_values[17], row_values[18], row_values[19],
                           row_values[20], row_values[21], row_values[22])) {

    for (std::size_t i = 0; i < num_features; i++) {
      features_list[i].push_back(row_values[i]);
    }
    num_rows++;
    std::fill(row_values.begin(), row_values.end(), std::numeric_limits<float>::quiet_NaN());

  }

  std::shared_ptr<PredictionDataRowInformation> X_test = 
    std::make_shared<PredictionDataRowInformation>(features_list, num_features, num_rows);

  // xgboost->PrintModel();
  xgboost->Predict(X_test);

  std::ofstream scores_file("logits3.csv");
  xgboost->StoreScores(X_test, scores_file);
  scores_file.close();

  std::ofstream probabilities_file("probabilities3.csv");
  xgboost->StoreProbabilties(X_test, probabilities_file);
  probabilities_file.close();

  std::ofstream predictions_file("predictions3.csv");
  xgboost->StorePredictions(X_test, predictions_file);
  predictions_file.close();
}

#pragma once

#include <memory>
#include <fstream>
#include <vector>

#include "../data/model_data_structure.h"
#include "../data/training_data.h"
#include "../data/predictions_data.h"
#include "decision_tree.h"

class XGBoost {
public:
  XGBoost(
    std::size_t max_depth, 
    std::shared_ptr<Data>& data,
    std::shared_ptr<TrainingDataRowInformation>& data_row_information,
    std::vector<int>& responses,
    int wqs_splits,
    float lambda,
    float learning_rate,
    float gamma,
    std::size_t num_trees
  );

  XGBoost();

  void Train();

  void Predict(std::shared_ptr<PredictionDataRowInformation>& X_test);

  void StoreScores(std::shared_ptr<PredictionDataRowInformation>& X_test, std::ofstream& scores_file);

  void StoreProbabilties(std::shared_ptr<PredictionDataRowInformation>& X_test, std::ofstream& probabilities_file);

  void StorePredictions(std::shared_ptr<PredictionDataRowInformation>& X_test, std::ofstream& predictions_file);

  void Save(std::ofstream& weights_file);

  void Load(std::ifstream& weights_file);

  void PrintModel();

private:
  std::vector<std::unique_ptr<DecisionTree>> trees;
  std::size_t max_depth;
  std::shared_ptr<Data> data;
  std::shared_ptr<TrainingDataRowInformation> data_row_information;
  std::vector<std::shared_ptr<ColumnBlock>> column_blocks;
  std::vector<int> responses;
  int wqs_splits;
  float lambda;
  float learning_rate;
  float gamma;
  std::size_t num_trees;

  void MakeColumnBlocks(std::vector<std::shared_ptr<ColumnBlock>>& column_blocks);

  void WeightedQuantileSketch(std::vector<std::shared_ptr<ColumnBlock>>& column_blocks, std::size_t feature_index);
};

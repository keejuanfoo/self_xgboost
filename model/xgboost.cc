#include "xgboost.h"

#include <iostream>
#include <algorithm>
#include <unordered_set>
#include <thread>

#include "../utils/math_utils.h"

XGBoost::XGBoost(
  std::size_t max_depth, 
  std::shared_ptr<Data>& data,
  std::shared_ptr<TrainingDataRowInformation>& data_row_information,
  std::vector<int>& responses,
  int wqs_splits,
  float lambda,
  float learning_rate,
  float gamma,
  std::size_t num_trees
)
  : max_depth(max_depth), 
    data(data), 
    data_row_information(data_row_information),
    responses(responses),
    wqs_splits(wqs_splits),
    lambda(lambda),
    learning_rate(learning_rate),
    gamma(gamma),
    num_trees(num_trees)
{
  trees.resize(num_trees);
}

XGBoost::XGBoost() {}

void XGBoost::Train() {
  std::vector<std::shared_ptr<ColumnBlock>> column_blocks;
  for (std::size_t tree_index = 0; tree_index < num_trees; tree_index++) {
    std::cout << "Training tree " << tree_index << "\n";
    trees[tree_index] = std::make_unique<DecisionTree>(
      max_depth, column_blocks, data->num_features, data_row_information , responses, wqs_splits, lambda, learning_rate, gamma
    );
    std::fill(data_row_information->node_indices.begin(), data_row_information->node_indices.end(), 0);
    if (tree_index == 0) {
      trees[tree_index]->CalculateGradientsAndHessians(data_row_information->indices);
      MakeColumnBlocks(column_blocks);
    }
    trees[tree_index]->BuildDecisionTree();
  }
}

void XGBoost::Predict(std::shared_ptr<PredictionDataRowInformation>& X_test) {
  // Perform the full pass for each tree
  for (std::size_t i = 0; i < num_trees; i++) {
    std::cout << "Traversing tree " << i << "\n";
    trees[i]->ScoringPass(X_test);
  }
}

void XGBoost::StoreScores(std::shared_ptr<PredictionDataRowInformation>& X_test, std::ofstream& scores_file) {
  for (const auto& logit : X_test->logits) {
    scores_file << logit << "\n";
  }
}

void XGBoost::StoreProbabilties(std::shared_ptr<PredictionDataRowInformation>& X_test, std::ofstream& probabilities_file) {
  for (const auto& logit : X_test->logits) {
    probabilities_file << Sigmoid(logit) << "\n";
  }
}

void XGBoost::StorePredictions(std::shared_ptr<PredictionDataRowInformation>& X_test, std::ofstream& predictions_file) {
  int predicted_class;
  for (const auto& logit : X_test->logits) {
    predicted_class = (logit >= 0) ? 1 : 0;
    predictions_file << predicted_class << "\n";
  }
}

void XGBoost::Save(std::ofstream& weights_file) {
  weights_file.write(reinterpret_cast<const char*>(&num_trees), sizeof(num_trees));

  for (const auto& tree : trees) {
    tree->SerialiseTree(weights_file);
  }
}

void XGBoost::Load(std::ifstream& weights_file) {
  weights_file.read(reinterpret_cast<char*>(&num_trees), sizeof(num_trees));
  trees.resize(num_trees);
  std::vector<std::shared_ptr<ColumnBlock>> column_blocks;

  for (std::size_t i = 0; i < num_trees; i++) {
    trees[i] = std::make_unique<DecisionTree>(column_blocks);
    trees[i]->DeserialiseTree(weights_file);
  }
}

void XGBoost::PrintModel() {
  for (std::size_t i = 0; i < num_trees; i++) {
    std::cout << "----- Tree " << i << " -----\n\n";
    trees[i]->PrintDecisionTree();
    std::cout << "\n\n";
  }
}

void XGBoost::MakeColumnBlocks(std::vector<std::shared_ptr<ColumnBlock>>& column_blocks) {
  column_blocks.resize(data->num_features);
  std::vector<std::thread> threads;

  for (std::size_t feature_index = 0; feature_index < data->num_features; feature_index++) {
    threads.emplace_back([&, feature_index]() {
      column_blocks[feature_index] = std::make_shared<ColumnBlock>(data->num_rows, wqs_splits);
      WeightedQuantileSketch(column_blocks, feature_index);
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }
}

void XGBoost::WeightedQuantileSketch(std::vector<std::shared_ptr<ColumnBlock>>& column_blocks, std::size_t feature_index) {
  std::unordered_set<std::size_t> missing_row_set;
  for (std::size_t missing_row_index = 0; missing_row_index < data->num_rows; missing_row_index++) {
    missing_row_set.insert(missing_row_index);
  }

  std::size_t row_index;
  float total_gradient = 0;
  float total_hessian = 0;
  for (std::size_t block_index = 0; block_index < data->feature_blocks[feature_index]->values.size(); block_index++) {
    row_index = data->feature_blocks[feature_index]->indices[block_index];

    total_gradient += data_row_information->gradients[row_index];
    total_hessian += data_row_information->hessians[row_index];
    missing_row_set.erase(row_index);
  }

  float cumulative_gradients = 0;
  float cumulative_hessians = 0;
  int bin_cnt = 1;
  float previous_value = 0;
  float current_score, current_value;
  for (std::size_t block_index = 0; block_index < data->feature_blocks[feature_index]->values.size(); block_index++) {
    row_index = data->feature_blocks[feature_index]->indices[block_index];
    current_value = data->feature_blocks[feature_index]->values[block_index];

    cumulative_gradients += data_row_information->gradients[row_index];
    cumulative_hessians += data_row_information->hessians[row_index];

    if (cumulative_hessians > (total_hessian / wqs_splits * bin_cnt)) {
      column_blocks[feature_index]->thresholds[bin_cnt - 1] = (current_value + previous_value) / 2;
      bin_cnt++;
    }

    column_blocks[feature_index]->bin_index[row_index] = bin_cnt - 1;

    previous_value = current_value;
  }
}

#pragma once

#include <memory>
#include <vector>
#include <fstream>
#include <arm_neon.h>

#include "../data/model_data_structure.h"
#include "../data/training_data.h"
#include "../data/predictions_data.h"
#include "tree_node.h"

class DecisionTree {
public:
  DecisionTree(
    std::size_t max_depth, 
    // std::shared_ptr<Data>& data,
    std::vector<std::shared_ptr<ColumnBlock>>& column_blocks,
    std::size_t num_features,
    std::shared_ptr<TrainingDataRowInformation>& data_row_information,
    std::vector<int>& responses,
    int wqs_splits,
    float lambda,
    float learning_rate,
    float gamma
  );

  DecisionTree(std::vector<std::shared_ptr<ColumnBlock>>& column_blocks);

  void CalculateGradientsAndHessians(std::vector<std::size_t>& update_list);

  void BuildDecisionTree();

  void PrintDecisionTree();

  void SerialiseTree(std::ofstream& weights_file);

  void DeserialiseTree(std::ifstream& weights_file);

  void ScoringPass(std::shared_ptr<PredictionDataRowInformation>& X_test);

private:
  std::vector<std::unique_ptr<TreeNode>> tree_nodes;
  std::size_t max_depth;
  std::vector<std::size_t> leaf_indices;
  std::vector<std::shared_ptr<ColumnBlock>>& column_blocks;
  std::size_t num_features;
  std::shared_ptr<TrainingDataRowInformation> data_row_information;
  std::vector<int> responses;
  std::size_t last_layer_left_index;
  std::size_t tree_nodes_size;
  int wqs_splits;
  float lambda;
  float learning_rate;
  float gamma;


  float CalculateGain(float cumulative_gradients, float cumulative_hessians, float total_gradient, float total_hessian);

  float32x4_t CalculateGainSimd(
    float32x4_t cumulative_gradients,
    float32x4_t cumulative_hessians,
    float32x4_t total_gradient,
    float32x4_t total_hessian
  );

  float CalculateGainsInNodeForFeature(
    std::size_t current_node, 
    std::size_t feature_index, 
    std::size_t& feature_best_threshold_index,
    bool& feature_left_default
  );

  float CalculateGainsInNode(std::size_t current_node, std::size_t& node_threshold_index);

  // returns node to split, changes the threshold internally, and we will update based on that
  std::size_t GetNodeToSplit(std::size_t& threshold_index);

  float CalculateWeights(float gradients, float hessians);

  bool PerformSplit();
};

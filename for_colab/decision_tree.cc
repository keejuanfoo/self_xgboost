#include "decision_tree.h"

#include <iostream>
#include <unordered_set>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <chrono>
#include <utility>
#include <array>
#include <algorithm>
#include <cmath>

#include "../utils/math_utils.h"
#include "calculate_gain_helper.cuh"

DecisionTree::DecisionTree(
  std::size_t max_depth,
  std::vector<std::shared_ptr<ColumnBlock>>& column_blocks,
  std::size_t num_features,
  std::shared_ptr<TrainingDataRowInformation>& data_row_information,
  std::vector<int>& responses,
  int wqs_splits,
  float lambda,
  float learning_rate,
  float gamma
)
  : max_depth(max_depth), 
    column_blocks(column_blocks),
    num_features(num_features),
    data_row_information(data_row_information),
    responses(responses),
    wqs_splits(wqs_splits),
    lambda(lambda),
    learning_rate(learning_rate),
    gamma(gamma)
{
  last_layer_left_index = static_cast<std::size_t>(std::pow(2, max_depth) - 1);
  tree_nodes_size = static_cast<std::size_t>(std::pow(2, max_depth + 1) - 1);
  tree_nodes.resize(tree_nodes_size);
  std::generate(tree_nodes.begin(), tree_nodes.end(), []() {
    return nullptr;
  });
  tree_nodes[0] = std::make_unique<TreeNode>(0);
  leaf_indices.push_back(0);
}

DecisionTree::DecisionTree(std::vector<std::shared_ptr<ColumnBlock>>& column_blocks) : column_blocks(column_blocks) {}

void DecisionTree::CalculateGradientsAndHessians(std::vector<std::size_t>& update_list) {
  for (const auto& row_index : update_list) {
    data_row_information->gradients[row_index] = CalculateGradient(
      data_row_information->logits[row_index], responses[row_index]
    );
    data_row_information->hessians[row_index] = CalculateHessian(data_row_information->logits[row_index]);
  }
}

// void DecisionTree::CalculateGradientsAndHessiansSimd(std::vector<std::size_t>& update_list) {
//   for (std::size_t base_index = 0; base_index + 3 < update_list.size(); base_index += 4) {
//     std::vector
//     for (std::size_t offset = 0; offset < 4; offset++) {

//     }
//   }
// }

void DecisionTree::BuildDecisionTree() {
  bool continue_building = true;
  // set up the weights first node, I couldn't do this in the constructor as I didn't have the gradients yet
  float total_gradients = 0;
  float total_hessians = 0;
  for (std::size_t row_index = 0; row_index < data_row_information->num_rows; row_index++) {
    total_gradients += data_row_information->gradients[row_index];
    total_hessians += data_row_information->hessians[row_index];
  }
  tree_nodes[0]->weights = CalculateWeights(total_gradients, total_hessians);
  // update the logits
  for (std::size_t row_index = 0; row_index < data_row_information->num_rows; row_index++) {
    data_row_information->logits[row_index] += learning_rate * tree_nodes[0]->weights;
  }

  while (continue_building) {
    continue_building = PerformSplit();
  }
}

void DecisionTree::PrintDecisionTree() {
  std::size_t index = 0;
  for (const auto& tree_node : tree_nodes) {
    if (tree_node != nullptr) {
      std::cout << "index = " << index << "\n";
      std::cout << "feature = " << tree_node->feature_index << "\n";
      std::cout << "threshold = " << tree_node->threshold << "\n";
      std::cout << "weights = " << tree_node->weights << "\n\n";
    }
    index++;
  }
}

void DecisionTree::SerialiseTree(std::ofstream& weights_file) {
  weights_file.write(reinterpret_cast<const char*>(&tree_nodes_size), sizeof(tree_nodes_size));

  for (const auto& node : tree_nodes) {
    bool no_node = (node == nullptr);
    weights_file.write(reinterpret_cast<const char*>(&no_node), sizeof(no_node));

    if (!no_node) {
      node->SerialiseNode(weights_file);
    }
  }
}

void DecisionTree::DeserialiseTree(std::ifstream& weights_file) {
  weights_file.read(reinterpret_cast<char*>(&tree_nodes_size), sizeof(tree_nodes_size));
  tree_nodes.resize(tree_nodes_size);
  bool no_node;
  
  for (std::size_t i = 0; i < tree_nodes_size; i++) {
    weights_file.read(reinterpret_cast<char*>(&no_node), sizeof(no_node));

    if (!no_node) {
      tree_nodes[i] = std::make_unique<TreeNode>(i);
      tree_nodes[i]->DeserialiseNode(weights_file);
    } else {
      tree_nodes[i] = nullptr;
    }
  }
}

void DecisionTree::ScoringPass(std::shared_ptr<PredictionDataRowInformation>& X_test) {
  // Allow each row to pass all the way to the leaf each time
  std::size_t cur_node_index = 0;
  std::size_t next_node_index;
  for (std::size_t i = 0; i < X_test->num_rows; i++) {
    // Traverse tree
    next_node_index = tree_nodes[0]->SendToNextNode(X_test->features, i);
    while (next_node_index < tree_nodes_size && tree_nodes[next_node_index] != nullptr) {
      cur_node_index = next_node_index;
      next_node_index = tree_nodes[cur_node_index]->SendToNextNode(X_test->features, i);
    }
    // Update logit for that row
    X_test->logits[i] += tree_nodes[cur_node_index]->weights;
  }
}

float DecisionTree::CalculateGain(
  float cumulative_gradients, 
  float cumulative_hessians, 
  float total_gradient, 
  float total_hessian
) 
{
  float left_gain, right_gain, total_gain;
  float right_gradients = total_gradient - cumulative_gradients;
  float right_hessians = total_hessian - cumulative_hessians;
  const float epsilon = 1e-8;

  left_gain = (cumulative_gradients * cumulative_gradients) / (cumulative_hessians + lambda + epsilon);
  right_gain = (right_gradients * right_gradients) / (right_hessians + lambda + epsilon);
  total_gain = (total_gradient * total_gradient) / (total_hessian + lambda + epsilon);
  
  return 0.5 * (left_gain + right_gain - total_gain) - gamma;
}

// float32x4_t DecisionTree::CalculateGainSimd(
//   float32x4_t cumulative_gradients,
//   float32x4_t cumulative_hessians,
//   float32x4_t total_gradient,
//   float32x4_t total_hessian
// )
// {
//   float32x4_t left_gain, right_gain, total_gain;
//   float32x4_t right_gradients = vsubq_f32(total_gradient, cumulative_gradients);
//   float32x4_t right_hessians = vsubq_f32(total_hessian, cumulative_hessians);
//   const float epsilon = 1e-8;
//   float32x4_t epsilon_vector = vdupq_n_f32(epsilon);

//   float32x4_t lambda_vector = vdupq_n_f32(lambda);
//   left_gain = vdivq_f32(
//     vmulq_f32(cumulative_gradients, cumulative_gradients), 
//     vaddq_f32(vaddq_f32(cumulative_hessians, lambda_vector), epsilon_vector)
//   );
//   right_gain = vdivq_f32(
//     vmulq_f32(right_gradients, right_gradients), 
//     vaddq_f32(vaddq_f32(right_hessians, lambda_vector), epsilon_vector)
//   );
//   total_gain = vdivq_f32(
//     vmulq_f32(total_gradient, total_gradient), 
//     vaddq_f32(vaddq_f32(total_hessian, lambda_vector), epsilon_vector)
//   );

//   float32x4_t gamma_vector = vdupq_n_f32(gamma);
//   float32x4_t half_vector = vdupq_n_f32(0.5f);
//   return vsubq_f32(vmulq_f32(half_vector, vsubq_f32(vaddq_f32(left_gain, right_gain), total_gain)), gamma_vector);
// }


float DecisionTree::CalculateGainsInNodeForFeature(
  std::size_t current_node, 
  std::size_t feature_index, 
  std::size_t& feature_threshold_index,
  bool& feature_left_default) 
{
  // calculate the gradients in each bin as well as the missing gradients
  std::vector<float> bin_gradients(wqs_splits, 0);
  std::vector<float> bin_hessians(wqs_splits, 0);
  float missing_gradients = 0;
  float missing_hessians = 0;
  float total_gradient = 0;
  float total_hessian = 0;
  // std::size_t current_bin_index;
  // bool missing_value_flag = false;
  int missing_value_flag = 0;

  LaunchBinGeneratorKernel(
    data_row_information->num_rows,
    data_row_information->node_indices.data(),
    current_node,
    wqs_splits,
    column_blocks[feature_index]->bin_index.data(),
    bin_gradients.data(),
    bin_hessians.data(),
    data_row_information->gradients.data(),
    data_row_information->hessians.data(),
    &total_gradient,
    &total_hessian,
    &missing_gradients,
    &missing_hessians,
    &missing_value_flag
  );

  // std::cout << "total_gradient = " << total_gradient << "\n";

  // for (std::size_t row_index = 0; row_index < data_row_information->num_rows; row_index++) {
  //   if (data_row_information->node_indices[row_index] == current_node) {
  //     current_bin_index = column_blocks[feature_index]->bin_index[row_index];
  //     if (current_bin_index != -1) {
  //       bin_gradients[current_bin_index] += data_row_information->gradients[row_index];
  //       bin_hessians[current_bin_index] += data_row_information->hessians[row_index];
  //       total_gradient += data_row_information->gradients[row_index];
  //       total_hessian += data_row_information->hessians[row_index];
  //     } else {
  //       missing_value_flag = true;
  //       missing_gradients += data_row_information->gradients[row_index];
  //       missing_hessians += data_row_information->hessians[row_index];
  //     }
  //   }
  // }

  // Get best split gain
  float maximum_node_score = 0;
  float current_score;
  float cumulative_gradients = 0;
  float cumulative_hessians = 0;
  for (std::size_t bin_index = 0; bin_index < wqs_splits - 1; bin_index++) {
    cumulative_gradients += bin_gradients[bin_index];
    cumulative_hessians += bin_hessians[bin_index];
    
    if (!missing_value_flag) {
      current_score = CalculateGain(cumulative_gradients, cumulative_hessians, total_gradient, total_hessian);
      if (current_score > maximum_node_score) {
        maximum_node_score = current_score;
        feature_threshold_index = bin_index;
        feature_left_default = true;
      }
    } else {
      // add missing to left
      current_score = CalculateGain(
                        cumulative_gradients + missing_gradients, 
                        cumulative_hessians + missing_hessians, 
                        total_gradient + missing_gradients, 
                        total_hessian + missing_hessians
                      );
      if (current_score > maximum_node_score) {
        maximum_node_score = current_score;
        feature_threshold_index = bin_index;
        feature_left_default = true;
      }

      // add missing to right
      current_score = CalculateGain(
                        cumulative_gradients, 
                        cumulative_hessians, 
                        total_gradient + missing_gradients, 
                        total_hessian + missing_hessians
                      );
      if (current_score > maximum_node_score) {
        maximum_node_score = current_score;
        feature_threshold_index = bin_index;
        feature_left_default = false;
      }
    }
  }

  return maximum_node_score;
}

// // Get best split gain
// std::vector<float> gradient_prefix_sum(wqs_splits);
// std::vector<float> hessian_prefix_sum(wqs_splits);
// for (std::size_t bin_index = 0; bin_index < wqs_splits - 1; bin_index++) {
//   cumulative_gradients += bin_gradients[bin_index];
//   cumulative_hessians += bin_hessians[bin_index];
//   gradient_prefix_sum[bin_index + 1] = cumulative_gradients;
//   hessian_prefix_sum[bin_index + 1] = cumulative_hessians;
// }

// float DecisionTree::CalculateGainsInNodeForFeature(
//   std::size_t current_node, 
//   std::size_t feature_index, 
//   std::size_t& feature_threshold_index,
//   bool& feature_left_default) 
// {
//   // calculate the gradients in each bin as well as the missing gradients
//   std::vector<float> bin_gradients(wqs_splits, 0);
//   std::vector<float> bin_hessians(wqs_splits, 0);
//   float missing_gradients = 0;
//   float missing_hessians = 0;
//   float total_gradient = 0;
//   float total_hessian = 0;
//   std::size_t current_bin_index;
//   bool missing_value_flag = false;
//   for (std::size_t row_index = 0; row_index < data_row_information->num_rows; row_index++) {
//     if (data_row_information->node_indices[row_index] == current_node) {
//       current_bin_index = column_blocks[feature_index]->bin_index[row_index];
//       if (current_bin_index != -1) {
//         bin_gradients[current_bin_index] += data_row_information->gradients[row_index];
//         bin_hessians[current_bin_index] += data_row_information->hessians[row_index];
//         total_gradient += data_row_information->gradients[row_index];
//         total_hessian += data_row_information->hessians[row_index];
//       } else {
//         missing_value_flag = true;
//         missing_gradients += data_row_information->gradients[row_index];
//         missing_hessians += data_row_information->hessians[row_index];
//       }
//     }
//   }

//   // Get best split gain, with usage of SIMD intrinsics for gain calculation
//   float cumulative_gradients = 0;
//   float cumulative_hessians = 0;
//   float maximum_node_score = 0;
//   if (!missing_value_flag) {
//     for (std::size_t bin_index = 0; bin_index < wqs_splits - 1; bin_index += 4) {
//       std::array<float, 4> cumulative_gradients_vector_create = {};
//       std::array<float, 4> cumulative_hessians_vector_create = {};
//       for (std::size_t offset = 0; offset < 4; offset++) {
//         cumulative_gradients += bin_gradients[bin_index + offset];
//         cumulative_hessians += bin_hessians[bin_index + offset];
//         cumulative_gradients_vector_create[offset] = cumulative_gradients;
//         cumulative_hessians_vector_create[offset] = cumulative_hessians;
//       }
//       float32x4_t cumulative_gradients_vector = vld1q_f32(cumulative_gradients_vector_create.data());
//       float32x4_t cumulative_hessians_vector = vld1q_f32(cumulative_hessians_vector_create.data());

//       float32x4_t total_gradient_vector = vdupq_n_f32(total_gradient);
//       float32x4_t total_hessian_vector = vdupq_n_f32(total_hessian);
//       float32x4_t current_scores_vector = CalculateGainSimd(
//         cumulative_gradients_vector, 
//         cumulative_hessians_vector, 
//         total_gradient_vector, 
//         total_hessian_vector
//       );

//       std::array<float, 4> current_scores = {};
//       vst1q_f32(current_scores.data(), current_scores_vector);
//       for (std::size_t offset = 0; offset < 4; offset++) {
//         if (current_scores[offset] > maximum_node_score) {
//           maximum_node_score = current_scores[offset];
//           feature_threshold_index = bin_index + offset;
//           feature_left_default = true;
//         }
//       }
//     }
//   } else {
//     for (std::size_t bin_index = 0; bin_index < wqs_splits - 1; bin_index += 2) {
//       std::array<float, 4> cumulative_gradients_vector_create = {};
//       std::array<float, 4> cumulative_hessians_vector_create = {};
//       for (std::size_t offset = 0; offset < 2; offset++) {
//         cumulative_gradients += bin_gradients[bin_index + offset];
//         cumulative_hessians += bin_hessians[bin_index + offset];
//         cumulative_gradients_vector_create[offset * 2] = cumulative_gradients + missing_gradients;
//         cumulative_hessians_vector_create[offset * 2] = cumulative_hessians + missing_hessians;
//         cumulative_gradients_vector_create[offset * 2 + 1] = cumulative_gradients;
//         cumulative_hessians_vector_create[offset * 2 + 1] = cumulative_hessians;
//       }
//       float32x4_t cumulative_gradients_vector = vld1q_f32(cumulative_gradients_vector_create.data());
//       float32x4_t cumulative_hessians_vector = vld1q_f32(cumulative_hessians_vector_create.data());

//       float32x4_t total_gradient_vector = vdupq_n_f32(total_gradient + missing_gradients);
//       float32x4_t total_hessian_vector = vdupq_n_f32(total_hessian + missing_gradients);
//       float32x4_t current_scores_vector = CalculateGainSimd(
//         cumulative_gradients_vector, 
//         cumulative_hessians_vector, 
//         total_gradient_vector, 
//         total_hessian_vector
//       );

//       std::array<float, 4> current_scores = {};
//       vst1q_f32(current_scores.data(), current_scores_vector);
//       for (std::size_t offset = 0; offset < 4; offset++) {
//         if (current_scores[offset] > maximum_node_score) {
//           maximum_node_score = current_scores[offset];
//           feature_threshold_index = bin_index + offset / 2;
//           feature_left_default = (offset % 2 == 0);
//         }
//       }
//     }
//   }

//   return maximum_node_score;
// }

float DecisionTree::CalculateGainsInNode(std::size_t current_node, std::size_t& node_threshold_index) {
  std::mutex change_node_parameters_mutex;
  float best_gain = 0;
  std::vector<std::thread> threads;

  for (std::size_t feature_index = 0; feature_index < num_features; feature_index++) {
    threads.emplace_back([&, feature_index]() {
      std::size_t feature_threshold_index;
      bool feature_left_default = true;
      float feature_best_gain = CalculateGainsInNodeForFeature(
                                  current_node, 
                                  feature_index, 
                                  feature_threshold_index,
                                  feature_left_default
                                );

      change_node_parameters_mutex.lock();
      if (feature_best_gain > best_gain) {
        best_gain = feature_best_gain;
        node_threshold_index = feature_threshold_index;
        tree_nodes[current_node]->feature_index = feature_index;
        tree_nodes[current_node]->threshold = column_blocks[feature_index]->thresholds[feature_threshold_index];
        tree_nodes[current_node]->left_default = feature_left_default;
      }
      change_node_parameters_mutex.unlock();
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }
  
  return best_gain;
}

// returns node to split, changes the threshold internally, and we will update based on that
std::size_t DecisionTree::GetNodeToSplit(std::size_t& threshold_index) {
  std::mutex maximum_score_mutex;
  float maximum_score = 0;
  std::size_t node_to_split = tree_nodes_size;
  std::vector<std::thread> threads;

  for (std::size_t current_node : leaf_indices) {
    threads.emplace_back([&, current_node]() {
      std::size_t node_threshold_index = 0;
      float current_score = CalculateGainsInNode(current_node, node_threshold_index);
      maximum_score_mutex.lock();
      if (current_score > maximum_score) {
        maximum_score = current_score;
        node_to_split = current_node;
        threshold_index = node_threshold_index;
      }
      maximum_score_mutex.unlock();
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  return node_to_split;
}

float DecisionTree::CalculateWeights(float gradients, float hessians) {
  // this epsilon is just to ensure numerical stability, so we don't divide by 0 if lambda is 0
  const float epsilon = 1e-8;
  return -gradients / (hessians + lambda + epsilon);
}

bool DecisionTree::PerformSplit() {
  // std::size_t node_to_split = FindBestSplit();
  std::size_t threshold_index = 0;
  std::size_t node_to_split = GetNodeToSplit(threshold_index);
  if (node_to_split >= last_layer_left_index) {
    return false;
  }

  std::size_t left_index = node_to_split * 2 + 1;
  std::size_t right_index = left_index + 1;
  tree_nodes[left_index] = std::make_unique<TreeNode>(left_index);
  tree_nodes[right_index] = std::make_unique<TreeNode>(right_index);

  auto it = std::find(leaf_indices.begin(), leaf_indices.end(), node_to_split);
  leaf_indices.erase(it);
  leaf_indices.push_back(left_index);
  leaf_indices.push_back(right_index);

  std::size_t feature_index = tree_nodes[node_to_split]->feature_index;
  float threshold = tree_nodes[node_to_split]->threshold;

  float left_gradients = 0;
  float left_hessians = 0;
  float right_gradients = 0;
  float right_hessians = 0;

  // assign active rows present in column block
  std::vector<std::size_t> update_list;
  for (std::size_t row_index = 0; row_index < data_row_information->num_rows; row_index++) {
    if (data_row_information->node_indices[row_index] == node_to_split) {
      update_list.push_back(row_index);
      // non-missing values
      if (column_blocks[feature_index]->bin_index[row_index] != -1) {
        if (column_blocks[feature_index]->bin_index[row_index] <= threshold_index) {
          data_row_information->node_indices[row_index] = left_index;
          left_gradients += data_row_information->gradients[row_index];
          left_hessians += data_row_information->hessians[row_index];
        } else {
          data_row_information->node_indices[row_index] = right_index;
          right_gradients += data_row_information->gradients[row_index];
          right_hessians += data_row_information->hessians[row_index];
        }

      // missing values
      } else {
        if (tree_nodes[node_to_split]->left_default) {
          data_row_information->node_indices[row_index] = left_index;
          left_gradients += data_row_information->gradients[row_index];
          left_hessians += data_row_information->hessians[row_index];
        } else {
          data_row_information->node_indices[row_index] = right_index;
          right_gradients += data_row_information->gradients[row_index];
          right_hessians += data_row_information->hessians[row_index];
        }
      }
    }
  }

  tree_nodes[left_index]->weights = CalculateWeights(left_gradients, left_hessians);
  tree_nodes[right_index]->weights = CalculateWeights(right_gradients, right_hessians);

  for (const auto& row_index : update_list) {
    if (data_row_information->node_indices[row_index] == left_index) {
      data_row_information->logits[row_index] += learning_rate * tree_nodes[left_index]->weights;
    } else {
      data_row_information->logits[row_index] += learning_rate * tree_nodes[right_index]->weights;
    }
  }

  CalculateGradientsAndHessians(update_list);

  return true;
}

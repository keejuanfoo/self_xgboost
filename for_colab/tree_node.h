#pragma once

#include <vector>
#include <fstream>

struct TreeNode {
  std::size_t node_index;
  std::size_t left_index;
  std::size_t right_index;
  std::size_t feature_index;
  float threshold;
  bool left_default = true;
  float weights; // how much this node changes the logits (times learning rate)

  explicit TreeNode(std::size_t node_index);

  void SerialiseNode(std::ofstream& weights_file);

  void DeserialiseNode(std::ifstream& weights_file);

  std::size_t SendToNextNode(std::vector<std::vector<float>>& features, std::size_t i);
};

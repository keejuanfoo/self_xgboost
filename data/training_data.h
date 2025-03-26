#pragma once

#include <vector>

struct TrainingDataRowInformation {
  std::vector<float> gradients;
  std::vector<float> hessians;
  std::vector<std::size_t> indices;
  std::vector<std::size_t> node_indices;
  std::vector<float> logits;
  std::size_t num_rows;

  explicit TrainingDataRowInformation(std::size_t num_rows);
};

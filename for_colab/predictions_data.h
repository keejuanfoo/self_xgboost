#pragma once

#include <vector>

struct PredictionDataRowInformation {
  std::vector<std::vector<float>> features;
  std::vector<std::size_t> indices;
  std::vector<std::size_t> node_indices;
  std::vector<float> logits;
  std::size_t num_features;
  std::size_t num_rows;

  explicit PredictionDataRowInformation(
    std::vector<std::vector<float>>& features, 
    std::size_t num_features, 
    std::size_t num_rows
  );
};

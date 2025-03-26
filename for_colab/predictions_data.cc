#include "predictions_data.h"

#include <numeric>

PredictionDataRowInformation::PredictionDataRowInformation(
  std::vector<std::vector<float>>& features, 
  std::size_t num_features, 
  std::size_t num_rows
)
: features(features),
  num_features(num_features),
  num_rows(num_rows)
{
  indices.resize(num_rows);
  std::iota(indices.begin(), indices.end(), 0);
  node_indices.resize(num_rows, 0);
  logits.resize(num_rows, 0);
}

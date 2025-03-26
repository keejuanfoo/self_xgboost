#include "training_data.h"

#include <numeric>

TrainingDataRowInformation::TrainingDataRowInformation(std::size_t num_rows) : num_rows(num_rows) {
  gradients.resize(num_rows);
  hessians.resize(num_rows);
  indices.resize(num_rows);
  std::iota(indices.begin(), indices.end(), 0);
  node_indices.resize(num_rows, 0);
  logits.resize(num_rows, 0);
}

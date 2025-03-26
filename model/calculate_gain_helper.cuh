#pragma once

void LaunchBinGeneratorKernel(
  const std::size_t num_rows,
  const std::size_t* node_indices,
  const std::size_t current_node,
  const int wqs_splits,
  const int* bin_indices,
  float* bin_gradients,
  float* bin_hessians,
  const float* gradients,
  const float* hessians,
  float* total_gradient,
  float* total_hessian,
  float* missing_gradients,
  float* missing_hessians,
  int* missing_value_flag
);

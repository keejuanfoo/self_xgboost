#include "calculate_gain_helper.cuh"

#include <iostream>

__global__ void BinGeneratorKernel(
  const std::size_t num_rows,
  const std::size_t* d_node_indices,
  const std::size_t current_node,
  const int* d_bin_indices, 
  float* d_bin_gradients, 
  float* d_bin_hessians, 
  const float* d_gradients, 
  const float* d_hessians,
  float* d_total_gradient, 
  float* d_total_hessian, 
  float* d_missing_gradients, 
  float* d_missing_hessians,
  int* d_missing_value_flag
) {
  std::size_t row_index = blockDim.x * blockIdx.x + threadIdx.x;
  if (row_index >= num_rows) {
    return;
  }

  if (d_node_indices[row_index] != current_node) {
    return;
  }
  int current_bin_index = d_bin_indices[row_index];
  float current_gradient = d_gradients[row_index];
  float current_hessian = d_hessians[row_index];

  if (current_bin_index != -1) {
    atomicAdd(&d_bin_gradients[current_bin_index], current_gradient);
    atomicAdd(&d_bin_hessians[current_bin_index], current_hessian);
    atomicAdd(d_total_gradient, current_gradient);
    atomicAdd(d_total_hessian, current_hessian);
  } else {
    atomicAdd(d_missing_gradients, current_gradient);
    atomicAdd(d_missing_hessians, current_hessian);
    atomicExch(d_missing_value_flag, 1);
  }
}

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
) {
  std::size_t *d_node_indices;
  int *d_bin_indices;
  float *d_bin_gradients, *d_bin_hessians;
  float *d_gradients, *d_hessians; 
  float *d_total_gradient, *d_total_hessian, *d_missing_gradients, *d_missing_hessians;
  int *d_missing_value_flag;

  cudaMalloc(&d_node_indices, num_rows * sizeof(std::size_t));
  cudaMalloc(&d_bin_indices, num_rows * sizeof(int));
  cudaMalloc(&d_bin_gradients, wqs_splits * sizeof(float));
  cudaMalloc(&d_bin_hessians, wqs_splits * sizeof(float));
  cudaMalloc(&d_gradients, num_rows * sizeof(float));
  cudaMalloc(&d_hessians, num_rows * sizeof(float));
  cudaMalloc(&d_total_gradient, sizeof(float));
  cudaMalloc(&d_total_hessian, sizeof(float));
  cudaMalloc(&d_missing_gradients, sizeof(float));
  cudaMalloc(&d_missing_hessians, sizeof(float));
  cudaMalloc(&d_missing_value_flag, sizeof(int));

  cudaMemcpy(d_node_indices, node_indices, num_rows * sizeof(std::size_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bin_indices, bin_indices, num_rows * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bin_gradients, bin_gradients, wqs_splits * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bin_hessians, bin_hessians, wqs_splits * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_gradients, gradients, num_rows * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_hessians, hessians, num_rows * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_total_gradient, total_gradient, sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_total_hessian, total_hessian, sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_missing_gradients, missing_gradients, sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_missing_hessians, missing_hessians, sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_missing_value_flag, missing_value_flag, sizeof(int), cudaMemcpyHostToDevice);

  int threadsPerBlock = 256;
  int blocksPerGrid = (num_rows + threadsPerBlock - 1) / threadsPerBlock;
  BinGeneratorKernel<<<blocksPerGrid, threadsPerBlock>>>(
    num_rows, d_node_indices, current_node, d_bin_indices, d_bin_gradients, d_bin_hessians, d_gradients, d_hessians, 
    d_total_gradient, d_total_hessian, d_missing_gradients, d_missing_hessians, d_missing_value_flag
  );

  cudaMemcpy(bin_gradients, d_bin_gradients, wqs_splits * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(bin_hessians, d_bin_hessians, wqs_splits * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(total_gradient, d_total_gradient, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(total_hessian, d_total_hessian, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(missing_gradients, d_missing_gradients, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(missing_hessians, d_missing_hessians, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(missing_value_flag, d_missing_value_flag, sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(d_node_indices);
  cudaFree(d_bin_indices);
  cudaFree(d_bin_gradients);
  cudaFree(d_bin_hessians);
  cudaFree(d_gradients);
  cudaFree(d_hessians);
  cudaFree(d_total_gradient);
  cudaFree(d_total_hessian);
  cudaFree(d_missing_gradients);
  cudaFree(d_missing_hessians);
  cudaFree(d_missing_value_flag);
}

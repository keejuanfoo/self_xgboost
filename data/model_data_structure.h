#pragma once

#include <vector>
#include <memory>

struct ColumnBlock {
  std::vector<int> bin_index;
  std::vector<float> thresholds;

  ColumnBlock(std::size_t num_rows, int wqs_splits);
};

struct FeatureBlock {
  std::vector<float> values;
  std::vector<std::size_t> indices;
};

class Data {
public:
  std::vector<std::unique_ptr<FeatureBlock>> feature_blocks;
  std::size_t num_features;
  std::size_t num_rows = 0;

  explicit Data(std::size_t num_columns);
  void AddRow(const std::vector<float>& row_values);
  void SortFeatureBlocks();
};

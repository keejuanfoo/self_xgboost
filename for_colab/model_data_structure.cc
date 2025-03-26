#include "model_data_structure.h"

#include <vector>
#include <algorithm>
#include <cmath>

ColumnBlock::ColumnBlock(std::size_t num_rows, int wqs_splits) {
  bin_index.resize(num_rows);
  thresholds.resize(wqs_splits - 1);
}

Data::Data(std::size_t num_columns) {
  num_features = num_columns;
  feature_blocks.resize(num_columns);
  std::generate(feature_blocks.begin(), feature_blocks.end(), []() {
    return std::make_unique<FeatureBlock>();
  });
}

void Data::AddRow(const std::vector<float>& row_values) {
  for (std::size_t column_index = 0; column_index < num_features; column_index++) {
    if (!std::isnan(row_values[column_index])) {
      feature_blocks[column_index]->values.push_back(row_values[column_index]);
      feature_blocks[column_index]->indices.push_back(num_rows);
    }
  }
  num_rows++;
}

void Data::SortFeatureBlocks() {
  for (auto& feature_block : feature_blocks) {
    std::vector<std::size_t> indices_sorted = feature_block->indices;

    std::sort(indices_sorted.begin(), indices_sorted.end(), [&](std::size_t a, std::size_t b) {
      if (feature_block->values[a] == feature_block->values[b]) {
        return a < b;
      }
      return feature_block->values[a] < feature_block->values[b];
    });

    std::vector<float> values_sorted(feature_block->values.size());
    for (std::size_t i = 0; i < indices_sorted.size(); i++) {
      values_sorted[i] = feature_block->values[indices_sorted[i]];
    }

    feature_block->values = std::move(values_sorted);
    feature_block->indices = std::move(indices_sorted);
  }
}

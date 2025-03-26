#include "tree_node.h"

#include <cmath>

TreeNode::TreeNode(std::size_t node_index) : node_index(node_index) {
	left_index = node_index * 2 + 1;
	right_index = left_index + 1;
}

void TreeNode::SerialiseNode(std::ofstream& weights_file) {
	weights_file.write(reinterpret_cast<const char*>(&left_index), sizeof(left_index));
	weights_file.write(reinterpret_cast<const char*>(&right_index), sizeof(right_index));
	weights_file.write(reinterpret_cast<const char*>(&feature_index), sizeof(feature_index));
	weights_file.write(reinterpret_cast<const char*>(&threshold), sizeof(threshold));
	weights_file.write(reinterpret_cast<const char*>(&left_default), sizeof(left_default));
	weights_file.write(reinterpret_cast<const char*>(&weights), sizeof(weights));
}

void TreeNode::DeserialiseNode(std::ifstream& weights_file) {
	weights_file.read(reinterpret_cast<char*>(&left_index), sizeof(left_index));
	weights_file.read(reinterpret_cast<char*>(&right_index), sizeof(right_index));
	weights_file.read(reinterpret_cast<char*>(&feature_index), sizeof(feature_index));
	weights_file.read(reinterpret_cast<char*>(&threshold), sizeof(threshold));
	weights_file.read(reinterpret_cast<char*>(&left_default), sizeof(left_default));
	weights_file.read(reinterpret_cast<char*>(&weights), sizeof(weights));
}

std::size_t TreeNode::SendToNextNode(std::vector<std::vector<float>>& features, std::size_t i) {
	if (std::isnan(features[feature_index][i])) {
		return (left_default) ? left_index : right_index;
	} else {
		return (features[feature_index][i] < threshold) ? left_index : right_index;
	}
}

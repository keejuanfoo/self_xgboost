#pragma once

#include <vector>

constexpr float LOG2_E = 1.4426950408889634f;

float Sigmoid(float predicted);

// predicted is a logit
float CalculateGradient(float predicted, float actual);

float CalculateHessian(float predicted);

#include "math_utils.h"

#include <cmath>


float Sigmoid(float predicted) {
  return 1.0 / (1 + exp(-predicted));
}

// predicted is a logit
float CalculateGradient(float predicted, float actual) {
  return Sigmoid(predicted) - actual;
}

float CalculateHessian(float predicted) {
  return Sigmoid(predicted) * (1 - Sigmoid(predicted));
}

#pragma once

#include <vector>
#include <arm_neon.h>

constexpr float LOG2_E = 1.4426950408889634f;

float Sigmoid(float predicted);

// predicted is a logit
float CalculateGradient(float predicted, float actual);

float CalculateHessian(float predicted);

float32x4_t ExponentialSimd(float32x4_t x);

float32x4_t SigmoidSimd(float32x4_t predicted);

float32x4_t CalculateGradientSimd(float32x4_t predicted, float32x4_t actual);

float32x4_t CalculateHessianSimd(float32x4_t predicted);

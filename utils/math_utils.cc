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

// NEON SIMD doesn't support this; this is an approximation: exp(x) = 2^(x * log2(e))
float32x4_t ExponentialSimd(float32x4_t x) {
  // clip x to a stable bound
  x = vmaxq_f32(vdupq_n_f32(-15.0f), vminq_f32(x, vdupq_n_f32(15.0f)));
  float32x4_t exponent = vmulq_f32(x, vdupq_n_f32(LOG2_E));
  
  /* 
   * To calculate 2^x, we separate x into an integer and a fraction
   * The integer part can be easily resolved by bit manipulation (left shifts)
   * For the fraction part, I'm using a best-fit (minimax fitting) 3rd degree polynomial suggested by GPT
   */
  int32x4_t integer_exponent = vcvtq_s32_f32(exponent);
  // convert to float so we can perform greater than operation
  float32x4_t integer_exponent_float = vcvtq_f32_s32(integer_exponent);
  // Original exponent is expected to be greater, those will be marked as 1
  uint32x4_t fix_conversion = vcgtq_f32(integer_exponent_float, exponent);
  // Deduct 1 from those that need to be fixed
  integer_exponent = vsubq_s32(integer_exponent, vreinterpretq_s32_u32(fix_conversion));
  float32x4_t fraction_exponent = vsubq_f32(exponent, vcvtq_f32_s32(integer_exponent));

  // Hardcode the polynomial coefficients: 2^f = a + bf + cf^2 + df^3 = a + f(b + f(c + fd))
  float32x4_t a = vdupq_n_f32(1.0f);
  float32x4_t b = vdupq_n_f32(0.69314718f);
  float32x4_t c = vdupq_n_f32(0.227411278f);
  float32x4_t d = vdupq_n_f32(0.079440238f);

  float32x4_t polynomial = vmlaq_f32(c, fraction_exponent, d); // c + fraction_exponent * d
  polynomial = vmlaq_f32(b, fraction_exponent, polynomial);
  polynomial = vmlaq_f32(a, fraction_exponent, polynomial);

  // float32: [ sign (1 bit) | exponent (8 bits) | mantissa (23 bits) ]
  int32x4_t integer_term = vshlq_n_s32(vaddq_s32(integer_exponent, vdupq_n_s32(127)), 23);
  float32x4_t result = vmulq_f32(vreinterpretq_f32_s32(integer_term), polynomial);

  return result;
}

float32x4_t SigmoidSimd(float32x4_t predicted) {
  // float32x4_t result = vdivq_f32(vdupq_n_f32(1.0), vaddq_f32(vdupq_n_f32(1.0), ExponentialSimd(vnegq_f32(predicted))));
  float32x4_t result = vdivq_f32(ExponentialSimd(predicted), vaddq_f32(vdupq_n_f32(1.0), ExponentialSimd(predicted)));
  // clamp result to the range [0, 1]
  result = vmaxq_f32(vdupq_n_f32(0.0f), vminq_f32(result, vdupq_n_f32(1.0f)));

  return result;
}

float32x4_t CalculateGradientSimd(float32x4_t predicted, float32x4_t actual) {
  return vsubq_f32(SigmoidSimd(predicted), actual);
}

float32x4_t CalculateHessianSimd(float32x4_t predicted) {
  return vmulq_f32(SigmoidSimd(predicted), vsubq_f32(vdupq_n_f32(1.0), SigmoidSimd(predicted)));
}

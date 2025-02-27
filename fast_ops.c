// fast_ops.c
#include <stddef.h>

// A simple ReLU function: applies ReLU element-wise on an array of floats.
void fast_relu(const float* input, float* output, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float val = input[i];
        output[i] = (val > 0.0f) ? val : 0.0f;
    }
}

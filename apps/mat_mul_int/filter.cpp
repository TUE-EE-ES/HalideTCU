#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "Halide.h"
#include "HalideBuffer.h"
#include "benchmark_util.h"
#include "mat_mul.h"
#include "mat_mul_tensor.h"
#include "transpose.h"
#include <assert.h>
#include <cuda.h>
#include <mma.h>
#include <stdio.h>

using namespace std;
using namespace Halide;
void init_host_matrices(uint8_t *a, uint8_t *b, int32_t *c, int size) {
  int M_GLOBAL = size;
  int N_GLOBAL = size;
  int K_GLOBAL = size;
  for (int i = 0; i < M_GLOBAL; i++) {
    for (int j = 0; j < K_GLOBAL; j++) {
      a[i * K_GLOBAL + j] = static_cast<uint8_t>(rand() % 64);
    }
  }

  for (int i = 0; i < N_GLOBAL; i++) {
    for (int j = 0; j < K_GLOBAL; j++) {
      b[i * K_GLOBAL + j] = static_cast<uint8_t>(rand() % 32);
    }
  }

  for (int t = 0; t < M_GLOBAL * N_GLOBAL; t++) {
    c[t] = static_cast<int32_t>(0);
  }
}

void matMultiplyOnHost(uint8_t *A, uint8_t *B, uint8_t *C, int32_t alpha,
                       float beta, int numARows, int numAColumns, int numBRows,
                       int numBColumns, int numCRows, int numCColumns) {
  for (int i = 0; i < numCRows; i++) {
    for (int j = 0; j < numCColumns; j++) {
      float temp = 0.0;

      for (int k = 0; k < numAColumns; k++) {
        temp += (float)A[i * numARows + k] * (float)B[k * numBRows + j];
      }

      C[i * numCColumns + j] = temp * alpha + beta * C[i * numCColumns + j];
    }
  }
}

int main(int argc, char **argv) {

  int matrix_size = 1024;
  if (argc > 1) {
    matrix_size = atoi(argv[1]);
  }
  int B_transpose = 0;
  if (argc > 2) {
    B_transpose = atoi(argv[2]);
  }
  Halide::Runtime::Buffer<uint8_t> mat_A(matrix_size, matrix_size);
  Halide::Runtime::Buffer<uint8_t> mat_B(matrix_size, matrix_size);
  Halide::Runtime::Buffer<float16_t> mat_BT(matrix_size, matrix_size);
  Halide::Runtime::Buffer<int32_t> output(matrix_size, matrix_size);
  Halide::Runtime::Buffer<int32_t> outputt(matrix_size, matrix_size);
  Halide::Runtime::Buffer<int32_t> output_cuda(matrix_size, matrix_size);
  init_host_matrices(&mat_A(0, 0), &mat_B(0, 0), &output(0, 0), matrix_size);

  halide_reuse_device_allocations(nullptr, true);
  halide_reuse_device_allocations(nullptr, true);
  mat_mul_tensor(mat_A, mat_B, outputt);
  if (B_transpose) {
    multi_way_bench({
        {"Manual",
         [&]() {
           mat_mul(mat_A, mat_B, output);
           output.device_sync();
         }},
        {"Auto-tensor",
         [&]() {
           transpose(mat_B, mat_BT);
           mat_mul_tensor(mat_A, mat_BT, outputt);
           outputt.device_sync();
         }},
    });
  }
  if (!B_transpose) {

    double t = Halide::Tools::benchmark(100, 100, [&]() {
      mat_mul_tensor(mat_A, mat_B, outputt);
      outputt.device_sync();
    });
    printf("Auto-tensor time: %f\n", t * 1000);
  }

#if 1
  {

    int size = matrix_size;
    uint8_t *A = NULL;
    uint8_t *B = NULL;
    int *C = NULL;

    cudaMalloc(reinterpret_cast<void **>(&A), sizeof(int8_t) * size * size);
    cudaMalloc(reinterpret_cast<void **>(&B), sizeof(int8_t) * size * size);
    cudaMalloc(reinterpret_cast<void **>(&C), sizeof(int32_t) * size * size);

    cudaMemcpy(A, &mat_A(0, 0), sizeof(uint8_t) * size * size,
               cudaMemcpyHostToDevice);
    cudaMemcpy(B, &mat_B(0, 0), sizeof(uint8_t) * size * size,
               cudaMemcpyHostToDevice);
    cudaMemcpy(C, &output_cuda(0, 0), sizeof(int32_t) * size * size,
               cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);
    int alpha = 1, beta = 0;
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    double t = Halide::Tools::benchmark(10, 100, [&]() {
      if (!B_transpose) {
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, size, size, size, &alpha,
                     B, CUDA_R_8I, size, A, CUDA_R_8I, size, &beta, C,
                     CUDA_R_32I, size, CUDA_R_32I,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      } else {
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, size, size, size, &alpha,
                     B, CUDA_R_8I, size, A, CUDA_R_8I, size, &beta, C,
                     CUDA_R_32I, size, CUDA_R_32I,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      }
      cudaDeviceSynchronize();
    });

    cudaMemcpy(&output_cuda(0, 0), C, sizeof(int32_t) * size * size,
               cudaMemcpyDeviceToHost);
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cublasDestroy(handle);
    printf("cublas time: %f\n", t * 1000);
  }
#endif
  output.copy_to_host();
  outputt.copy_to_host();
#if 1
  float max = 0.0f;
  int sum = 0;

  for (int iy = 0; iy < matrix_size; iy++) {
    for (int ix = 0; ix < matrix_size; ix++) {
      if (fabs(output_cuda(ix, iy) - outputt(ix, iy)) > 0) {

        sum++;
      }
    }
  }

  int *tensor = &outputt(0, 0);
  int *core = &output_cuda(0, 0);
  for (int i = 0; i < matrix_size * matrix_size; i++) {
    if (fabs(tensor[i] - core[i]) > max) {
      max = fabs(tensor[i] - core[i]);
    }
  }
  std::cout << "sum err " << sum << " " << max << std::endl;
#endif
  return 0;
}

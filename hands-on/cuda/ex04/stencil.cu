// C++ standard headers
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
#include <cassert>

// CUDA headers
#include <cuda_runtime.h>

// local headers
#include "cuda_check.h"

// Here you can set the device ID that was assigned to you
#define MYDEVICE 2
#define BLOCK_SIZE 256
#define RADIUS 3

__global__ void stencil_1d(const int *in, int *out, int n)
{
  __shared__ int tmp[BLOCK_SIZE + 2 * RADIUS];
  auto g_index = threadIdx.x + blockIdx.x * blockDim.x;
  auto s_index = threadIdx.x + RADIUS;

  // Read input elements into shared memory
  tmp[s_index] = in[g_index];
  if (threadIdx.x < RADIUS)
  {
    tmp[s_index - RADIUS] = g_index - RADIUS < 0 ? 0 : in[g_index - RADIUS];
    tmp[s_index + BLOCK_SIZE] = g_index + BLOCK_SIZE < n ? in[g_index + BLOCK_SIZE] : 0;
  }
  __syncthreads();
  int result = 0;
  for (int offset = -RADIUS; offset <= RADIUS; offset++)
  {
    result += tmp[s_index + offset];
  }
  out[g_index] = result;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(void)
{
  cudaSetDevice(MYDEVICE);
  cudaStream_t queue;
  cudaStreamCreate(&queue);

  std::random_device rd;  // Will be used to obtain a seed for the random engine
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<> distrib(-10, 10);
  // Create array of 256ki elements
  const int num_elements = 1 << 18;
  // Generate random input on the host
  std::vector<int> h_input(num_elements);
  for (auto &elt : h_input)
  {
    elt = distrib(gen);
  }
  std::vector<int> h_output(num_elements, 0);

  // Allocate memory on GPU
  int *d_input, *d_output;
  cudaMalloc(&d_input, num_elements * sizeof(int));
  cudaMalloc(&d_output, num_elements * sizeof(int));

  // Copy input to GPU
  cudaMemcpyAsync(d_input, h_input.data(), num_elements * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_output, h_output.data(), num_elements * sizeof(int), cudaMemcpyHostToDevice);

  // Launch stencil_1d() kernel on GPU
  stencil_1d<<<(num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, queue>>>(d_input, d_output, num_elements);

  // Copy output to CPU
  cudaMemcpyAsync(h_output.data(), d_output, num_elements * sizeof(int), cudaMemcpyDeviceToHost);

  cudaFreeAsync(d_input, queue);
  cudaFreeAsync(d_output, queue);
  cudaStreamSynchronize(queue);

  // Perform stencil operation on CPU
  std::vector<int> h_verify(num_elements);
  for (int i = 0; i < num_elements; ++i)
  {
    int result = 0;
    for (int offset = -RADIUS; offset <= RADIUS; ++offset)
    {
      int index = i + offset;
      if (index >= 0 && index < num_elements)
      {
        result += h_input[index];
      }
    }
    h_verify[i] = result;
  }

  // print out the first 10 results
  for (int i = 0; i < 10; ++i)
  {
    std::cout << "CPU: " << h_verify[i] << ", GPU: " << h_output[i] << std::endl;
  }

  return 0;
}

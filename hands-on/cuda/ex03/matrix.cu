// C++ standard headers
#include <cassert>
#include <iostream>
#include <vector>

// CUDA headers
#include <cuda_runtime.h>

// local headers
#include "cuda_check.h"

// Here you can set the device ID that was assigned to you
#define MYDEVICE 2

// Fill matrix incrementally. Kernel should be able to work for any matrix size.
// Blocks made of 8x8 threads. Compute the needed blocks along the x and y axis. Make sure indexes are within axes sizes
// threadId.x and blockId.x are dim3 types -> structure with x, y, z fields: kernel<<(5,2),(7,7),0>>.

// Part 2 of 4: implement the kernel
__global__ void kernel(int *a, int dimx, int dimy)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int idy = threadIdx.y + blockIdx.y * blockDim.y;
  if (idx < dimx && idy < dimy)
  {
    a[idy * dimx + idx] = idy * dimx + idx;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main()
{
  CUDA_CHECK(cudaSetDevice(MYDEVICE));

  // Create a CUDA stream to execute asynchronous operations on this device
  cudaStream_t queue;
  CUDA_CHECK(cudaStreamCreate(&queue));

  // Part 1 and 4 of 4: set the dimensions of the matrix
  int dimx = 69;
  int dimy = 42;

  // Allocate enough memory on the host
  std::vector<int> h_a(dimx * dimy, 0);
  int num_bytes = dimx * dimy * sizeof(int);

  // Pointer for the device memory
  int *d_a;

  // Allocate enough memory on the device
  CUDA_CHECK(cudaMallocAsync(&d_a, num_bytes, queue));

  // Part 2 of 4: define grid and block size and launch the kernel
  dim3 numberOfBlocks, numberOfThreadsPerBlock;
  numberOfThreadsPerBlock.x = 8;
  numberOfThreadsPerBlock.y = 4;
  numberOfBlocks.x = ceil(static_cast<double>(dimx) / static_cast<double>(numberOfThreadsPerBlock.x));
  numberOfBlocks.y = ceil(static_cast<double>(dimy) / static_cast<double>(numberOfThreadsPerBlock.y));

  std::cout << "Blocks: " << numberOfBlocks.x << " " << numberOfBlocks.y << std::endl;

  // CUDA_CHECK(cudaMemcpyAsync(d_a, h_a.data(), num_bytes, cudaMemcpyHostToDevice, queue));

  kernel<<<numberOfBlocks, numberOfThreadsPerBlock, 0, queue>>>(d_a, dimx, dimy);
  CUDA_CHECK(cudaGetLastError());

  // Device to host copy
  CUDA_CHECK(cudaMemcpyAsync(h_a.data(), d_a, num_bytes, cudaMemcpyDeviceToHost, queue));

  // Free the device memory
  CUDA_CHECK(cudaFreeAsync(d_a, queue));

  // Wait for all asynchronous operations to complete
  CUDA_CHECK(cudaStreamSynchronize(queue));

  // verify the data returned to the host is correct
  std::cout << "Bytes: " << num_bytes << std::endl;
  for (int row = 0; row < dimy; ++row)
  {
    for (int col = 0; col < dimx; ++col)
    {
      // std::cout << h_a[row * dimx + col] << " " << row * dimx + col << " ";
      assert(h_a[row * dimx + col] == row * dimx + col);
    }
    // std::cout << std::endl;
  }

  // Destroy the CUDA stream
  CUDA_CHECK(cudaStreamDestroy(queue));

  // If the program makes it this far, then the results are correct and
  // there are no run-time errors.  Good work!
  std::cout << "Correct!" << std::endl;

  return 0;
}

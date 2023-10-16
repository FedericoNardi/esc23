#include <iostream>
// Add an empty kernel
__global__ void mykernel()
{
}

int main()
{
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    // launch the kernel on the stream
    mykernel<<<1, 1, 0, stream>>>(); // kernel launch is asynchronous << _#_blocks_, _#_threads_per_block, ...>>
    std::cout << "Hello, world!\n";
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}
#include <iostream>
#include <math.h>
 

__global__
void add(int n, float *x, float *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  // 当前线程在整个 grid 里的全局编号
  int stride = blockDim.x * gridDim.x;
  // 这里说的是整个grid中一共有多少个thread
  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}


// 这是 host code
int main(void)
{
 int N = 1<<20;
 float *x, *y;
 
 // Allocate Unified Memory – accessible from CPU or GPU
 cudaMallocManaged(&x, N*sizeof(float));
 cudaMallocManaged(&y, N*sizeof(float));
// 用cudaMallocManaged(&x, N*sizeof(float)); 代替 float *x = new float[N];
// 返回的是指针

 
 // initialize x and y arrays on the host
 for (int i = 0; i < N; i++) {
   x[i] = 1.0f;
   y[i] = 2.0f;
 }

  // 预取到 GPU 0
  cudaMemPrefetchAsync(x, N * sizeof(float), 0, 0);
  cudaMemPrefetchAsync(y, N * sizeof(float), 0, 0);
  // 提前把数据搬到 GPU,可以大大加速，节省GPU算力等数据的时间

 int blockSize = 256;
 int numBlocks = (N + blockSize - 1) / blockSize;
 add<<<numBlocks, blockSize>>>(N, x, y);

 // Wait for GPU to finish before accessing on host
 cudaDeviceSynchronize();
 // 等待GPU完成计算，确保数据同步

 // Check for errors (all values should be 3.0f)
 float maxError = 0.0f;
 for (int i = 0; i < N; i++) {
   maxError = fmax(maxError, fabs(y[i]-3.0f));
 }
 std::cout << "Max error: " << maxError << std::endl;
 
 // Free memory
 cudaFree(x);
//用 cudaFree(x)代替 delete [] x
 cudaFree(y);
  return 0;
}

/*
保存为.cu文件
> nvcc add.cu -o add_cuda
> ./add_cuda
Max error: 0.000000
*/
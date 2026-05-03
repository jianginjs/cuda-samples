#include <iostream>
#include <math.h>
 
// Kernel function to add the elements of two arrays
// 这是 device code


/*
// 这是 add<<<1, 1>>>(N, x, y);可以用的
__global__
void add(int n, float *x, float *y)
{
 for (int i = 0; i < n; i++)
   y[i] = x[i] + y[i];
}
*/

__global__
void add(int n, float *x, float *y)
{
  int index = threadIdx.x;
  int stride = blockDim.x;
  for (int i = index; i < n; i += stride)
      y[i] = x[i] + y[i];
}
 // 注意：这个版本只用了 一个 block 内的线程
 // threadIdx.x contains the index of the current thread within its block, and blockDim.x contains the number of threads in the block
 // threadIdx.x的范围是0，1，2...threadIdx.x-1，
 // blockDim.x 当前 block 在 x 方向上一共有多少个线程
 /*
  threadIdx.x = 0 的线程：
  i = 0, 256, 512, 768, ...

  threadIdx.x = 1 的线程：
  i = 1, 257, 513, 769, ...
  
  threadIdx.x = 2 的线程：
  i = 2, 258, 514, 770, ...
  ...
  threadIdx.x = 255 的线程：
  i = 255, 511, 767, 1023, ...
 */
 


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
 
 // Run kernel on 1M elements on the GPU
//  add<<<1, 1>>>(N, x, y);
 // CUDA上的kernel调用方式
 // <<<1,1>>> 单个thread
 // <<<blocks,threads>>>是告诉CUDA runtime在GPU上启动多少线程，每个线程做一次计算
 // threads一般是32整数倍，如add<<<1,256>>>(N,x,y)

add<<<1, 256>>>(N, x, y);
// 适配上面新add kernel
 
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
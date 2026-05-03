#include <iostream>
#include <math.h>
 
// function to add the elements of two arrays
void add(int n, float *x, float *y)
{
 for (int i = 0; i < n; i++)
     y[i] = x[i] + y[i];
}
 
int main(void)
{
 int N = 1<<20; // 1M elements
 // a << b 等价于 a * 2^b
 float *x = new float[N];
 // new 是动态内存分配 对应后面的delete释放, new分配的是堆内存，可以很大
 float *y = new float[N];
 
 // initialize x and y arrays on the host
 for (int i = 0; i < N; i++) {
   x[i] = 1.0f;
   y[i] = 2.0f;
 }
 
 // Run kernel on 1M elements on the CPU
 add(N, x, y);
 //这里调用函数，用的是地址x和y，
 
 // Check for errors (all values should be 3.0f)
 float maxError = 0.0f;
 for (int i = 0; i < N; i++)
   maxError = fmax(maxError, fabs(y[i]-3.0f));
   // fabs 取差值的绝对值，fmax取最大值
 std::cout << "Max error: " << maxError << std::endl;
 
 // Free memory
 delete [] x;
 delete [] y;
 
 return 0;
}
![alt text](image.png)
gridDim.x 当前grid中block number
blockIdx.x 当前block在grid中的index
一维数组index = blockIdx.x * blockDim.x + threadIdx.x

SM 是 Streaming Multiprocessor，如某块 GPU 里面有 40 个计算单元集群

比如：add<<<numBlocks, blockSize>>>(N, x, y);
一个 block 会被完整地放到某一个 SM 上执行，不会拆到多个 SM 上。

### 对于grid，block，thread的概念：
举一个1D的简单例子：
N = 20
blockSize = 4
numBlocks = 3

那么：
gridDim.x  = 3
blockDim.x = 4
总线程数 stride = gridDim.x * blockDim.x =3 * 4 = 12

它们的全局 index 是：
block 0:
thread 0 -> index 0
thread 1 -> index 1
thread 2 -> index 2
thread 3 -> index 3

block 1:
thread 0 -> index 4
thread 1 -> index 5
thread 2 -> index 6
thread 3 -> index 7

block 2:
thread 0 -> index 8
thread 1 -> index 9
thread 2 -> index 10
thread 3 -> index 11

但是如果数据有 20 个，线程只有 12 个，所以每个线程通过 i += stride 继续往后跳。
所以所有线程的分工是：
线程全局 index 0:  处理 i = 0, 12
线程全局 index 1:  处理 i = 1, 13
线程全局 index 2:  处理 i = 2, 14
线程全局 index 3:  处理 i = 3, 15
线程全局 index 4:  处理 i = 4, 16
线程全局 index 5:  处理 i = 5, 17
线程全局 index 6:  处理 i = 6, 18
线程全局 index 7:  处理 i = 7, 19
线程全局 index 8:  处理 i = 8
线程全局 index 9:  处理 i = 9
线程全局 index 10: 处理 i = 10
线程全局 index 11: 处理 i = 11

### 在具体代码中
```cpp
__global__
void add(int n, float *x, float *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}
```
关于int i = index的具体取值
n = 20 的例子里面
index的取值范围是0到11，所以for 循环中i的取值范围是从0开始，具体如下：

```plaintext
index = 0:
  i = 0
  i = 0 + 12 = 12
  i = 12 + 12 = 24，超过 20，停止

index = 1:
  i = 1
  i = 13
  i = 25，停止

index = 2:
  i = 2
  i = 14
  i = 26，停止

index = 3:
  i = 3
  i = 15
  i = 27，停止

index = 4:
  i = 4
  i = 16
  i = 28，停止

index = 5:
  i = 5
  i = 17
  i = 29，停止

index = 6:
  i = 6
  i = 18
  i = 30，停止

index = 7:
  i = 7
  i = 19
  i = 31，停止

index = 8:
  i = 8
  i = 20，不满足 i < 20，停止

index = 9:
  i = 9
  i = 21，停止

index = 10:
  i = 10
  i = 22，停止

index = 11:
  i = 11
  i = 23，停止
```
也就是i取值0的时候，12 个线程的 i 分别是：
```
T0:  i = 0
T1:  i = 1
T2:  i = 2
T3:  i = 3
T4:  i = 4
T5:  i = 5
T6:  i = 6
T7:  i = 7
T8:  i = 8
T9:  i = 9
T10: i = 10
T11: i = 11
12个线程大致同时处理
y[0]  = x[0]  + y[0]
y[1]  = x[1]  + y[1]
...
y[11] = x[11] + y[11]
```

### 扩展到2维
记住下面的
```cpp
dim3 block(16, 16);
dim3 grid(
    (width + block.x - 1) / block.x,
    (height + block.y - 1) / block.y
);

int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
```
在二维数据处理中，99% 的代码默认约定是：
x → width（列）
y → height（行）
也就是先x再y

下面的这个例子里，比较简单，不涉及for循环，因为线程数量>元素个数，一次运算可以算完
```cpp
__global__ void matrixAdd(float* A, float* B, float* C, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // 这里知道元素在col也就是x上的编号
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    // 这里知道元素在row也就是y上的编号

    if (row < height && col < width) {
        int idx = row * width + col;
        // 重点看这个映射，根据上面的编号，计算得到在二维展开的一维序列中的编号
        C[idx] = A[idx] + B[idx];
    }
}
int width = 1024;
int height = 768;
dim3 block(16, 16);
dim3 grid(
    (width + block.x - 1) / block.x,
    (height + block.y - 1) / block.y
);
matrixAdd<<<grid, block>>>(A, B, C, width, height);
```
在二维里面，一个矩阵一般是row-major：
matrix[row][col]  →  array[row * width + col]

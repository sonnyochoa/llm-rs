// Implements matrix multiplication

// Notes: this is my current understanding of matrix multiplication in CUDA programming model.
// Computation is ordered in a three-level hierarchy.
// Each invoation of a CUDA kernel creates a new grid.
// Each grid contains multiple blocks.
// Each block consists of up to 1024 threads.
// Threads in the same block can access shared memory. (SMEM)
//
// The number of threads in a block can be configured using a varialble called "blockDim",
// which is a 3-tuple of integers.
// The entries fo that vector specify the size of: blockDim.x, blockDim.y, blockDim.z.
//
// The number of blocks in a grid can be configured using a varialble called "gridDim",
// 
// The thread hierarchy concerns program correctness.
// For program performance, it's not a good idea to treat all threads in the same block as identical.

// First kernel: we will use the grid, block and thread hierarcy to assign each thread a unique entry
// in the result matrix C.
// That thread will then compute the dot product of the corresponding row of A and the corresponding column of B,
// and store the result in the corresponding entry of C.
// 
// Due to each location of C is computed by a unique thread, we don't have to worry about synchronization issues.

// Kernel #1:
// create as many blocks as necessary to map all of C
// dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);
// 
// 32 * 32 = 1024 threads per block
// dim3 blockDim(32, 32, 1);
// 
// launch the asynchronous execution of the kernel on the device
// The function call returns immediately on the host
// sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);

// CUDA code is written from a single-threaded perspective. In the code of the kernel,
// we access the blockIdx and threadIdx variables. These will return different values for each thread.

// ```cuda
// __global__ void sgemm_naive(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C) {
//     // compute position in C that this thread is responsible for
//     const uint x = blockIdx.x * blockDim.x + threadIdx.x;
//     const uint y = blockIdx.y * blockDim.y + threadIdx.y;
//
//     // if condition is necessary for when M or N are not multiples of 32 (block size)
//     if (x < M && y < N) {
//       float tmp = 0.0
//       for (int i = 0; i < K; i++) {
//         tmp += A[x * K + i] * B[i * N + y];
//       }
//       // c = a*(A@B) + β*C
//       C[x * N + y] = alpha * tmp + beta * C[x * N + y];
//     }
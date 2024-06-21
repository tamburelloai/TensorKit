//
//  matMul.metal
//  TensorKit
//
//  Created by Michael Tamburello on 6/18/24.
//

#include <metal_stdlib>
using namespace metal;

/// Matrix multiplication between two tensors
/// `tensorA` and `tensorB` are passed to buffers (memory allocations on the GPU), and then
/// a thread is assigned to each of them. The results are stored in `tensorC`, which has its own buffer
/// and will be read back to the CPU to get our summed result.
kernel void matMul_Float(
                          const device float* tensorA [[buffer(0)]],
                          const device float* tensorB [[buffer(1)]],
                          device float* tensorC [[buffer(2)]],
                          constant uint& M [[buffer(3)]],
                          constant uint& K [[buffer(4)]],
                          constant uint& N [[buffer(5)]],
                          uint id [[thread_position_in_grid]])
{
  uint row = id / N;
  uint col = id % N;
  float sum = 0.0;
  for (uint k = 0; k < K; ++k) {
    sum += tensorA[row*K+k] * tensorB[k*N+col];
  }
  tensorC[row*N+col] = sum;
}

kernel void matMul_Int(
                          const device long* tensorA [[buffer(0)]],
                          const device long* tensorB [[buffer(1)]],
                          device long* tensorC [[buffer(2)]],
                          constant uint& M [[buffer(3)]],
                          constant uint& K [[buffer(4)]],
                          constant uint& N [[buffer(5)]],
                          uint id [[thread_position_in_grid]])
{
  uint row = id / N;
  uint col = id % N;
  long sum = 0;
  for (uint k = 0; k < K; ++k) {
    sum += tensorA[row*K+k] * tensorB[k*N+col];
  }
  tensorC[row*N+col] = sum;
}


//
//// Define tile size
//constant uint TILE_SIZE = 16;
//
//// The kernel function to perform matrix multiplication with tiling
//kernel void tiledMatMul_Float(
//    device float* A [[ buffer(0) ]],
//    device float* B [[ buffer(1) ]],
//    device float* C [[ buffer(2) ]],
//    constant uint* dimensions [[ buffer(3) ]],
//    uint2 gid [[ thread_position_in_grid ]],
//    uint2 tid [[ thread_position_in_threadgroup ]],
//    uint2 group_size [[ threads_per_threadgroup ]])
//{
//    uint M = dimensions[0];
//    uint N = dimensions[1];
//    uint K = dimensions[2];
//
//    // Shared memory to store tiles of A and B
//    threadgroup float Asub[TILE_SIZE][TILE_SIZE];
//    threadgroup float Bsub[TILE_SIZE][TILE_SIZE];
//
//    uint row = gid.y;
//    uint col = gid.x;
//
//    float sum = 0.0;
//
//    // Loop over all tiles
//    for (uint m = 0; m < (K + TILE_SIZE - 1) / TILE_SIZE; ++m)
//    {
//        // Load A and B tiles into shared memory
//        uint aIndex = row * K + m * TILE_SIZE + tid.x;
//        uint bIndex = (m * TILE_SIZE + tid.y) * N + col;
//        
//        Asub[tid.y][tid.x] = (row < M && (m * TILE_SIZE + tid.x) < K) ? A[aIndex] : 0.0;
//        Bsub[tid.y][tid.x] = ((m * TILE_SIZE + tid.y) < K && col < N) ? B[bIndex] : 0.0;
//
//        // Wait for all threads to load the tiles
//        threadgroup_barrier(mem_flags::mem_threadgroup);
//
//        // Compute product for the current tile
//        for (uint k = 0; k < TILE_SIZE; ++k)
//        {
//            sum += Asub[tid.y][k] * Bsub[k][tid.x];
//        }
//
//        // Wait for all threads to finish computing
//        threadgroup_barrier(mem_flags::mem_threadgroup);
//    }
//
//    // Write the result to the output matrix
//    if (row < M && col < N)
//    {
//        C[row * N + col] = sum;
//    }
//}
//
//
//// The kernel function to perform matrix multiplication with tiling
//kernel void tiledMatMul_Int(
//    device long* A [[ buffer(0) ]],
//    device long* B [[ buffer(1) ]],
//    device long* C [[ buffer(2) ]],
//    constant uint* dimensions [[ buffer(3) ]],
//    uint2 gid [[ thread_position_in_grid ]],
//    uint2 tid [[ thread_position_in_threadgroup ]],
//    uint2 group_size [[ threads_per_threadgroup ]])
//{
//    uint M = dimensions[0];
//    uint N = dimensions[1];
//    uint K = dimensions[2];
//
//    // Shared memory to store tiles of A and B
//    threadgroup int Asub[TILE_SIZE][TILE_SIZE];
//    threadgroup int Bsub[TILE_SIZE][TILE_SIZE];
//
//    uint row = gid.y;
//    uint col = gid.x;
//
//    long sum = 0;
//
//    // Loop over all tiles
//    for (uint m = 0; m < (K + TILE_SIZE - 1) / TILE_SIZE; ++m)
//    {
//        // Load A and B tiles into shared memory
//        uint aIndex = row * K + m * TILE_SIZE + tid.x;
//        uint bIndex = (m * TILE_SIZE + tid.y) * N + col;
//        
//        Asub[tid.y][tid.x] = (row < M && (m * TILE_SIZE + tid.x) < K) ? A[aIndex] : 0.0;
//        Bsub[tid.y][tid.x] = ((m * TILE_SIZE + tid.y) < K && col < N) ? B[bIndex] : 0.0;
//
//        // Wait for all threads to load the tiles
//        threadgroup_barrier(mem_flags::mem_threadgroup);
//
//        // Compute product for the current tile
//        for (uint k = 0; k < TILE_SIZE; ++k)
//        {
//            sum += Asub[tid.y][k] * Bsub[k][tid.x];
//        }
//
//        // Wait for all threads to finish computing
//        threadgroup_barrier(mem_flags::mem_threadgroup);
//    }
//
//    // Write the result to the output matrix
//    if (row < M && col < N)
//    {
//        C[row * N + col] = sum;
//    }
//}
//
//

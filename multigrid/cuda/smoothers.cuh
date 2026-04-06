#ifndef SMOOTHERS_CUDA_H
#define SMOOTHERS_CUDA_H

#include <utility>

#include "grid_device.cuh"

enum SmootherType { JACOBI, JACOBI_AMORTECIDO, GAUSS_SEIDEL_RB };

// No host: declarar u_new e swap
__global__ void jacobi_kernel(Grid2D grid, double* u_new) {

    double h2 = grid.h * grid.h;

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= 1 && i < grid.nx && j >= 1 && j < grid.ny) {
            u_new[grid.idx(i, j)] = (grid.u[grid.idx(i-1, j)] + grid.u[grid.idx(i+1, j)] +
                                     grid.u[grid.idx(i, j-1)] + grid.u[grid.idx(i, j+1)] +
                                     h2 * grid.f[grid.idx(i,j)]) / 4.0;
    }
}

__global__ void jacobi_amortecido_kernel(Grid2D grid, double* u_new) {

    double h2 = grid.h * grid.h;
    double omega = 4.0/5.0; // valor otimo para suavizacao

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= 1 && i < grid.nx && j >= 1 && j < grid.ny) {
        double u_jacobi = (grid.u[grid.idx(i-1, j)] + grid.u[grid.idx(i+1, j)] +
                           grid.u[grid.idx(i, j-1)] + grid.u[grid.idx(i, j+1)] +
                           h2 * grid.f[grid.idx(i,j)]) / 4.0;
        u_new[grid.idx(i, j)] = grid.u[grid.idx(i, j)] + omega * (u_jacobi - grid.u[grid.idx(i, j)]);
    }
}

// 0: red | 1: black
__global__ void gauss_seidel_rb_kernel(Grid2D grid, int color) {

    double h2 = grid.h * grid.h;

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= 1 && i < grid.nx && j >= 1 && j < grid.ny) {
        // Divergencia de threads (otimizar dps)
        if ((i + j) % 2 == color) {
                grid.u[grid.idx(i, j)] = (grid.u[grid.idx(i-1, j)] + grid.u[grid.idx(i+1, j)] +
                                          grid.u[grid.idx(i, j-1)] + grid.u[grid.idx(i, j+1)] +
                                          h2 * grid.f[grid.idx(i,j)]) / 4.0;
        }
    }
}

__host__ void smooth_grid(Grid2D* g, SmootherType smoother, dim3 numBlocks, dim3 numThreads) {
    if (smoother == GAUSS_SEIDEL_RB) {
        gauss_seidel_rb_kernel<<<numBlocks, numThreads>>>(*g, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
        gauss_seidel_rb_kernel<<<numBlocks, numThreads>>>(*g, 1);
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        if (smoother == JACOBI)
            jacobi_kernel<<<numBlocks, numThreads>>>(*g, g->u_new);
        else
            jacobi_amortecido_kernel<<<numBlocks, numThreads>>>(*g, g->u_new);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::swap(g->u, g->u_new);
    }
}

#endif

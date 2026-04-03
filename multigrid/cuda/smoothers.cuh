#ifndef SMOOTHERS_CUDA_H
#define SMOOTHERS_CUDA_H

#include "grid_device.cuh"

// No host: declarar u_new e swap
__global__ void jacobi_kernel(Grid2D* grid, double* u_new) {

    double hx2 = grid->hx * grid->hx;
    double hy2 = grid->hy * grid->hy;
    double diag = 2.0 * (1.0/hx2 + 1.0/hy2);

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= 1 && i < grid->nx && j >= 1 && j < grid->ny) {
            u_new[grid->idx(i, j)] = ((grid->u[grid->idx(i-1, j)] + grid->u[grid->idx(i+1, j)]) / hx2 +
                                     (grid->u[grid->idx(i, j-1)] + grid->u[grid->idx(i, j+1)]) / hy2 +
                                     grid->f[grid->idx(i,j)]) / diag;
    }
}

__global__ void jacobi_amortecido_kernel(Grid2D* grid, double* u_new) {

    double hx2 = grid->hx * grid->hx;
    double hy2 = grid->hy * grid->hy;
    double omega = 4.0/5.0; // valor otimo para suavizacao
    double diag = 2.0 * (1.0/hx2 + 1.0/hy2);

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= 1 && i < grid->nx && j >= 1 && j < grid->ny) {
        double u_jacobi = ((grid->u[grid->idx(i-1, j)] + grid->u[grid->idx(i+1, j)]) / hx2 +
                        (grid->u[grid->idx(i, j-1)] + grid->u[grid->idx(i, j+1)]) / hy2 +
                        grid->f[grid->idx(i,j)]) / diag;
        u_new[grid->idx(i, j)] = grid->u[grid->idx(i, j)] + omega * (u_jacobi - grid->u[grid->idx(i, j)]);
    }
}

// 0: red | 1: black
__global__ void gauss_seidel_rb_kernel(Grid2D* grid, int color) {

    double hx2 = grid->hx * grid->hx;
    double hy2 = grid->hy * grid->hy;
    double diag = 2.0 * (1.0/hx2 + 1.0/hy2);

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= 1 && i < grid->nx && j >= 1 && j < grid->ny) {
        // Divergencia de threads (otimizar dps)
        if ((i + j) % 2 == color) {
                grid->u[grid->idx(i, j)] = ((grid->u[grid->idx(i-1, j)] + grid->u[grid->idx(i+1, j)]) / hx2 +
                                          (grid->u[grid->idx(i, j-1)] + grid->u[grid->idx(i, j+1)]) / hy2 +
                                          grid->f[grid->idx(i,j)]) / diag;
        }
    }
}

#endif

#ifndef SMOOTHERS_CUDA_H
#define SMOOTHERS_CUDA_H

#include "grid_device.cuh"

// No host: declarar u_new e swap
__global__ void jacobi_kernel(Grid2D* grid, double* u_new) {

    double hx2 = grid->hx * grid->hx;
    double hy2 = grid->hy * grid->hy;
    double diag = 2.0 * (1.0/hx2 + 1.0/hy2);

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= 1 && i < grid->nx && j >= 1 && j < grid->ny) {
            u_new[grid->idx(i, j)] = ((grid->u[grid->idx(i-1, j)] + grid->u[grid->idx(i+1, j)]) / hx2 +
                                     (grid->u[grid->idx(i, j-1)] + grid->u[grid->idx(i, j+1)]) / hy2 +
                                     grid->f[grid->idx(i,j)]) / diag;
    }
}

#endif

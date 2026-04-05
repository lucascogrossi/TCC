#ifndef MULTIGRID_UTILS_CUDA_H
#define MULTIGRID_UTILS_CUDA_H

#include <cmath>

#include "grid_device.cuh"
#include "smoothers.cuh"

// pre alocar r no host
__global__ void compute_residual_kernel(Grid2D grid, double* r) {
    double hx2 = grid.hx * grid.hx;
    double hy2 = grid.hy * grid.hy;

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= 1 && i < grid.nx && j >= 1 && j < grid.ny) {
            double Au_ij = (-grid.u[grid.idx(i-1,j)] + 2*grid.u[grid.idx(i,j)] - grid.u[grid.idx(i+1,j)]) / hx2
                         + (-grid.u[grid.idx(i,j-1)] + 2*grid.u[grid.idx(i,j)] - grid.u[grid.idx(i,j+1)]) / hy2;
            r[grid.idx(i, j)] = grid.f[grid.idx(i, j)] - Au_ij;
    }
}

// Cada bloco reduz sua porcao de r[] para uma soma parcial de r^2,
// depois acumula atomicamente em result.
// Requer compute capability >= 6.0 para atomicAdd em double.
// Chamar com shared memory = blockDim.x * blockDim.y * sizeof(double).
__global__ void residual_norm_kernel(const double* r, double* result, int nx, int ny) {

    extern __shared__ double sdata[];

    int j   = blockIdx.x * blockDim.x + threadIdx.x;
    int i   = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    double val = 0.0;
    if (i >= 1 && i < nx && j >= 1 && j < ny) {
        double rij = r[i * (ny+1) + j];
        val = rij * rij;
    }
    sdata[tid] = val;
    __syncthreads();

    // reducao convergente em shared memory (baseada em SimpleSumReduction)
    for (unsigned int stride = (blockDim.x * blockDim.y) / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    if (tid == 0)
        atomicAdd(result, sdata[0]);
}

// Executa compute_residual + reducao na GPU e retorna a norma L2.
// d_result deve ser um ponteiro para double em unified memory, pre-zerado nao e necessario
// (a funcao zera antes de chamar os kernels).
__host__ double residual_norm_gpu(Grid2D* grid, double* d_result) {

    dim3 numThreadsPerBlock(16, 16);
    dim3 numBlocks((grid->ny + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x,
                   (grid->nx + numThreadsPerBlock.y - 1) / numThreadsPerBlock.y);
    int sharedMem = numThreadsPerBlock.x * numThreadsPerBlock.y * sizeof(double);

    CUDA_CHECK(cudaMemset(d_result, 0, sizeof(double)));
    compute_residual_kernel<<<numBlocks, numThreadsPerBlock>>>(*grid, grid->r);
    CUDA_CHECK(cudaDeviceSynchronize());
    residual_norm_kernel<<<numBlocks, numThreadsPerBlock, sharedMem>>>(grid->r, d_result, grid->nx, grid->ny);
    CUDA_CHECK(cudaDeviceSynchronize());

    double h_result;
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost));
    return sqrt(h_result * grid->hx * grid->hy);
}

// cada thread cuida de um ponto do grid grosso
// pre alocar r_coarse no host
__global__ void restriction_kernel(const double *r, double* r_coarse, int nx, int ny) {

    // numero de intervalos do grid gross em cada direcao
    int nx_c = nx / 2;
    int ny_c = ny / 2;

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= 1 && i < nx_c && j >= 1 && j < ny_c) {
        int i_fine = 2 * i;
        int j_fine = 2 * j;

        r_coarse[i*(ny_c+1) + j] =
            (1.0/16.0) * (
                // cantos (peso 1)
                r[(i_fine-1)*(ny+1) + (j_fine-1)] + r[(i_fine-1)*(ny+1) + (j_fine+1)] +
                r[(i_fine+1)*(ny+1) + (j_fine-1)] + r[(i_fine+1)*(ny+1) + (j_fine+1)] +
                // arestas (peso 2)
                2.0 * (r[(i_fine-1)*(ny+1) + j_fine] + r[(i_fine+1)*(ny+1) + j_fine] +
                       r[i_fine*(ny+1) + (j_fine-1)] + r[i_fine*(ny+1) + (j_fine+1)]) +
                // centro (peso 4)
                4.0 * r[i_fine*(ny+1) + j_fine]
            );

    }
}

// cada thread cuida de um ponto do grid fino
__global__ void prolongation_kernel(const double* e_coarse, double* e_fine, int nx_c, int ny_c) {

    // grid fino tem n_coarse*2 intervalos
    int ny_f = ny_c * 2;

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i <= nx_c && j <= ny_c) {
        // caso 1: copia direto
        e_fine[2*i*(ny_f+1) + 2*j] = e_coarse[i*(ny_c+1) + j];

        // caso 2: media horizontal
        if (j < ny_c)
            e_fine[2*i*(ny_f+1) + 2*j+1] = (e_coarse[i*(ny_c+1) + j] + e_coarse[i*(ny_c+1) + j+1]) / 2.0;

        // caso 3: media vertical
        if (i < nx_c)
             e_fine[(2*i+1)*(ny_f+1) + 2*j] = (e_coarse[i*(ny_c+1) + j] + e_coarse[(i+1)*(ny_c+1) + j]) / 2.0;

        // caso 4: media dos 4 vizinhos
        if (i < nx_c && j < ny_c)
            e_fine[(2*i+1)*(ny_f+1) + 2*j+1] = (e_coarse[i*(ny_c+1) + j] + e_coarse[i*(ny_c+1) + j+1] +
                                                    e_coarse[(i+1)*(ny_c+1) + j] + e_coarse[(i+1)*(ny_c+1) + j+1]) / 4.0;
    }
}

__global__ void correct_kernel(Grid2D grid, const double* e_fine) {

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= 1 && i < grid.nx && j >= 1 && j < grid.ny) {
        grid.u[grid.idx(i, j)] += e_fine[grid.idx(i, j)];
    }
}

// No momento o grid mais grosso tem apenas 1 ponto interior.
// 255 threads ficam ociosas
__host__ void solve_coarse(Grid2D* grid, int sweeps = 1) {
    CUDA_CHECK(cudaMemset(grid->u, 0, (grid->nx+1)*(grid->ny+1)*sizeof(double)));

    dim3 numThreadsPerBlock(16, 16);
    dim3 numBlocks((grid->ny + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x,
              (grid->nx + numThreadsPerBlock.y - 1) / numThreadsPerBlock.y);

    for (int k = 0; k < sweeps; k++) {
        gauss_seidel_rb_kernel<<<numBlocks, numThreadsPerBlock>>>(*grid, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
        gauss_seidel_rb_kernel<<<numBlocks, numThreadsPerBlock>>>(*grid, 1);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}


#endif

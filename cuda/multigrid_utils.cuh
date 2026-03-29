#ifndef MULTIGRID_UTILS_H
#define MULTIGRID_UTILS_H

#include <cmath>

#include "grid_device.cuh"
#include "smoothers.cuh"

// pre alocar r no host
__global__ void compute_residual_kernel(const Grid2D* grid, double* r) {
    double hx2 = grid->hx * grid->hx;
    double hy2 = grid->hy * grid->hy;

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= 1 && i < grid->nx && j >= 1 && j < grid->ny) {
            double Au_ij = (-grid->u[grid->idx(i-1,j)] + 2*grid->u[grid->idx(i,j)] - grid->u[grid->idx(i+1,j)]) / hx2
                         + (-grid->u[grid->idx(i,j-1)] + 2*grid->u[grid->idx(i,j)] - grid->u[grid->idx(i,j+1)]) / hy2;
            r[grid->idx(i, j)] = grid->f[grid->idx(i, j)] - Au_ij;
    }
}

/*
inline double residual_norm(const Grid2D& grid) {
    std::vector<double> r = compute_residual(grid);
    double norm = 0.0;
    for (int i = 1; i < grid.nx; i++) {
        for (int j = 1; j < grid.ny; j++) {
            norm += r[grid.idx(i, j)] * r[grid.idx(i, j)];
        }
    }
    return sqrt(norm * grid.hx * grid.hy);
}
*/

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

/*
inline std::vector<double> prolongation(const std::vector<double>& e_coarse, int nx_c, int ny_c) {
    // grid fino tem n_coarse*2 intervalos
    int nx_f = nx_c * 2;
    int ny_f = ny_c * 2;
    std::vector<double> e_fine((nx_f+1) * (ny_f+1), 0.0);

    // Interpolacao linear
    for (int i = 0; i <= nx_c; i++) {
        for (int j = 0; j <= ny_c; j++) {
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
    return e_fine;
}

inline void solve_coarse(Grid2D& coarse) {
    std::fill(coarse.u.begin(), coarse.u.end(), 0.0);
    gauss_seidel_rb(coarse);
}
*/

__global__ void correct_kernel(Grid2D* grid, const double* e_fine) {

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= 1 && i < grid->nx && j >= 1 && j < grid->ny) {
        grid->u[grid->idx(i, j)] += e_fine[grid->idx(i, j)];
    }
}

#endif

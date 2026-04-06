#ifndef MULTIGRID_UTILS_H
#define MULTIGRID_UTILS_H
#include <cmath>

#include "grid.h"
#include "smoothers.h"


inline std::vector<double> compute_residual(const Grid2D& grid) {
    std::vector<double> r((grid.nx+1) * (grid.ny+1), 0.0);
    double h2 = grid.h * grid.h;

    for (int i = 1; i < grid.nx; i++) {
        for (int j = 1; j < grid.ny; j++) {
            double Au_ij = (4*grid.u[grid.idx(i,j)] - grid.u[grid.idx(i-1,j)] - grid.u[grid.idx(i+1,j)]
                           - grid.u[grid.idx(i,j-1)] - grid.u[grid.idx(i,j+1)]) / h2;
            r[grid.idx(i, j)] = grid.f[grid.idx(i, j)] - Au_ij;
        }
    }
    return r;
}

inline double residual_norm(const Grid2D& grid) {
    std::vector<double> r = compute_residual(grid);
    double norm = 0.0;
    for (int i = 1; i < grid.nx; i++) {
        for (int j = 1; j < grid.ny; j++) {
            norm += r[grid.idx(i, j)] * r[grid.idx(i, j)];
        }
    }
    return sqrt(norm * grid.h * grid.h);
}

inline std::vector<double> restriction(const std::vector<double>& r, int nx, int ny) {
    // numero de intervalos do grid gross em cada direcao
    int nx_c = nx / 2;
    int ny_c = ny / 2;
    
    std::vector<double> r_coarse((nx_c+1) * (ny_c + 1), 0.0);
    
    // Full weighting
    // Pesos:
    // 1  2  1
    // 2  4  2   × (1/16)
    // 1  2  1
    for (int i = 1; i < nx_c; i++) {
        for (int j = 1; j < ny_c; j++) {
        int i_fine = 2*i;     // indice correspondente no grid fino
        int j_fine = 2*j;

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
    return r_coarse;
}

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

inline void correct(Grid2D& grid, const std::vector<double>& e_fine) {
    for (int i = 1; i < grid.nx; i++)
        for (int j = 1; j < grid.ny; j++)
            grid.u[grid.idx(i, j)] += e_fine[grid.idx(i, j)];
}

#endif

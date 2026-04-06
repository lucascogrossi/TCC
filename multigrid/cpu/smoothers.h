#ifndef SMOOTHERS_H
#define SMOOTHERS_H

#include <vector>
#include "grid.h"

inline void jacobi(Grid2D& grid) {
    double h2 = grid.h * grid.h;

    for (int i = 1; i < grid.nx; i++) {
        for (int j = 1; j < grid.ny; j++) {
            grid.u_new[grid.idx(i, j)] = (grid.u[grid.idx(i-1, j)] + grid.u[grid.idx(i+1, j)] +
                                           grid.u[grid.idx(i, j-1)] + grid.u[grid.idx(i, j+1)] +
                                           h2 * grid.f[grid.idx(i,j)]) / 4.0;
        }
    }
    std::swap(grid.u, grid.u_new);
}

inline void jacobi_amortecido(Grid2D& grid) {
    double omega = 4.0/5.0; // valor otimo para suavizacao
    double h2 = grid.h * grid.h;

    for (int i = 1; i < grid.nx; i++) {
        for (int j = 1; j < grid.ny; j++) {
            double u_jacobi = (grid.u[grid.idx(i-1, j)] + grid.u[grid.idx(i+1, j)] +
                               grid.u[grid.idx(i, j-1)] + grid.u[grid.idx(i, j+1)] +
                               h2 * grid.f[grid.idx(i,j)]) / 4.0;

            grid.u_new[grid.idx(i, j)] = grid.u[grid.idx(i, j)] + omega * (u_jacobi - grid.u[grid.idx(i, j)]);
        }
    }
    std::swap(grid.u, grid.u_new);
}

inline void gauss_seidel(Grid2D& grid) {
    double h2 = grid.h * grid.h;

    for (int i = 1; i < grid.nx; i++) {
        for (int j = 1; j < grid.ny; j++) {
            grid.u[grid.idx(i, j)] = (grid.u[grid.idx(i-1, j)] + grid.u[grid.idx(i+1, j)] +
                                      grid.u[grid.idx(i, j-1)] + grid.u[grid.idx(i, j+1)] +
                                      h2 * grid.f[grid.idx(i,j)]) / 4.0;
        }
    }
}

// Gauss-Seidel Sobrerelaxado
inline void sor(Grid2D& grid) {
    double omega = 1.15;
    double h2 = grid.h * grid.h;

    for (int i = 1; i < grid.nx; i++) {
        for (int j = 1; j < grid.ny; j++) {
            double u_gs = (grid.u[grid.idx(i-1, j)] + grid.u[grid.idx(i+1, j)] +
                           grid.u[grid.idx(i, j-1)] + grid.u[grid.idx(i, j+1)] +
                           h2 * grid.f[grid.idx(i,j)]) / 4.0;
            grid.u[grid.idx(i, j)] = grid.u[grid.idx(i, j)] + omega * (u_gs - grid.u[grid.idx(i, j)]);
        }
    }
}

inline void gauss_seidel_rb(Grid2D& grid) {
    double h2 = grid.h * grid.h;

    // vermelho: indices pares
    for (int i = 1; i < grid.nx; i++) {
        for (int j = 1; j < grid.ny; j++) {
            if ((i + j) % 2 == 0) {
                grid.u[grid.idx(i, j)] = (grid.u[grid.idx(i-1, j)] + grid.u[grid.idx(i+1, j)] +
                                          grid.u[grid.idx(i, j-1)] + grid.u[grid.idx(i, j+1)] +
                                          h2 * grid.f[grid.idx(i,j)]) / 4.0;
            }
        }
    }
    // preto: indices impares
    for (int i = 1; i < grid.nx; i++) {
        for (int j = 1; j < grid.ny; j++) {
            if ((i + j) % 2 != 0) {
                grid.u[grid.idx(i, j)] = (grid.u[grid.idx(i-1, j)] + grid.u[grid.idx(i+1, j)] +
                                          grid.u[grid.idx(i, j-1)] + grid.u[grid.idx(i, j+1)] +
                                          h2 * grid.f[grid.idx(i,j)]) / 4.0;
            }
        }
    }
}

#endif

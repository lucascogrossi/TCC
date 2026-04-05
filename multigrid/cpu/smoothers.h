#ifndef SMOOTHERS_H
#define SMOOTHERS_H

#include <vector>
#include "grid.h"

inline void jacobi(Grid2D& grid) {
    double hx2 = grid.hx * grid.hx;
    double hy2 = grid.hy * grid.hy;
    double diag = 2.0 * (1.0/hx2 + 1.0/hy2);

    for (int i = 1; i < grid.nx; i++) {
        for (int j = 1; j < grid.ny; j++) {
            grid.u_new[grid.idx(i, j)] = ((grid.u[grid.idx(i-1, j)] + grid.u[grid.idx(i+1, j)]) / hx2 +
                                           (grid.u[grid.idx(i, j-1)] + grid.u[grid.idx(i, j+1)]) / hy2 +
                                           grid.f[grid.idx(i,j)]) / diag;
        }
    }
    std::swap(grid.u, grid.u_new);
}

inline void jacobi_amortecido(Grid2D& grid) {
    double omega = 4.0/5.0; // valor otimo para suavizacao
    double hx2 = grid.hx * grid.hx;
    double hy2 = grid.hy * grid.hy;
    double diag = 2.0 * (1.0/hx2 + 1.0/hy2);

    for (int i = 1; i < grid.nx; i++) {
        for (int j = 1; j < grid.ny; j++) {
            double u_jacobi = ((grid.u[grid.idx(i-1, j)] + grid.u[grid.idx(i+1, j)]) / hx2 +
                               (grid.u[grid.idx(i, j-1)] + grid.u[grid.idx(i, j+1)]) / hy2 +
                               grid.f[grid.idx(i,j)]) / diag;

            grid.u_new[grid.idx(i, j)] = grid.u[grid.idx(i, j)] + omega * (u_jacobi - grid.u[grid.idx(i, j)]);
        }
    }
    std::swap(grid.u, grid.u_new);
}

inline void gauss_seidel(Grid2D& grid) {
    double hx2 = grid.hx * grid.hx;
    double hy2 = grid.hy * grid.hy;
    double diag = 2.0 * (1.0/hx2 + 1.0/hy2);

    for (int i = 1; i < grid.nx; i++) {
        for (int j = 1; j < grid.ny; j++) {
            grid.u[grid.idx(i, j)] = ((grid.u[grid.idx(i-1, j)] + grid.u[grid.idx(i+1, j)]) / hx2 +
                                      (grid.u[grid.idx(i, j-1)] + grid.u[grid.idx(i, j+1)]) / hy2 +
                                      grid.f[grid.idx(i,j)]) / diag;
        }
    }
}

// Gauss-Seidel Sobrerelaxado
inline void sor(Grid2D& grid) {
    double omega = 1.15;
    double hx2 = grid.hx * grid.hx;
    double hy2 = grid.hy * grid.hy;
    double diag = 2.0 * (1.0/hx2 + 1.0/hy2);

    for (int i = 1; i < grid.nx; i++) {
        for (int j = 1; j < grid.ny; j++) {
            double u_gs = ((grid.u[grid.idx(i-1, j)] + grid.u[grid.idx(i+1, j)]) / hx2 +
                           (grid.u[grid.idx(i, j-1)] + grid.u[grid.idx(i, j+1)]) / hy2 +
                           grid.f[grid.idx(i,j)]) / diag;
            grid.u[grid.idx(i, j)] = grid.u[grid.idx(i, j)] + omega * (u_gs - grid.u[grid.idx(i, j)]);
        }
    }
}

inline void gauss_seidel_rb(Grid2D& grid) {
    double hx2 = grid.hx * grid.hx;
    double hy2 = grid.hy * grid.hy;
    double diag = 2.0 * (1.0/hx2 + 1.0/hy2);

    // vermelho: indices pares
    for (int i = 1; i < grid.nx; i++) {
        for (int j = 1; j < grid.ny; j++) {
            if ((i + j) % 2 == 0) {
                grid.u[grid.idx(i, j)] = ((grid.u[grid.idx(i-1, j)] + grid.u[grid.idx(i+1, j)]) / hx2 +
                                          (grid.u[grid.idx(i, j-1)] + grid.u[grid.idx(i, j+1)]) / hy2 +
                                          grid.f[grid.idx(i,j)]) / diag;
            }
        }
    }
    // preto: indices impares
    for (int i = 1; i < grid.nx; i++) {
        for (int j = 1; j < grid.ny; j++) {
            if ((i + j) % 2 != 0) {
                grid.u[grid.idx(i, j)] = ((grid.u[grid.idx(i-1, j)] + grid.u[grid.idx(i+1, j)]) / hx2 +
                                          (grid.u[grid.idx(i, j-1)] + grid.u[grid.idx(i, j+1)]) / hy2 +
                                          grid.f[grid.idx(i,j)]) / diag;
            }
        }
    }
}

inline void gauss_seidel_linha(Grid2D& grid) {
    // TODO
    return;
}

inline void gauss_seidel_coluna(Grid2D& grid) {
    // TODO
    return;
}

#endif

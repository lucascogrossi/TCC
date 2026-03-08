#ifndef SMOOTHERS_2D_H
#define SMOOTHERS_2D_H

#include <vector>
#include "grid_2d.h"

inline void jacobi(Grid2D& grid) {
    std::vector<double> u_new((grid.nx+1) * (grid.ny+1), 0.0);
    double hx2 = grid.hx * grid.hx;
    double hy2 = grid.hy * grid.hy;
    double diag = 2.0 * (1.0/hx2 + 1.0/hy2);

    for (int i = 1; i < grid.nx; i++) {
        for (int j = 1; j < grid.ny; j++) {
            u_new[grid.idx(i, j)] = ((grid.u[grid.idx(i-1, j)] + grid.u[grid.idx(i+1, j)]) / hx2 +
                                     (grid.u[grid.idx(i, j-1)] + grid.u[grid.idx(i, j+1)]) / hy2 +
                                     grid.f[grid.idx(i,j)]) / diag;
        }
    }
    grid.u = u_new;
}

inline void jacobi_amortecido(Grid2D& grid) {
    std::vector<double> u_new((grid.nx + 1) * (grid.ny+1), 0.0);
    double u_jacobi;
    double omega = 4.0/5.0; // valor otimo para suavizacao
    double hx2 = grid.hx * grid.hx;
    double hy2 = grid.hy * grid.hy;
    double diag = 2.0 * (1.0/hx2 + 1.0/hy2);

    for (int i = 1; i < grid.nx; i++) {
        for (int j = 1; j < grid.ny; j++) {
            u_jacobi = ((grid.u[grid.idx(i-1, j)] + grid.u[grid.idx(i+1, j)]) / hx2 +
                        (grid.u[grid.idx(i, j-1)] + grid.u[grid.idx(i, j+1)]) / hy2 +
                        grid.f[grid.idx(i,j)]) / diag;

            u_new[grid.idx(i, j)] = grid.u[grid.idx(i, j)] + omega * (u_jacobi - grid.u[grid.idx(i, j)]);
        }
    }
    grid.u = u_new;
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
inline void sor(Grid2D& grid, double omega) {
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

#endif
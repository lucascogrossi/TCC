#ifndef SMOOTHERS_H
#define SMOOTHERS_H

#include <vector>
#include <cmath>
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
    // omega otimo para o problema modelo de Poisson 2D com Dirichlet:
    // 2 / (1 + sin(pi*h))
    double omega = 2.0 / (1.0 + std::sin(M_PI * grid.h));
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

// Algoritmo de Thomas (TDMA): resolve A x = d em O(n), onde A eh tridiagonal
// com sub-diagonal a, diagonal b e super-diagonal c.
//   a[0]    nao usado (nao ha sub-diagonal na primeira linha)
//   c[n-1]  nao usado (nao ha super-diagonal na ultima linha)
//   d       contem o RHS na entrada e a solucao na saida (in-place)
inline void thomas(const std::vector<double>& a, const std::vector<double>& b,
                   const std::vector<double>& c, std::vector<double>& d, int n) {
    std::vector<double> c_prime(n);

    // Eliminacao (forward sweep)
    c_prime[0] = c[0] / b[0];
    d[0]       = d[0] / b[0];
    for (int i = 1; i < n; i++) {
        double m = 1.0 / (b[i] - a[i] * c_prime[i-1]);
        c_prime[i] = c[i] * m;
        d[i]       = (d[i] - a[i] * d[i-1]) * m;
    }

    // Substituicao reversa (backward substitution)
    for (int i = n - 2; i >= 0; i--) {
        d[i] = d[i] - c_prime[i] * d[i+1];
    }
}

// Gauss-Seidel linha em x (x-linha-GS):
// Atualiza uma linha horizontal (j fixo) inteira por vez resolvendo um tridiagonal
// em i pelo algoritmo de Thomas. Ordem das linhas: lexicografica (j = 1, 2, ..., ny-1).
//
// Para cada ponto (i,j) interno:
//   -u[i-1,j] + 4 u[i,j] - u[i+1,j] = h^2 f[i,j] + u[i,j-1] + u[i,j+1]
//                                     '-------- conhecido --------'
//   incognitas da linha j               u[i,j-1] ja atualizado (linha anterior)
//                                       u[i,j+1] valor antigo  (proxima linha)
inline void gauss_seidel_linha_x(Grid2D& grid) {
    double h2 = grid.h * grid.h;
    int n = grid.nx - 1;  // numero de pontos internos por linha

    // Coeficientes constantes do tridiagonal [-1, 4, -1] (Poisson com stencil 5 pontos)
    std::vector<double> a(n, -1.0), b(n, 4.0), c(n, -1.0);
    std::vector<double> d(n);

    for (int j = 1; j < grid.ny; j++) {
        // Monta RHS. Dirichlet u=0 na fronteira: u[0,j] e u[nx,j] nao aparecem
        // explicitamente (entrariam somando 0 em d[0] e d[n-1]).
        for (int i = 1; i < grid.nx; i++) {
            d[i-1] = h2 * grid.f[grid.idx(i, j)]
                   + grid.u[grid.idx(i, j-1)]
                   + grid.u[grid.idx(i, j+1)];
        }
        thomas(a, b, c, d, n);
        for (int i = 1; i < grid.nx; i++) {
            grid.u[grid.idx(i, j)] = d[i-1];
        }
    }
}

// Gauss-Seidel linha em y (y-linha-GS):
// Atualiza uma coluna (i fixo) inteira por vez resolvendo um tridiagonal em j.
// Ordem das colunas: lexicografica (i = 1, 2, ..., nx-1).
//
//   -u[i,j-1] + 4 u[i,j] - u[i,j+1] = h^2 f[i,j] + u[i-1,j] + u[i+1,j]
inline void gauss_seidel_linha_y(Grid2D& grid) {
    double h2 = grid.h * grid.h;
    int n = grid.ny - 1;

    std::vector<double> a(n, -1.0), b(n, 4.0), c(n, -1.0);
    std::vector<double> d(n);

    for (int i = 1; i < grid.nx; i++) {
        for (int j = 1; j < grid.ny; j++) {
            d[j-1] = h2 * grid.f[grid.idx(i, j)]
                   + grid.u[grid.idx(i-1, j)]
                   + grid.u[grid.idx(i+1, j)];
        }
        thomas(a, b, c, d, n);
        for (int j = 1; j < grid.ny; j++) {
            grid.u[grid.idx(i, j)] = d[j-1];
        }
    }
}

// Gauss-Seidel zebra em x (x-zebra-GS):
// Mesma logica do x-linha, mas com ordenacao red/black das linhas.
// Todas as linhas vermelhas (j impar) sao independentes entre si pois so
// dependem das linhas pretas (j par), e vice-versa.
inline void gauss_seidel_zebra_x(Grid2D& grid) {
    double h2 = grid.h * grid.h;
    int n = grid.nx - 1;

    std::vector<double> a(n, -1.0), b(n, 4.0), c(n, -1.0);
    std::vector<double> d(n);

    // Linhas vermelhas (j impar)
    for (int j = 1; j < grid.ny; j += 2) {
        for (int i = 1; i < grid.nx; i++) {
            d[i-1] = h2 * grid.f[grid.idx(i, j)]
                   + grid.u[grid.idx(i, j-1)]
                   + grid.u[grid.idx(i, j+1)];
        }
        thomas(a, b, c, d, n);
        for (int i = 1; i < grid.nx; i++) {
            grid.u[grid.idx(i, j)] = d[i-1];
        }
    }

    // Linhas pretas (j par)
    for (int j = 2; j < grid.ny; j += 2) {
        for (int i = 1; i < grid.nx; i++) {
            d[i-1] = h2 * grid.f[grid.idx(i, j)]
                   + grid.u[grid.idx(i, j-1)]
                   + grid.u[grid.idx(i, j+1)];
        }
        thomas(a, b, c, d, n);
        for (int i = 1; i < grid.nx; i++) {
            grid.u[grid.idx(i, j)] = d[i-1];
        }
    }
}

// Gauss-Seidel zebra em y (y-zebra-GS):
// Mesma logica do y-linha, mas com ordenacao red/black das colunas.
inline void gauss_seidel_zebra_y(Grid2D& grid) {
    double h2 = grid.h * grid.h;
    int n = grid.ny - 1;

    std::vector<double> a(n, -1.0), b(n, 4.0), c(n, -1.0);
    std::vector<double> d(n);

    // Colunas vermelhas (i impar)
    for (int i = 1; i < grid.nx; i += 2) {
        for (int j = 1; j < grid.ny; j++) {
            d[j-1] = h2 * grid.f[grid.idx(i, j)]
                   + grid.u[grid.idx(i-1, j)]
                   + grid.u[grid.idx(i+1, j)];
        }
        thomas(a, b, c, d, n);
        for (int j = 1; j < grid.ny; j++) {
            grid.u[grid.idx(i, j)] = d[j-1];
        }
    }

    // Colunas pretas (i par)
    for (int i = 2; i < grid.nx; i += 2) {
        for (int j = 1; j < grid.ny; j++) {
            d[j-1] = h2 * grid.f[grid.idx(i, j)]
                   + grid.u[grid.idx(i-1, j)]
                   + grid.u[grid.idx(i+1, j)];
        }
        thomas(a, b, c, d, n);
        for (int j = 1; j < grid.ny; j++) {
            grid.u[grid.idx(i, j)] = d[j-1];
        }
    }
}

#endif

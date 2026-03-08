#include <iostream>
#include <vector>
#include <cmath>

#include "smoothers_2d.h"
#include "grid_2d.h"
#include "multigrid_utils_2d.h"

void v_cycle(Grid2D& grid) {
    // Condicao de parada: grid com 1 ponto interior (1,1) cercado pela fronteira
    if (grid.nx == 2 && grid.ny == 2) {
        solve_coarse(grid);
        return;
    }

    // 1. pre-suavizacao
    for (int k = 0; k < 5; k++)
        jacobi_amortecido(grid);

    // 2. calcula residuo no grid fino
    std::vector<double> r = compute_residual(grid);

    // 3. restricao: leva residuo para grid grosso
    int nx_c = grid.nx / 2;
    int ny_c = grid.ny / 2;
    std::vector<double> r_coarse = restriction(r, grid.nx, grid.ny);

    // 4. resolve no grid grosso
    Grid2D coarse_grid(nx_c, ny_c, grid.Lx, grid.Ly);
    coarse_grid.f = r_coarse;
    v_cycle(coarse_grid);

    // 5. prolongamento: leva correcao de volta para grid fino
    std::vector<double> e_fine = prolongation(coarse_grid.u, nx_c, ny_c);

    // 6. corrige solucao
    for (int i = 1; i < grid.nx; i++)
        for (int j = 1; j < grid.ny; j++) {
            grid.u[grid.idx(i, j)] += e_fine[grid.idx(i, j)];
        }

    // 7. pos-suavizacao
    for (int k = 0; k < 5; k++)
        jacobi_amortecido(grid);
}

int main(void) {
    // cria grid com 128 intervalos em cada direcao em [0, 1] x [0, 1]
    Grid2D grid(128, 128, 1.0, 1.0);

    
    // preenche f
    for (int i = 1; i < grid.nx; i++) {
        for (int j = 1; j < grid.ny; j++) {
            double x = i * grid.hx;
            double y = j * grid.hy;
            grid.f[grid.idx(i, j)] = 2 * M_PI * M_PI * sin(M_PI * x) * sin(M_PI * y);
            // solucao analitica: u(x,y) = sin(pi*x) * sin(pi*y)
        }
    }

    for (int k = 0; k < 10; k++) {
        v_cycle(grid);
        std::cout << "residuo " << "k = " << k << " " << residual_norm(grid) << std::endl;
    }
    std::cout << "residuo final: " << residual_norm(grid) << std::endl;

    /*
    std::cout << "\nSolucao aproximada:" << std::endl;
    for (int i = 0; i <= grid.nx; i++) {
        for (int j = 0; j <= grid.ny; j++) {
            double x = i * grid.hx;
            double y = j * grid.hy;
            std::cout << "u(" << x << ", " << y << ") = " << grid.u[grid.idx(i,j)] << std::endl;
        }
    }
        */
    return 0;


}
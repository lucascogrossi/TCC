#ifndef VCYCLE_2D_H
#define VCYCLE_2D_H

#include <functional>

#include "grid_2d.h"
#include "multigrid_utils_2d.h"

using Smoother = std::function<void(Grid2D&)>;

void v_cycle(Grid2D& grid, Smoother smooth) {
    if (grid.nx == 2 && grid.ny == 2) {
        solve_coarse(grid);
        return;
    }

    // 1. pre-suavizacao
    for (int k = 0; k < 2; k++)
        smooth(grid);

    // 2. calcula residuo no grid fino
    std::vector<double> r = compute_residual(grid);

    // 3. restricao: leva residuo para grid grosso
    int nx_c = grid.nx / 2;
    int ny_c = grid.ny / 2;
    std::vector<double> r_coarse = restriction(r, grid.nx, grid.ny);

    // 4. resolve no grid grosso
    Grid2D coarse_grid(nx_c, ny_c, grid.Lx, grid.Ly);
    coarse_grid.f = r_coarse;
    v_cycle(coarse_grid, smooth);

    // 5. prolongamento: leva correcao de volta para grid fino
    std::vector<double> e_fine = prolongation(coarse_grid.u, nx_c, ny_c);

    // 6. corrige solucao
    for (int i = 1; i < grid.nx; i++)
        for (int j = 1; j < grid.ny; j++)
            grid.u[grid.idx(i, j)] += e_fine[grid.idx(i, j)];

    // 7. pos-suavizacao
    for (int k = 0; k < 2; k++)
        smooth(grid);
}

#endif
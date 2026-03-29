#ifndef VCYCLE_H
#define VCYCLE_H

#include <functional>

#include "grid.h"
#include "multigrid_utils.h"

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
    correct(grid, e_fine);

    // 7. pos-suavizacao
    for (int k = 0; k < 2; k++)
        smooth(grid);
}

#endif
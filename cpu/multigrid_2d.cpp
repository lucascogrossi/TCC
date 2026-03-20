#include <iostream>
#include <vector>
#include <cmath>
#include <functional>
#include <chrono>

#include "smoothers_2d.h"
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

void print_usage() {
    std::cout << "Uso: ./multigrid_2d --n <grid_size> --smoother <smoother> [--tol <tolerancia>]\n"
              << "\n"
              << "Argumentos:\n"
              << "  --n         Tamanho do grid (potencia de 2: 64, 128, 256, ...)\n"
              << "  --smoother  Metodo de suavizacao:\n"
              << "                jacobi, jacobi_amortecido, gauss_seidel,\n"
              << "                gauss_seidel_rb, sor\n"
              << "  --tol       Tolerancia para convergencia (default: 1e-10)\n"
              << "\n"
              << "Exemplo:\n"
              << "  ./multigrid_2d --n 256 --smoother gauss_seidel_rb --tol 1e-8\n";
}

int main(int argc, char* argv[]) {

    int n = 0;
    std::string smoother_name;
    double tol = 1e-10;
    int max_vcycles = 200;

    // parse de argumentos
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--n" && i + 1 < argc)
            n = std::atoi(argv[++i]);
        else if (arg == "--smoother" && i + 1 < argc)
            smoother_name = argv[++i];
        else if (arg == "--tol" && i + 1 < argc)
            tol = std::atof(argv[++i]);
        else if (arg == "--help" || arg == "-h") {
            print_usage();
            return 0;
        }
        else {
            std::cerr << "Argumento desconhecido: " << arg << "\n\n";
            print_usage();
            return 1;
        }
    }

    // valida argumentos obrigatorios
    if (n == 0 || smoother_name.empty()) {
        std::cerr << "Erro: --n e --smoother sao obrigatorios.\n\n";
        print_usage();
        return 1;
    }

    // seleciona smoother
    Smoother smooth;
    if (smoother_name == "jacobi")
        smooth = jacobi;
    else if (smoother_name == "jacobi_amortecido")
        smooth = jacobi_amortecido;
    else if (smoother_name == "gauss_seidel")
        smooth = gauss_seidel;
    else if (smoother_name == "gauss_seidel_rb")
        smooth = gauss_seidel_rb;
    else if (smoother_name == "sor")
        smooth = sor;
    else {
        std::cerr << "Smoother invalido: " << smoother_name << "\n";
        return 1;
    }

    std::cout << "\n=== Multigrid V-cycle 2D ===\n"
              << "grid:     " << n << "x" << n << " em [0,1]x[0,1]\n"
              << "smoother: " << smoother_name << "\n"
              << "tol:      " << tol << "\n\n";

    // cria grid com n intervalos em cada direcao
    Grid2D grid(n, n, 1.0, 1.0);

    // Equação: −∇²u(x,y) = 2π²sin(πx)sin(πy)
    // Solução analítica: u(x,y) = sin(πx) sin(πy)
    for (int i = 1; i < grid.nx; i++) {
        for (int j = 1; j < grid.ny; j++) {
            double x = i * grid.hx;
            double y = j * grid.hy;
            grid.f[grid.idx(i, j)] = 2.0 * M_PI * M_PI * sin(M_PI * x) * sin(M_PI * y);
        }
    }

    // resolve com V-cycles ate convergir
    auto t_start = std::chrono::high_resolution_clock::now();

    int k;
    for (k = 1; k <= max_vcycles; k++) {
        v_cycle(grid, smooth);
        double res = residual_norm(grid);
        std::cout << "v-cycle " << k << "  residuo = " << res << std::endl;

        if (res < tol) {
            std::cout << "\nConvergiu em " << k << " v-cycles.\n";
            break;
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    // calcula erro contra solucao analitica
    double max_err = 0.0;
    for (int i = 1; i < grid.nx; i++) {
        for (int j = 1; j < grid.ny; j++) {
            double x = i * grid.hx;
            double y = j * grid.hy;
            double u_exact = sin(M_PI * x) * sin(M_PI * y);
            double err = fabs(grid.u[grid.idx(i, j)] - u_exact);
            if (err > max_err) max_err = err;
        }
    }

    std::cout << "\n=== Resultados ===\n"
              << "residuo final:  " << residual_norm(grid) << "\n"
              << "erro maximo:    " << max_err << "\n"
              << "tempo total:    " << elapsed_ms << " ms\n";

    return 0;
}

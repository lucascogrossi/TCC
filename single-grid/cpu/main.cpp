#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <functional>

#include "../../multigrid/cpu/grid.h"
#include "../../multigrid/cpu/smoothers.h"
#include "../../multigrid/cpu/multigrid_utils.h"

using Smoother = std::function<void(Grid2D&)>;

void print_usage() {
    std::cout << "Uso: ./single_grid_cpu --n <grid_size> --smoother <smoother> [--tol <tol>] [--max_iters <n>]\n"
              << "\n"
              << "Argumentos:\n"
              << "  --n          Tamanho do grid (potencia de 2: 64, 128, 256, ...)\n"
              << "  --smoother   Metodo de suavizacao:\n"
              << "                 jacobi, jacobi_amortecido, gauss_seidel,\n"
              << "                 gauss_seidel_rb, sor\n"
              << "  --tol        Tolerancia para convergencia (default: 1e-6)\n"
              << "  --max_iters  Numero maximo de iteracoes (default: 100000)\n"
              << "\n"
              << "Exemplo:\n"
              << "  ./single_grid_cpu --n 256 --smoother gauss_seidel_rb --tol 1e-6\n";
}

int main(int argc, char* argv[]) {

    int n = 0;
    std::string smoother_name;
    int max_iters = 100000;
    double tol = 1e-6;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--n" && i + 1 < argc)
            n = std::atoi(argv[++i]);
        else if (arg == "--smoother" && i + 1 < argc)
            smoother_name = argv[++i];
        else if (arg == "--max_iters" && i + 1 < argc)
            max_iters = std::atoi(argv[++i]);
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

    if (n == 0 || smoother_name.empty()) {
        std::cerr << "Erro: --n e --smoother sao obrigatorios.\n\n";
        print_usage();
        return 1;
    }

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

    std::cout << "\n=== Single-grid 2D ===\n"
              << "grid:      " << n << "x" << n << " em [0,1]x[0,1]\n"
              << "smoother:  " << smoother_name << "\n"
              << "max_iters: " << max_iters << "\n"
              << "tol:       " << tol << "\n\n";

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

    auto t_start = std::chrono::high_resolution_clock::now();

    int k;
    for (k = 1; k <= max_iters; k++) {
        smooth(grid);
        double res = residual_norm(grid);
        std::cout << "iter " << k << "  residuo = " << res << "\n";
        if (res < tol) 
            break;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    double max_err = 0.0;
    for (int i = 1; i < grid.nx; i++) {
        for (int j = 1; j < grid.ny; j++) {
            double x = i * grid.hx;
            double y = j * grid.hy;
            double u_exact = sin(M_PI * x) * sin(M_PI * y);
            double err = fabs(grid.u[grid.idx(i, j)] - u_exact);
            if (err > max_err) 
                max_err = err;
        }
    }

    std::cout << "\n=== Resultados ===\n"
              << "residuo final:  " << residual_norm(grid) << "\n"
              << "erro maximo:    " << max_err << "\n"
              << "tempo total:    " << elapsed_ms << " ms\n";

    return 0;
}

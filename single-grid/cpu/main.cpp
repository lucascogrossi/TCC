#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <chrono>
#include <functional>

#include "../../multigrid/cpu/grid.h"
#include "../../multigrid/cpu/smoothers.h"
#include "../../multigrid/cpu/multigrid_utils.h"

using Smoother = std::function<void(Grid2D&)>;

void print_usage() {
    std::cout << "Uso: ./sg_cpu <n> <smoother> [tol] [max_iters] [--csv]\n"
              << "\n"
              << "Argumentos:\n"
              << "  n          Tamanho do grid nxn (potencia de 2: 64, 128, 256, ...)\n"
              << "  smoother   jacobi | jacobi_amortecido | gauss_seidel | gauss_seidel_rb | sor\n"
              << "  tol        Tolerancia para convergencia (default: 1e-6)\n"
              << "  max_iters  Numero maximo de iteracoes (default: 100000)\n"
              << "  --csv      Saida em formato CSV (para benchmark.sh)\n"
              << "\n"
              << "Exemplo:\n"
              << "  ./sg_cpu 256 gauss_seidel_rb\n"
              << "  ./sg_cpu 256 gauss_seidel_rb 1e-8 50000\n";
}

int main(int argc, char* argv[]) {

    if (argc < 3 || std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
        print_usage();
        return argc < 3 ? 1 : 0;
    }

    // checa flag --csv em qualquer posicao
    bool csv_output = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--csv") == 0)
            csv_output = true;
    }

    int n = std::atoi(argv[1]);
    std::string smoother_name = argv[2];
    double tol = (argc > 3 && strcmp(argv[3], "--csv") != 0) ? std::atof(argv[3]) : 1e-6;
    int max_iters = (argc > 4 && strcmp(argv[4], "--csv") != 0) ? std::atoi(argv[4]) : 100000;

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

    if (!csv_output) {
        std::cout << "\n=== Single-grid 2D ===\n"
                  << "grid:      " << n << "x" << n << " em [0,1]x[0,1]\n"
                  << "smoother:  " << smoother_name << "\n"
                  << "max_iters: " << max_iters << "\n"
                  << "tol:       " << tol << "\n\n";
    }

    Grid2D grid(n, n, 1.0, 1.0);

    // Equacao: -nabla^2 u(x,y) = 2*pi^2*sin(pi*x)*sin(pi*y)
    // Solucao analitica: u(x,y) = sin(pi*x) * sin(pi*y)
    for (int i = 1; i < grid.nx; i++) {
        for (int j = 1; j < grid.ny; j++) {
            double x = i * grid.hx;
            double y = j * grid.hy;
            grid.f[grid.idx(i, j)] = 2.0 * M_PI * M_PI * sin(M_PI * x) * sin(M_PI * y);
        }
    }

    auto t_start = std::chrono::high_resolution_clock::now();

    int k;
    double res = 0.0;
    for (k = 1; k <= max_iters; k++) {
        smooth(grid);
        res = residual_norm(grid);
        if (!csv_output)
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

    if (csv_output) {
        // sg,cpu,n,smoother,iterations,residual,max_error,time_ms
        std::cout << "sg,cpu," << n << "," << smoother_name << ","
                  << k << "," << res << "," << max_err << "," << elapsed_ms << "\n";
    } else {
        std::cout << "\n=== Resultados ===\n"
                  << "residuo final:  " << res << "\n"
                  << "erro maximo:    " << max_err << "\n"
                  << "tempo total:    " << elapsed_ms << " ms\n";
    }

    return 0;
}

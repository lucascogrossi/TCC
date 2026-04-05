#include <iostream>
#include <vector>
#include <cmath>

#include "grid_device.cuh"
#include "multigrid_utils.cuh"
#include "vcycle.cuh"

void print_usage() {
    std::cout << "Uso: ./multigrid_cuda <n> <smoother> [tol] [max_iters]\n"
              << "\n"
              << "Argumentos:\n"
              << "  n          Tamanho do grid (potencia de 2: 64, 128, 256, ...)\n"
              << "  smoother   jacobi | jacobi_amortecido | gauss_seidel_rb\n"
              << "  tol        Tolerancia para convergencia (default: 1e-6)\n"
              << "  max_iters  Numero maximo de v-cycles (default: 10000)\n"
              << "\n"
              << "Exemplo:\n"
              << "  ./multigrid_cuda 256 jacobi_amortecido\n"
              << "  ./multigrid_cuda 256 gauss_seidel_rb 1e-8 500\n";
}

int main(int argc, char* argv[]) {

    if (argc < 3 || std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
        print_usage();
        return argc < 3 ? 1 : 0;
    }

    int n = std::atoi(argv[1]);
    std::string smoother_name = argv[2];
    double tol = (argc > 3) ? std::atof(argv[3]) : 1e-6;
    int max_vcycles = (argc > 4) ? std::atoi(argv[4]) : 10000;

    SmootherType smoother;
    if (smoother_name == "jacobi")
        smoother = JACOBI;
    else if (smoother_name == "jacobi_amortecido")
        smoother = JACOBI_AMORTECIDO;
    else if (smoother_name == "gauss_seidel_rb")
        smoother = GAUSS_SEIDEL_RB;
    else {
        std::cerr << "Smoother invalido: " << smoother_name << "\n"
                  << "Opcoes: jacobi | jacobi_amortecido | gauss_seidel_rb\n";
        return 1;
    }

    std::cout << "\n=== Multigrid V-cycle 2D (CUDA) ===\n"
              << "grid:      " << n << "x" << n << " em [0,1]x[0,1]\n"
              << "smoother:  " << smoother_name << "\n"
              << "max_iters: " << max_vcycles << "\n"
              << "tol:       " << tol << "\n\n";

    // pre-aloca hierarquia de grids em unified memory
    std::vector<Grid2D*> grids;
    int nx = n;
    while (nx >= 2) {
        Grid2D* g;
        cudaMallocManaged(&g, sizeof(Grid2D));
        new (g) Grid2D(nx, nx, 1.0, 1.0);
        grids.push_back(g);
        nx /= 2;
    }

    // buffer para acumular soma da reducao na GPU
    double* d_result;
    cudaMallocManaged(&d_result, sizeof(double));

    // inicializa f no grid fino
    // Equação: −∇²u(x,y) = 2π²sin(πx)sin(πy)
    // Solução analítica: u(x,y) = sin(πx) sin(πy)
    Grid2D* fine = grids[0];
    for (int i = 1; i < fine->nx; i++) {
        for (int j = 1; j < fine->ny; j++) {
            double x = i * fine->hx;
            double y = j * fine->hy;
            fine->f[fine->idx(i, j)] = 2.0 * M_PI * M_PI * sin(M_PI * x) * sin(M_PI * y);
        }
    }

    // mede tempo com cudaEvent
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int k;
    for (k = 1; k <= max_vcycles; k++) {
        v_cycle(grids, smoother);
        double res = residual_norm_gpu(fine, d_result);
        std::cout << "v-cycle " << k << "  residuo = " << res << "\n";
        if (res < tol)
            break;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_ms = 0;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaDeviceSynchronize();

    // calcula erro maximo contra solucao analitica na CPU
    double max_err = 0.0;
    for (int i = 1; i < fine->nx; i++) {
        for (int j = 1; j < fine->ny; j++) {
            double x = i * fine->hx;
            double y = j * fine->hy;
            double u_exact = sin(M_PI * x) * sin(M_PI * y);
            double err = fabs(fine->u[fine->idx(i, j)] - u_exact);
            if (err > max_err) max_err = err;
        }
    }

    double residuo_final = residual_norm_gpu(fine, d_result);
    std::cout << "\n=== Resultados ===\n"
              << "residuo final:  " << residuo_final << "\n"
              << "erro maximo:    " << max_err << "\n"
              << "tempo total:    " << elapsed_ms << " ms\n";

    // libera memoria
    cudaFree(d_result);
    for (auto g : grids) {
        cudaFree(g->u);
        cudaFree(g->u_new);
        cudaFree(g->f);
        cudaFree(g->r);
        cudaFree(g->e);
        g->~Grid2D();
        cudaFree(g);
    }

    return 0;
}

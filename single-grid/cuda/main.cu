#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>

#include "../../multigrid/cuda/cuda_check.cuh"
#include "../../multigrid/cuda/grid_device.cuh"
#include "../../multigrid/cuda/smoothers.cuh"
#include "../../multigrid/cuda/multigrid_utils.cuh"

void print_usage() {
    std::cout << "Uso: ./sg_cuda <n> <smoother> [tol] [max_iters]\n"
              << "\n"
              << "Argumentos:\n"
              << "  n          Tamanho do grid (potencia de 2: 64, 128, 256, ...)\n"
              << "  smoother   jacobi | jacobi_amortecido | gauss_seidel_rb\n"
              << "  tol        Tolerancia para convergencia (default: 1e-6)\n"
              << "  max_iters  Numero maximo de iteracoes (default: 100)\n"
              << "\n"
              << "Exemplo:\n"
              << "  ./sg_cuda 256 gauss_seidel_rb\n"
              << "  ./sg_cuda 256 jacobi_amortecido 1e-8 50000\n";
}

int main(int argc, char* argv[]) {

    if (argc < 3 || std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
        print_usage();
        return argc < 3 ? 1 : 0;
    }

    int n = std::atoi(argv[1]);
    std::string smoother_name = argv[2];
    double tol = (argc > 3) ? std::atof(argv[3]) : 1e-6;
    int max_iters = (argc > 4) ? std::atoi(argv[4]) : 100;

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

    std::cout << "\n=== Single-grid 2D (CUDA) ===\n"
              << "grid:      " << n << "x" << n << " em [0,1]x[0,1]\n"
              << "smoother:  " << smoother_name << "\n"
              << "max_iters: " << max_iters << "\n"
              << "tol:       " << tol << "\n\n";

    // aloca grid (struct no host, arrays no device)
    Grid2D* g = new Grid2D(n, n, 1.0, 1.0);

    double* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(double)));

    // inicializa f: -nabla^2 u = 2*pi^2*sin(pi*x)*sin(pi*y), solucao: u = sin(pi*x)*sin(pi*y)
    int grid_size = (g->nx+1) * (g->ny+1);
    std::vector<double> h_f(grid_size, 0.0);
    for (int i = 1; i < g->nx; i++) {
        for (int j = 1; j < g->ny; j++) {
            double x = i * g->h;
            double y = j * g->h;
            h_f[g->idx(i, j)] = 2.0 * M_PI * M_PI * sin(M_PI * x) * sin(M_PI * y);
        }
    }
    CUDA_CHECK(cudaMemcpy(g->f, h_f.data(), grid_size * sizeof(double), cudaMemcpyHostToDevice));

    dim3 numThreadsPerBlock(16, 16);
    dim3 numBlocks((g->ny + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x,
                   (g->nx + numThreadsPerBlock.y - 1) / numThreadsPerBlock.y);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    int k;
    double res = 0.0;
    for (k = 1; k <= max_iters; k++) {
        smooth_grid(g, smoother, numBlocks, numThreadsPerBlock);
        res = residual_norm_gpu(g, d_result);
        std::cout << "iter " << k << "  residuo = " << res << "\n";
        if (res < tol)
            break;
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaDeviceSynchronize());

    // calcula erro maximo contra solucao analitica
    std::vector<double> h_u(grid_size);
    CUDA_CHECK(cudaMemcpy(h_u.data(), g->u, grid_size * sizeof(double), cudaMemcpyDeviceToHost));
    double max_err = 0.0;
    for (int i = 1; i < g->nx; i++) {
        for (int j = 1; j < g->ny; j++) {
            double x = i * g->h;
            double y = j * g->h;
            double u_exact = sin(M_PI * x) * sin(M_PI * y);
            double err = fabs(h_u[g->idx(i, j)] - u_exact);
            if (err > max_err) max_err = err;
        }
    }

    std::cout << "\n=== Resultados ===\n"
              << "residuo final:  " << res << "\n"
              << "erro maximo:    " << max_err << "\n"
              << "tempo total:    " << elapsed_ms << " ms\n";

    // libera memoria
    CUDA_CHECK(cudaFree(d_result));
    CUDA_CHECK(cudaFree(g->u));
    CUDA_CHECK(cudaFree(g->u_new));
    CUDA_CHECK(cudaFree(g->f));
    CUDA_CHECK(cudaFree(g->r));
    CUDA_CHECK(cudaFree(g->e));
    delete g;

    return 0;
}

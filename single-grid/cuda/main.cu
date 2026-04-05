#include <iostream>
#include <cmath>

#include "../../multigrid/cuda/grid_device.cuh"
#include "../../multigrid/cuda/smoothers.cuh"
#include "../../multigrid/cuda/multigrid_utils.cuh"

enum SmootherType { JACOBI, JACOBI_AMORTECIDO, GAUSS_SEIDEL_RB };

void print_usage() {
    std::cout << "Uso: ./sg_cuda <n> <smoother> [tol] [max_iters]\n"
              << "\n"
              << "Argumentos:\n"
              << "  n          Tamanho do grid (potencia de 2: 64, 128, 256, ...)\n"
              << "  smoother   jacobi | jacobi_amortecido | gauss_seidel_rb\n"
              << "  tol        Tolerancia para convergencia (default: 1e-6)\n"
              << "  max_iters  Numero maximo de iteracoes (default: 100000)\n"
              << "\n"
              << "Exemplo:\n"
              << "  ./sg_cuda 256 gauss_seidel_rb\n"
              << "  ./sg_cuda 256 jacobi_amortecido 1e-8 50000\n";
}

void smooth_once(Grid2D* g, SmootherType smoother, dim3 numBlocks, dim3 numThreads) {
    if (smoother == GAUSS_SEIDEL_RB) {
        gauss_seidel_rb_kernel<<<numBlocks, numThreads>>>(g, 0);
        cudaDeviceSynchronize();
        gauss_seidel_rb_kernel<<<numBlocks, numThreads>>>(g, 1);
        cudaDeviceSynchronize();
    } else {
        if (smoother == JACOBI)
            jacobi_kernel<<<numBlocks, numThreads>>>(g, g->u_new);
        else
            jacobi_amortecido_kernel<<<numBlocks, numThreads>>>(g, g->u_new);
        cudaDeviceSynchronize();
        std::swap(g->u, g->u_new);
    }
}

int main(int argc, char* argv[]) {

    if (argc < 3 || std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
        print_usage();
        return argc < 3 ? 1 : 0;
    }

    int n = std::atoi(argv[1]);
    std::string smoother_name = argv[2];
    double tol = (argc > 3) ? std::atof(argv[3]) : 1e-6;
    int max_iters = (argc > 4) ? std::atoi(argv[4]) : 100000;

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

    // aloca grid em unified memory
    Grid2D* g;
    cudaMallocManaged(&g, sizeof(Grid2D));
    new (g) Grid2D(n, n, 1.0, 1.0);

    double* d_result;
    cudaMallocManaged(&d_result, sizeof(double));

    // inicializa f: −∇²u = 2π²sin(πx)sin(πy), solucao: u = sin(πx)sin(πy)
    for (int i = 1; i < g->nx; i++) {
        for (int j = 1; j < g->ny; j++) {
            double x = i * g->hx;
            double y = j * g->hy;
            g->f[g->idx(i, j)] = 2.0 * M_PI * M_PI * sin(M_PI * x) * sin(M_PI * y);
        }
    }

    dim3 numThreadsPerBlock(16, 16);
    dim3 numBlocks((g->ny + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x,
                   (g->nx + numThreadsPerBlock.y - 1) / numThreadsPerBlock.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int k;
    for (k = 1; k <= max_iters; k++) {
        smooth_once(g, smoother, numBlocks, numThreadsPerBlock);
        double res = residual_norm_gpu(g, d_result);
        std::cout << "iter " << k << "  residuo = " << res << "\n";
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

    // calcula erro maximo contra solucao analitica
    double max_err = 0.0;
    for (int i = 1; i < g->nx; i++) {
        for (int j = 1; j < g->ny; j++) {
            double x = i * g->hx;
            double y = j * g->hy;
            double u_exact = sin(M_PI * x) * sin(M_PI * y);
            double err = fabs(g->u[g->idx(i, j)] - u_exact);
            if (err > max_err) max_err = err;
        }
    }

    std::cout << "\n=== Resultados ===\n"
              << "residuo final:  " << residual_norm_gpu(g, d_result) << "\n"
              << "erro maximo:    " << max_err << "\n"
              << "tempo total:    " << elapsed_ms << " ms\n";

    // libera memoria
    cudaFree(d_result);
    cudaFree(g->u);
    cudaFree(g->u_new);
    cudaFree(g->f);
    cudaFree(g->r);
    cudaFree(g->e);
    g->~Grid2D();
    cudaFree(g);

    return 0;
}

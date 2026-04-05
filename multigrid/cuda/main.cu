#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>

#include "cuda_check.cuh"
#include "grid_device.cuh"
#include "multigrid_utils.cuh"
#include "vcycle.cuh"

void print_usage() {
    std::cout << "Uso: ./multigrid_cuda <n> <smoother> [tol] [max_iters] [--csv]\n"
              << "\n"
              << "Argumentos:\n"
              << "  n          Tamanho do grid (potencia de 2: 64, 128, 256, ...)\n"
              << "  smoother   jacobi | jacobi_amortecido | gauss_seidel_rb\n"
              << "  tol        Tolerancia para convergencia (default: 1e-6)\n"
              << "  max_iters  Numero maximo de v-cycles (default: 10000)\n"
              << "  --csv      Saida em formato CSV (para benchmarks)\n"
              << "\n"
              << "Exemplo:\n"
              << "  ./multigrid_cuda 256 jacobi_amortecido\n"
              << "  ./multigrid_cuda 256 gauss_seidel_rb 1e-8 500\n"
              << "  ./multigrid_cuda 256 gauss_seidel_rb 1e-6 10000 --csv\n";
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
    int max_vcycles = (argc > 4 && strcmp(argv[4], "--csv") != 0) ? std::atoi(argv[4]) : 10000;

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

    if (!csv_output) {
        std::cout << "\n=== Multigrid V-cycle 2D (CUDA) ===\n"
                  << "grid:      " << n << "x" << n << " em [0,1]x[0,1]\n"
                  << "smoother:  " << smoother_name << "\n"
                  << "max_iters: " << max_vcycles << "\n"
                  << "tol:       " << tol << "\n\n";
    }

    // pre-aloca hierarquia de grids (struct no host, arrays no device)
    std::vector<Grid2D*> grids;
    int nx = n;
    while (nx >= 2) {
        Grid2D* g = new Grid2D(nx, nx, 1.0, 1.0);
        grids.push_back(g);
        nx /= 2;
    }

    // buffer para acumular soma da reducao na GPU
    double* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(double)));

    // inicializa f no grid fino
    // Equacao: -nabla^2 u(x,y) = 2*pi^2*sin(pi*x)*sin(pi*y)
    // Solucao analitica: u(x,y) = sin(pi*x) * sin(pi*y)
    Grid2D* fine = grids[0];
    int fine_size = (fine->nx+1) * (fine->ny+1);
    std::vector<double> h_f(fine_size, 0.0);
    for (int i = 1; i < fine->nx; i++) {
        for (int j = 1; j < fine->ny; j++) {
            double x = i * fine->hx;
            double y = j * fine->hy;
            h_f[fine->idx(i, j)] = 2.0 * M_PI * M_PI * sin(M_PI * x) * sin(M_PI * y);
        }
    }
    CUDA_CHECK(cudaMemcpy(fine->f, h_f.data(), fine_size * sizeof(double), cudaMemcpyHostToDevice));

    // mede tempo com cudaEvent
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    int k;
    double res = 0.0;
    for (k = 1; k <= max_vcycles; k++) {
        v_cycle(grids, smoother);
        res = residual_norm_gpu(fine, d_result);
        if (!csv_output)
            std::cout << "v-cycle " << k << "  residuo = " << res << "\n";
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

    // calcula erro maximo contra solucao analitica na CPU
    std::vector<double> h_u(fine_size);
    CUDA_CHECK(cudaMemcpy(h_u.data(), fine->u, fine_size * sizeof(double), cudaMemcpyDeviceToHost));
    double max_err = 0.0;
    for (int i = 1; i < fine->nx; i++) {
        for (int j = 1; j < fine->ny; j++) {
            double x = i * fine->hx;
            double y = j * fine->hy;
            double u_exact = sin(M_PI * x) * sin(M_PI * y);
            double err = fabs(h_u[fine->idx(i, j)] - u_exact);
            if (err > max_err) max_err = err;
        }
    }

    if (csv_output) {
        // mg,cuda,n,smoother,iterations,residual,max_error,time_ms
        std::cout << "mg,cuda," << n << "," << smoother_name << ","
                  << k << "," << res << "," << max_err << "," << elapsed_ms << "\n";
    } else {
        std::cout << "\n=== Resultados ===\n"
                  << "residuo final:  " << res << "\n"
                  << "erro maximo:    " << max_err << "\n"
                  << "tempo total:    " << elapsed_ms << " ms\n";
    }

    // libera memoria
    CUDA_CHECK(cudaFree(d_result));
    for (auto g : grids) {
        CUDA_CHECK(cudaFree(g->u));
        CUDA_CHECK(cudaFree(g->u_new));
        CUDA_CHECK(cudaFree(g->f));
        CUDA_CHECK(cudaFree(g->r));
        CUDA_CHECK(cudaFree(g->e));
        delete g;
    }

    return 0;
}

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>

#include "cuda_check.cuh"
#include "grid_device.cuh"
#include "multigrid_utils.cuh"
#include "vcycle.cuh"

// Reduz (u - u_exact)^2 na GPU para a solucao analitica do problema modelo:
// u_exact(x,y) = (x^2 - x^4)(y^4 - y^2)
__global__ void error_l2_kernel(Grid2D grid, double* result) {
    extern __shared__ double sdata[];

    int j   = blockIdx.x * blockDim.x + threadIdx.x;
    int i   = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    double val = 0.0;
    if (i >= 1 && i < grid.nx && j >= 1 && j < grid.ny) {
        double x = i * grid.h;
        double y = j * grid.h;
        double u_exact = (x*x - x*x*x*x) * (y*y*y*y - y*y);
        double diff = grid.u[grid.idx(i, j)] - u_exact;
        val = diff * diff;
    }
    sdata[tid] = val;
    __syncthreads();

    for (unsigned int stride = (blockDim.x * blockDim.y) / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    if (tid == 0)
        atomicAdd(result, sdata[0]);
}

__host__ double error_l2_gpu(Grid2D* grid, double* d_result) {
    dim3 numThreadsPerBlock(16, 16);
    dim3 numBlocks((grid->ny + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x,
                   (grid->nx + numThreadsPerBlock.y - 1) / numThreadsPerBlock.y);
    int sharedMem = numThreadsPerBlock.x * numThreadsPerBlock.y * sizeof(double);

    CUDA_CHECK(cudaMemset(d_result, 0, sizeof(double)));
    error_l2_kernel<<<numBlocks, numThreadsPerBlock, sharedMem>>>(*grid, d_result);

    double h_result;
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost));
    return sqrt(h_result * grid->h * grid->h);
}

void print_usage() {
    std::cout << "Uso: ./multigrid_cuda <n> <smoother> [tol] [max_iters]\n"
              << "\n"
              << "Argumentos:\n"
              << "  n          Tamanho do grid (potencia de 2: 64, 128, 256, ...)\n"
              << "  smoother   jacobi | jacobi_amortecido | gauss_seidel_rb\n"
              << "  tol        Tolerancia para convergencia (default: 1e-6)\n"
              << "  max_iters  Numero maximo de v-cycles (default: 100)\n"
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
    int max_vcycles = (argc > 4) ? std::atoi(argv[4]) : 100;

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
    // -∇²u(x,y) = 2[(1-6x²)y²(1-y²) + (1-6y²)x²(1-x²)]
    // Solução analítica: u(x,y) = (x²-x⁴)(y⁴-y²)
    // Referência: Briggs, Henson & McCormick (2000), *A Multigrid Tutorial*, eq. (4.8).
    Grid2D* fine = grids[0];
    int fine_size = (fine->nx+1) * (fine->ny+1);
    std::vector<double> h_f(fine_size, 0.0);
    for (int i = 1; i < fine->nx; i++) {
        for (int j = 1; j < fine->ny; j++) {
            double x = i * fine->h;
            double y = j * fine->h;
            h_f[fine->idx(i, j)] = 2.0 * ((1.0 - 6.0*x*x) * y*y * (1.0 - y*y) + (1.0 - 6.0*y*y) * x*x * (1.0 - x*x));
        }
    }
    CUDA_CHECK(cudaMemcpy(fine->f, h_f.data(), fine_size * sizeof(double), cudaMemcpyHostToDevice));

    // mede tempo com cudaEvent
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    std::cout << "v-cycle 0  residuo = " << residual_norm_gpu(fine, d_result)
              << "  erro = " << error_l2_gpu(fine, d_result) << "\n";

    int k;
    double res = 0.0;
    for (k = 1; k <= max_vcycles; k++) {
        v_cycle(grids, smoother);
        res = residual_norm_gpu(fine, d_result);
        std::cout << "v-cycle " << k << "  residuo = " << res
                  << "  erro = " << error_l2_gpu(fine, d_result) << "\n";
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

    // calcula erro contra solucao analitica na CPU
    std::vector<double> h_u(fine_size);
    CUDA_CHECK(cudaMemcpy(h_u.data(), fine->u, fine_size * sizeof(double), cudaMemcpyDeviceToHost));
    double max_err = 0.0;
    double err_l2_tmp = 0.0;

    for (int i = 1; i < fine->nx; i++) {
        for (int j = 1; j < fine->ny; j++) {
            double x = i * fine->h;
            double y = j * fine->h;
            double u_exact = (x*x - x*x*x*x) * (y*y*y*y - y*y);
            double diff = h_u[fine->idx(i, j)] - u_exact;

            // Erro norma infinita (máximo absoluto)
            double err = fabs(diff);
                if (err > max_err)
                    max_err = err;

            // Acumula quadrados para norma L2
            err_l2_tmp += diff * diff;
        }

    }
    // Norma L2 discreta
    double err_l2 = sqrt(err_l2_tmp * fine->h * fine->h);

    std::cout << "\n=== Resultados ===\n"
              << "residuo final:  " << res << "\n"
              << "erro L-inf:     " << max_err << "\n"
              << "erro L2:        " << err_l2 << "\n"
              << "tempo total:    " << elapsed_ms << " ms\n";

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

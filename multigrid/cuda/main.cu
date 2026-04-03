#include <iostream>
#include <vector>
#include <cmath>

#include "grid_device.cuh"
#include "multigrid_utils.cuh"
#include "vcycle.cuh"

void print_usage() {
    std::cout << "Uso: ./multigrid_cuda --n <grid_size>\n"
              << "\n"
              << "Argumentos:\n"
              << "  --n   Tamanho do grid (potencia de 2: 64, 128, 256, ...)\n"
              << "\n"
              << "Exemplo:\n"
              << "  ./multigrid_cuda --n 256\n";
}

int main(int argc, char* argv[]) {

    int n = 0;
    int max_vcycles = 10;

    // parse de argumentos
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--n" && i + 1 < argc)
            n = std::atoi(argv[++i]);
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

    if (n == 0) {
        std::cerr << "Erro: --n e obrigatorio.\n\n";
        print_usage();
        return 1;
    }

    std::cout << "\n=== Multigrid V-cycle 2D (CUDA) ===\n"
              << "grid:     " << n << "x" << n << " em [0,1]x[0,1]\n"
              << "smoother: jacobi_amortecido\n\n";

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

    for (int k = 1; k <= max_vcycles; k++) {
        v_cycle(grids);
        std::cout << "v-cycle " << k << "\n";
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_ms = 0;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // garante que dados estao disponiveis na CPU (unified memory)
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

    // calcula residuo final na CPU
    double hx2 = fine->hx * fine->hx;
    double hy2 = fine->hy * fine->hy;
    double norm = 0.0;
    for (int i = 1; i < fine->nx; i++) {
        for (int j = 1; j < fine->ny; j++) {
            double Au = (-fine->u[fine->idx(i-1,j)] + 2*fine->u[fine->idx(i,j)] - fine->u[fine->idx(i+1,j)]) / hx2
                      + (-fine->u[fine->idx(i,j-1)] + 2*fine->u[fine->idx(i,j)] - fine->u[fine->idx(i,j+1)]) / hy2;
            double r = fine->f[fine->idx(i,j)] - Au;
            norm += r * r;
        }
    }
    double residuo = sqrt(norm * fine->hx * fine->hy);

    std::cout << "\n=== Resultados ===\n"
              << "residuo final:  " << residuo << "\n"
              << "erro maximo:    " << max_err << "\n"
              << "tempo total:    " << elapsed_ms << " ms\n";

    // libera memoria
    for (auto g : grids) {
        cudaFree(g->u);
        cudaFree(g->f);
        cudaFree(g->r);
        cudaFree(g->e);
        g->~Grid2D();
        cudaFree(g);
    }

    return 0;
}

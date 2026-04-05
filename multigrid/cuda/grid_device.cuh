#ifndef GRID_DEVICE_H
#define GRID_DEVICE_H

#include <cstdio>
#include <cstdlib>

#include "cuda_check.cuh"

struct Grid2D {
    int nx, ny;            // intervalos em cada direção
    double hx, hy;         // tamanho de cada intervalo
    double Lx, Ly;         // comprimento do domínio
    double* u;             // solucao aproximada em cada ponto
    double* u_new;         // buffer temporario para jacobi
    double* f;             // termo fonte
    double* r;             // buffer temporario para o residuo
    double* e;             // buffer temporario de correcao do nivel

    // Construtor
    Grid2D(int nx, int ny, double Lx, double Ly)
        : nx(nx), ny(ny), Lx(Lx), Ly(Ly),
          hx(Lx / nx), hy(Ly / ny),
          u(nullptr), u_new(nullptr), f(nullptr), r(nullptr), e(nullptr) {
        if (nx <= 0 || ny <= 0 || (nx & (nx - 1)) != 0 || (ny & (ny - 1)) != 0) {
            fprintf(stderr, "Erro: nx e ny devem ser potencias de 2 (recebido nx=%d, ny=%d)\n", nx, ny);
            exit(EXIT_FAILURE);
        }
        int size = (nx+1) * (ny+1) * sizeof(double);
        CUDA_CHECK(cudaMalloc(&u, size));
        CUDA_CHECK(cudaMemset(u, 0, size));
        CUDA_CHECK(cudaMalloc(&u_new, size));
        CUDA_CHECK(cudaMemset(u_new, 0, size));
        CUDA_CHECK(cudaMalloc(&f, size));
        CUDA_CHECK(cudaMemset(f, 0, size));
        CUDA_CHECK(cudaMalloc(&r, size));
        CUDA_CHECK(cudaMemset(r, 0, size));
        CUDA_CHECK(cudaMalloc(&e, size));
        CUDA_CHECK(cudaMemset(e, 0, size));
    }

    // Indexacao row major
    __host__ __device__ int idx(int i, int j) const {
        return i * (ny+1) + j;
    }
};

#endif

#ifndef GRID_DEVICE_H
#define GRID_DEVICE_H

#include <stdexcept>

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
        int size = (nx+1) * (ny+1) * sizeof(double);
        cudaMallocManaged(&u, size);
        cudaMemset(u, 0, size);
        cudaMallocManaged(&u_new, size);
        cudaMemset(u_new, 0, size);
        cudaMallocManaged(&f, size);
        cudaMemset(f, 0, size);
        cudaMallocManaged(&r, size);
        cudaMemset(r, 0, size);
        cudaMallocManaged(&e, size);
        cudaMemset(e, 0, size);
    }

    // Indexacao row major
    __host__ __device__ int idx(int i, int j) const {
        return i * (ny+1) + j;
    }
};

#endif
#ifndef GRID_2D_H
#define GRID_2D_H

#include <vector>
#include <stdexcept>

struct Grid2D {
    int nx, ny;            // intervalos em cada direção
    double hx, hy;         // espaçamentos
    double Lx, Ly;         // comprimento do domínio
    std::vector<double> u; // solucao aproximada em cada ponto (i, j)
    std::vector<double> f; // termo fonte

    // Construtor
    Grid2D(int nx, int ny, double Lx, double Ly)
        : nx(nx), ny(ny), Lx(Lx), Ly(Ly),
          hx(Lx / nx), hy(Ly / ny),
          u((nx+1) * (ny+1), 0.0), f((nx+1) * (ny+1), 0.0) {
        if (nx <= 0 || ny <= 0 || (nx & (nx - 1)) != 0 || (ny & (ny - 1)) != 0)
            throw std::invalid_argument("nx e ny devem ser potencias de 2");
    }

    // Indexacao row major
    int idx(int i, int j) const {
        return i * (ny+1) + j;
    }
};

#endif
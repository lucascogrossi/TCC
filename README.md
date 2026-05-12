**Implementação e paralelização do Método Multigrid com V-ciclo em processadores gráficos de propósito geral (GPGPUs)**

#### Problema

Equação de Poisson 2D com condições de contorno de Dirichlet em [0,1]x[0,1]:

-∇²u(x,y) = 2[(1-6x²)y²(1-y²) + (1-6y²)x²(1-x²)]
Referência: Briggs, Henson & McCormick (2000), *A Multigrid Tutorial*, eq. (4.8).

Solução analítica: `u(x,y) = (x²-x⁴)(y⁴-y²)`

#### Build e execução

```bash
make all        # compila CPU e CUDA (requer nvcc para CUDA)
make cpu        # compila apenas CPU
./benchmark.sh  # roda todos os benchmarks e gera results/summary.csv
```

Executar individualmente:

```bash
./multigrid/cpu/mg_cpu <n> <smoother> [tol] [max_iters]
./multigrid/cuda/mg_cuda <n> <smoother> [tol] [max_iters]
# exemplo: ./multigrid/cpu/mg_cuda 256 gauss_seidel_rb 1e-8 100
```

#### Resultados

| Grid       | CPU (ms)  | CUDA (ms) | Speedup |
|------------|-----------|-----------|---------|
| 64x64      | -         | -         | -       |
| 128x128    | -         | -         | -       |
| 256x256    | -         | -         | -       |
| 512x512    | -         | -         | -       |
| 1024x1024  | -         | -         | -       |
| 2048x2048  | -         | -         | -       |
| 4096x4096  | -         | -         | -       |

- Smoother: Gauss-Seidel Red-Black (2x pre & pos smoothing)
- Tol: 1e-8
- CPU: AMD Ryzen 7 5700X3D
- GPU: NVIDIA GeForce RTX 5060
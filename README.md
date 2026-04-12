Trabalho de conclusão de curso

Implementação e paralelização do Método Multigrid com V-ciclo em processadores gráficos de propósito geral (GPGPUs)



| Grid       | CPU (ms)  | CUDA (ms) | Speedup |
|------------|-----------|-----------|---------|
| 64x64      | 0.53      | 3.68      | 0.14x   |
| 128x128    | 2.14      | 4.47      | 0.48x   |
| 256x256    | 12.70     | 5.38      | 2.4x    |
| 512x512    | 48.13     | 7.08      | 6.8x    |
| 1024x1024  | 197.75    | 13.07     | 15.1x   |
| 2048x2048  | 1151.60   | 40.02     | 28.8x   |
| 4096x4096  | 5013.56   | 151.11    | 33.2x   |

- Smoother: Gauss-Seidel Red-Black (2x pre & pos smoothing)
- Tol: 1e-8
- CPU: AMD Ryzen 7 5700X3D
- GPU: NVIDIA GeForce RTX 5060
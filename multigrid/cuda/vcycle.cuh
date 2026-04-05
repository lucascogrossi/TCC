#ifndef VCYCLE_CUDA_H
#define VCYCLE_CUDA_H

#include <vector>

#include "grid_device.cuh"
#include "multigrid_utils.cuh"
#include "smoothers.cuh"

// Recebe a hierarquia de grids pre alocadas
__host__ void v_cycle(std::vector<Grid2D*>& grids, SmootherType smoother) {

    // Descida ate o penultimo nivel da grid
    for (int i = 0; i < (int)grids.size() - 1; i++) {

        dim3 numThreadsPerBlock(16, 16);
        dim3 numBlocks((grids[i]->ny + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x,
                       (grids[i]->nx + numThreadsPerBlock.y - 1) / numThreadsPerBlock.y);

        // 1. Pre suavizacao
        for (int k = 0; k < 2; k++)
            smooth_grid(grids[i], smoother, numBlocks, numThreadsPerBlock);

        // 2. Calcula residuo
        compute_residual_kernel<<<numBlocks, numThreadsPerBlock>>>(grids[i], grids[i]->r);
        CUDA_CHECK(cudaDeviceSynchronize());

        // 3. Restrict — numBlocks calculado para o grid grosso (destino)
        dim3 numBlocksCoarse((grids[i+1]->ny + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x,
                             (grids[i+1]->nx + numThreadsPerBlock.y - 1) / numThreadsPerBlock.y);
        restriction_kernel<<<numBlocksCoarse, numThreadsPerBlock>>>(grids[i]->r, grids[i+1]->f, grids[i]->nx, grids[i]->ny);
        CUDA_CHECK(cudaDeviceSynchronize());

        // 4. zera u do prox nivel
        CUDA_CHECK(cudaMemset(grids[i+1]->u, 0, (grids[i+1]->nx+1)*(grids[i+1]->ny+1)*sizeof(double)));
    }

    // 5. Calcula direto na malha mais grossa q existe
    solve_coarse(grids.back(), 1);

    // Subida ate o nivel mais fino (primeiro nivel)
    for (int i = (int)grids.size()-2; i >= 0; i--) {

        dim3 numThreadsPerBlock(16, 16);
        dim3 numBlocks((grids[i]->ny + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x,
                       (grids[i]->nx + numThreadsPerBlock.y - 1) / numThreadsPerBlock.y);

        // 6. Prolongation: numBlocks calculado para o grid grosso (fonte)
        dim3 numBlocksCoarse((grids[i+1]->ny + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x,
                             (grids[i+1]->nx + numThreadsPerBlock.y - 1) / numThreadsPerBlock.y);
        prolongation_kernel<<<numBlocksCoarse, numThreadsPerBlock>>>(grids[i+1]->u, grids[i]->e, grids[i+1]->nx, grids[i+1]->ny);
        CUDA_CHECK(cudaDeviceSynchronize());

        // 7. Correct
        correct_kernel<<<numBlocks, numThreadsPerBlock>>>(grids[i], grids[i]->e);
        CUDA_CHECK(cudaDeviceSynchronize());

        // 8. Pos suavizacao
        for (int k = 0; k < 2; k++)
            smooth_grid(grids[i], smoother, numBlocks, numThreadsPerBlock);
    }
}

#endif

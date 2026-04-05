CXX      := g++
CXXFLAGS := -O2 -std=c++17
NVCC     := nvcc
NVFLAGS  := -O2 -std=c++17

HAVE_CUDA := $(shell which nvcc > /dev/null 2>&1 && echo 1 || echo 0)

# Targets
CPU_TARGETS := single-grid/cpu/sg_cpu multigrid/cpu/mg_cpu

ifeq ($(HAVE_CUDA),1)
CUDA_TARGETS := single-grid/cuda/sg_cuda multigrid/cuda/mg_cuda
else
CUDA_TARGETS :=
endif

ALL_TARGETS := $(CPU_TARGETS) $(CUDA_TARGETS)

.PHONY: all clean cpu cuda

all: $(ALL_TARGETS)
	@echo "Built: $(ALL_TARGETS)"
ifeq ($(HAVE_CUDA),0)
	@echo "(CUDA not found — only CPU targets were compiled)"
endif

cpu: $(CPU_TARGETS)

cuda: $(CUDA_TARGETS)

# CPU
single-grid/cpu/sg_cpu: single-grid/cpu/main.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

multigrid/cpu/mg_cpu: multigrid/cpu/main.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

# CUDA
single-grid/cuda/sg_cuda: single-grid/cuda/main.cu
	$(NVCC) $(NVFLAGS) -o $@ $<

multigrid/cuda/mg_cuda: multigrid/cuda/main.cu
	$(NVCC) $(NVFLAGS) -o $@ $<

clean:
	rm -f $(CPU_TARGETS) $(CUDA_TARGETS)

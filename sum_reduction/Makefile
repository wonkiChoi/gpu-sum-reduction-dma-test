CUDA_INCLUDEPATH=/usr/local/cuda-5.0/include
NVCC_FLAGS=-I/usr/local/include/donard
NVCC_OPTS=-O3 -arch=sm_50 -Xcompiler -Wall -Xcompiler -Wextra -m64 -ldonard

reduce: main.cu reduce.o Makefile
	nvcc -o reduce main.cu reduce.o $(NVCC_OPTS) $(NVCC_FLAGS)

reduce.o: reduce.cu
	nvcc -c reduce.cu $(NVCC_OPTS) $(NVCC_FLAGS)

clean:
	rm -f *.o reduce

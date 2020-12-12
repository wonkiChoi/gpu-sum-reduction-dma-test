#include <iostream>
#include <ctime>
#include <fstream>

#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utils.h"

#include "reduce.h"
extern "C" {
#include <pinpool.h>
#include <filemap.h>
}

using namespace std;

void generate_input(unsigned int* input, unsigned int input_len)
{
	for (unsigned int i = 0; i < input_len; ++i)
	{
		input[i] = i;
	}
}

unsigned int cpu_simple_sum(unsigned int* h_in, unsigned int h_in_len)
{
	unsigned int total_sum = 0;

	for (unsigned int i = 0; i < h_in_len; ++i)
	{
		total_sum = total_sum + h_in[i];
	}

	return total_sum;
}

int main()
{
	// Set up clock for timing comparisons
	std::clock_t start;
	double duration;
        int check = pinpool_init(1, 67108864);

	for (int k = 1; k < 28; ++k)
	{
		unsigned int h_in_len = (1 << k);
		//unsigned int h_in_len = 2048;
		std::cout << "h_in_len: " << h_in_len << std::endl;
		unsigned int* t_in = new unsigned int[h_in_len];
		generate_input(t_in, h_in_len);
                FILE *testfile_out = fopen("/home/gpurocks/data_mysql/testfile.data", "wb");
                //ofstream testfile_out("/home/gpurocks/data_mysql/testfile.data", ios::out);
                for(unsigned int i = 0; i < h_in_len; i++)
                  fwrite((t_in + i), sizeof(unsigned int), 1, testfile_out);
                   //testfile_out << *(t_in + i);
                delete[] t_in;
                //testfile_out.close();
                fclose(testfile_out);

                start = std::clock();
                FILE *testfile_in = fopen("/home/gpurocks/data_mysql/testfile.data", "rb");
                //ifstream testfile_in("/home/gpurocks/data_mysql/testfile.data", ios::in);
                unsigned int* h_in = new unsigned int[h_in_len];
                for(unsigned int i = 0; i < h_in_len; i++)
                  if(!fread((h_in + i), sizeof(unsigned int), 1, testfile_in)) return 0;
                  //testfile_in >> *(h_in + i);
                //testfile_in.close();
                fclose(testfile_in);
 
                duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;              
                std::cout << "Read file time : " << duration << " s" << std::endl;

		// Do CPU sum for reference
		start = std::clock();
		unsigned int cpu_total_sum = cpu_simple_sum(h_in, h_in_len);
		duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
		std::cout << cpu_total_sum << std::endl;
		std::cout << "CPU time: " << duration << " s" << std::endl;

		// Do GPU scan
		start = std::clock();
                unsigned int* d_in;
                cudaMalloc(&d_in, sizeof(unsigned int) * h_in_len);
                cudaMemcpy(d_in, h_in, sizeof(unsigned int) * h_in_len, cudaMemcpyHostToDevice);

		unsigned int gpu_total_sum = gpu_sum_reduce(d_in, h_in_len);
		duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
		std::cout << gpu_total_sum << std::endl;
		std::cout << "GPU time: " << duration << " s" << std::endl;
                cudaFree(d_in);
                delete[] h_in;

                // Do GPU_DMA Scan
                start = std::clock();
                unsigned int* dma_in;     
                struct filemap* testfile_dma = filemap_open_cuda("/home/gpurocks/data_mysql/testfile.data");
                dma_in = (unsigned int*)testfile_dma->data;
                    
                unsigned int gpu_dma_total_sum = gpu_sum_reduce_dma(dma_in, h_in_len);
                duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
                std::cout << gpu_dma_total_sum << std::endl;
                std::cout << "GPU_DMA time: " << duration << " s" << std::endl;
                filemap_free(testfile_dma);


		bool match = (cpu_total_sum == gpu_total_sum) && (cpu_total_sum == gpu_dma_total_sum);
		std::cout << "Match: " << match << std::endl;

		std::cout << std::endl;
	}
}

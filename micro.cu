###########################################################################
#  micro -- GPU reliability microbenchmarks                               #
#                                                                         #
#  Copyright 2023-24 Jose M. Badia <barrachi@uji.es> and                  #
#                    German Leon <leon@uji.es>                            #
#                                                                         #
#  micro.cu is part of micro                                              #
#                                                                         #
#  micro is free software: you can redistribute it and/or modify          #
#  it under the terms of the GNU General Public License as published by   #
#  the Free Software Foundation; either version 3 of the License, or      #
#  (at your option) any later version.                                    #
#                                                                         #
#  micro is distributed in the hope that it will be useful, but           #
#  WITHOUT ANY WARRANTY; without even the implied warranty of             #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU      #
#  General Public License for more details.                               #
#                                                                         #
#  You should have received a copy of the GNU General Public License      #
#  along with this program.  If not, see <http://www.gnu.org/licenses/>   #
#                                                                         #
###########################################################################

// System includes
#include <unistd.h>
#include <stdio.h>
#include <stdbool.h>
#include <assert.h>
#include "jetson_nano.h"
// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>
#include <cuda_fp16.h>

//#include "input_device.h"
#define FREQ 921600000
#define T 1
#define BITSNOSIGNIFICATIVOS 16
#define CYCLES (T*(FREQ) >> BITSNOSIGNIFICATIVOS)
#define QUATUMINTERACIONES 1000
#define SIZEROW 1

#define myclock() (int) (clock64() >> BITSNOSIGNIFICATIVOS)

//#define SIM_ERROR

typedef int  btype;
typedef btype *btypePtr;

/**
 * Micro Kernel that performs the computation using only registers.
 * Version with a given number of iterations
 */
__global__ void microKernel_reg_iter (unsigned int nit, char *vadd) {

    btype regin, regout, local;
    btype id = (blockIdx.x*blockDim.x + threadIdx.x+1);

    regin = id;
    local = id;
#pragma unroll 2 
    for (int op = 0; op < nit; ++op) {
      regout = regin*local + id;
      local = (regout-local)/regin;
    }
    vadd[(int) id - 1] = (local == id);

}

/**
 * Micro Kernel that performs the computation using only registers 
 * Version with a given time
 */
__global__ void microKernel_reg_time (unsigned int cycles, char *vadd) {
    unsigned int fin,ahora;
    btype regin, regout, local;
    btype id = (blockIdx.x*blockDim.x + threadIdx.x+1);

    ahora=myclock();
    regin = id;
    local = id;
    fin=ahora+cycles;

    while (ahora < fin  )  {   
     ahora=myclock();

     #pragma unroll 2
     for (unsigned int op=0; op< QUATUMINTERACIONES;++op){
      regout = regin*local + id;
      
      local = (regout-local)/regin; 
      }  
    }  // end while

    vadd[(int) id - 1] = (local == id);

}

/**
 * Micro Kernel that performs the computation using global memory (and cache)
 * Version with a given number of iterations 
 */
__global__ void microKernel_global_iter(int nit, char *vadd, volatile btype *global) {
    btype regin, regout;
    btype id = (blockIdx.x*blockDim.x + threadIdx.x+1);
    int idInt = SIZEROW*(int) id;

    regin = id;
    global[idInt] = id;

    #pragma unroll 2 
    for (int op = 0; op < nit; ++op) {
      regout = regin*global[idInt] + id;
      global[idInt] = (regout-global[idInt])/regin;
    }
    vadd[(int) id - 1] = ( global[idInt] == id );
}

/**
 * Micro Kernel that performs the computation using global memory (and cache)
 * Version with a given time
 */
__global__ void microKernel_global_time(unsigned int cycles, char *vadd, volatile btype *global) {
    unsigned  int fin,ahora;
    btype regin, regout;
    btype id = (blockIdx.x*blockDim.x + threadIdx.x+1);
    volatile int idInt = SIZEROW*(int) id;

    ahora=myclock();
    regin = id;
    fin=ahora+cycles;
    global[idInt] = id;
    while (ahora < fin  )  {   
      ahora=myclock();

      #pragma unroll 2 
      for (unsigned  int op = 0; op < QUATUMINTERACIONES; ++op) {
        regout = regin*global[idInt] + id;
        global[idInt] = (regout-global[idInt])/regin;
      }
    }
    vadd[(int) id - 1] = ( global[idInt] == id );
}

/**
 * Micro Kernel that performs the computation using shared memory
 * Version with a given number of iterations
 */
__global__ void microKernel_shared_iter(unsigned int nit, char *vadd) {
    
  
    btype regin, regout;
    volatile btype id = (btype) (blockIdx.x*blockDim.x + threadIdx.x + 1);

    volatile extern __shared__ btype sh[];

    regin = id;
    sh[threadIdx.x] = id;

    #pragma unroll 2 
    for (unsigned int op = 0; op < nit; ++op) {
      regout = regin*sh[threadIdx.x] + id;
      sh[threadIdx.x] = (regout-sh[threadIdx.x])/regin;
    }
    vadd[(int) id - 1 ] = (sh[threadIdx.x] == id);
}

/**
 * Micro Kernel that performs the computation using shared memory
 * Version with a given number of time
 */
__global__ void microKernel_shared_time (unsigned int cycles, char *vadd) {
    
    unsigned int fin,ahora;
    btype regin, regout;
    volatile btype id = (btype) (blockIdx.x*blockDim.x + threadIdx.x + 1);

    volatile extern __shared__ btype sh[];
    ahora=myclock();
    regin = id;
    sh[threadIdx.x] = id;
    fin=ahora+cycles;

    while (ahora < fin  )  {   
     ahora=myclock();

     #pragma unroll 2 
     for (int op = 0; op < QUATUMINTERACIONES; ++op) {
       regout = regin*sh[threadIdx.x] + id;
       sh[threadIdx.x] = (regout-sh[threadIdx.x])/regin;
     }
    } 
    vadd[(int) id - 1 ] = (sh[threadIdx.x] == id);
}

/**
  * Checks if there is any error in the result of any thread
  * cont returns the number of threads with a wrong result
  */
bool check_error(char *h_vadd, int vsize, int *cont, int *id) {
    *cont = 0;
    for (int i = 0; i < vsize; i++) 
        if (!h_vadd[i] ) {
 	   (*cont)++;
           *id = (i+1);
	}
    return (*cont == 0);
}

/**
 * Run microKernel
 */
int launch_kernel(char *bench, int grid, int blk, unsigned int nitocycles,int time) {
    char *h_vadd;
    char *d_vadd;
    btypePtr d_global;
    int vsize = grid*blk;
   

    // Allocate CUDA events that we'll use for timing
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    h_vadd = (char *) malloc(vsize*sizeof(char));

    checkCudaErrors(cudaMalloc(&d_vadd, vsize*sizeof(char)));
    checkCudaErrors(cudaDeviceSynchronize());
 
    // Record the start event
    checkCudaErrors(cudaEventRecord(start));

    // Execute the kernel
  
    if (!strcmp(bench, "shm") ) {
        if(time) {
            microKernel_shared_time <<<grid, blk, blk*sizeof(btype)>>>(nitocycles, d_vadd);
        }
        else {
            microKernel_shared_iter <<<grid, blk, blk*sizeof(btype)>>>(nitocycles, d_vadd);
        } 
    } else if (!strcmp(bench, "glb") ) {
        checkCudaErrors(cudaMalloc(&d_global, SIZEROW*vsize*sizeof(btype)));
        if(time) {
            microKernel_global_time <<<grid, blk, blk*sizeof(btype)>>>(nitocycles, d_vadd, d_global);
        }                                                
        else {
            microKernel_global_iter <<<grid, blk, blk*sizeof(btype)>>>(nitocycles, d_vadd, d_global);
        }
    } else if (!strcmp(bench, "reg") ) {
        if(time) {
            microKernel_reg_time <<<grid, blk, blk*sizeof(btype)>>>(nitocycles, d_vadd);
        }
        else {
            microKernel_reg_iter <<<grid, blk, blk*sizeof(btype)>>>(nitocycles, d_vadd);
       } 
    } 


    // Record the stop event
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaEventRecord(stop));

    // Wait for the stop event to complete
    checkCudaErrors(cudaEventSynchronize(stop));

    // Compute and print the performance
    float msecTotal = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
//    printf("Kernel time= %.2f ms.\n", msecTotal);

    checkCudaErrors( cudaMemcpy(h_vadd, d_vadd, vsize*sizeof(char), cudaMemcpyDeviceToHost) );
    
    int cont, id;
    bool correct = check_error(h_vadd, vsize, &cont, &id);

    // Clean up memory
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaFree(d_vadd));
    if (!strcmp(bench, "glb") ) {
        checkCudaErrors(cudaFree(d_global));
    }
    free(h_vadd);

    if (!correct)
      printf("FAIL Num fails: %d, Thread id: %d\n", cont, id);

    return correct;

}

/**
 * Program main
 */
 
int main(int argc, char **argv) {
    unsigned int grid, blk, nitocycles;
    long int frec;
    char *bench = (char *) malloc(4);
    bool time;
    unsigned long int long_nitocycles;

    if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
            checkCmdLineFlag(argc, (const char **)argv, "?")) {
        printf("Usage -bench=bench_name ('shm', 'glb', 'reg')\n");
        printf("      -grid=grid_size (Grid size)\n");
        printf("      -blk=block_size (Thread block size)\n");
        printf("      -nit=number_its (number of iterations)\n");
        printf("      -time=time (time to run the microbenchark)\n");

        exit(EXIT_SUCCESS);
    }

    frec=frec_now(); // Get current frequency to compute time from cycles
//    printf("GPU frequency: %lu \n", frec);
    if (checkCmdLineFlag(argc, (const char **)argv, "bench")) {
        getCmdLineArgumentString(argc, (const char **)argv, "bench", &bench);
    }
    else
      printf ("FAIL: bench\n");

    // Grid size
    if (checkCmdLineFlag(argc, (const char **)argv, "grid")) {
        grid = getCmdLineArgumentInt(argc, (const char **)argv, "grid");
    }

    // Thread block size 
    if (checkCmdLineFlag(argc, (const char **)argv, "blk")) {
        blk = getCmdLineArgumentInt(argc, (const char **)argv, "blk");
    }
    else
      printf ("FAIL: blk\n");

    time=false;
    // Kernel time
    if (checkCmdLineFlag(argc, (const char **)argv, "time")) {
        long_nitocycles = ((long int) (frec * getCmdLineArgumentFloat(argc, (const char **)argv, "time")));
        nitocycles=(unsigned int) (long_nitocycles >> BITSNOSIGNIFICATIVOS);
        time=true;
    }
    else // Number of iterations
        if (checkCmdLineFlag(argc, (const char **)argv, "nit")) {
            nitocycles = getCmdLineArgumentInt(argc, (const char **)argv, "nit");
        }
        else
            printf ("FAIL:nit and/or time\n");

    printf("microKernel=%s, grid: %u, blk: %u, nit o cycles: %u\n", bench, grid, blk, nitocycles);

    int kernel_result = launch_kernel(bench, grid, blk, nitocycles, time);

    printf("Launch result: %d\n", kernel_result);

#ifdef SIM_ERROR
    /*** To simulate different kinds of errors that can be catched from an external script used to launch the experiment              PASS(0), SDC(1), CRASH(2), HANG = sleep(xx) + exit(3)
    ***/

    unsigned int seed = (unsigned) clock() % 1000;
    srand( seed ); 
    float aux = ((float) rand() / (float)(RAND_MAX));
//    printf("RAND  seed: %u result: %f\n", seed, aux);

    if (aux < 0.01) // CRASH
      exit(4);
    else if (aux < 0.05) { // HANG
      sleep(30);
      exit(3);
    }
    else if (aux < 0.1) // SDC
      exit(1);
    else // PASS
      exit(0);
#else
    exit(!kernel_result);
#endif

}


/**************************************************************************
*  GPU_microbench -- GPU reliability microbenchmarks                      *
*                                                                         *
*  Copyright 2023-24 Jose M. Badia <badia@uji.es> and                     *
*                    German Leon <leon@uji.es>                            *
*                                                                         *
*  micro.c is part of GPU_microbench                                      *
*                                                                         *
*  GPU_microbench is free software: you can redistribute it and/or modify *
*  it under the terms of the GNU General Public License as published by   *
*  the Free Software Foundation; either version 3 of the License, or      *
*  (at your option) any later version.                                    *
*                                                                         *
*  GPU_microbench is distributed in the hope that it will be useful, but  *
*  WITHOUT ANY WARRANTY; without even the implied warranty of             *
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU      *
*  General Public License for more details.                               *
*                                                                         *
*  You should have received a copy of the GNU General Public License      *
*  along with this program.  If not, see <http://www.gnu.org/licenses/>   *
*                                                                         *
***************************************************************************/

// System includes
#include <unistd.h>
#include <stdio.h>
#include <stdbool.h>
#include <assert.h>
#include <stdlib.h>
#include <dirent.h>
#include <string.h>

#include <cuda_runtime.h>

// Helper functions and utilities to get the command line arguments
#include <helper_string.h>

#include "micro.h"

#define DEVFREQ_PATH "/sys/class/devfreq/"

#define NVIDIA_SMI_COMMAND "nvidia-smi --id=%d --query-gpu=clocks.current.sm --format=csv,noheader,nounits 2>/dev/null"

long int get_first_gpu_frequency_from_nvidia_smi() {
    // Get the value of CUDA_VISIBLE_DEVICES
    char* cudaVisibleDevices = getenv("CUDA_VISIBLE_DEVICES");

    int gpu_id;

    if (cudaVisibleDevices == NULL) {
        // If CUDA_VISIBLE_DEVICES is not defined, use the default value 0
        gpu_id = 0;
//        printf("CUDA_VISIBLE_DEVICES not defined. Using the default value: %d\n", gpu_id);
    } else {
        // If CUDA_VISIBLE_DEVICES is defined, convert it to an integer
        gpu_id = atoi(cudaVisibleDevices);
//        printf("CUDA_VISIBLE_DEVICES is set to: %s\n", cudaVisibleDevices);
    }

    // Create the NVIDIA_SMI_COMMAND using the determined GPU ID
    char command[256];
    snprintf(command, sizeof(command), NVIDIA_SMI_COMMAND, gpu_id);

    FILE *nvidia_smi_pipe = popen(command, "r");

    if (nvidia_smi_pipe == NULL) {
        perror("Error opening pipe to nvidia-smi");
        exit(EXIT_FAILURE);
    }

    int gpu_freq;
    int scan_result = fscanf(nvidia_smi_pipe, "%d", &gpu_freq);

    pclose(nvidia_smi_pipe);

    if (scan_result != 1) {
        // Failed to get GPU frequency from nvidia-smi
        return -1;
    }

    // Convert frequency to Hz
    long int gpu_freq_hz = gpu_freq * 1000000;

    // printf("Current GPU freq: %ld Hz\n", gpu_freq_hz);

    return gpu_freq_hz;
}

long int freq_now() {
    // Try to get GPU frequency from nvidia-smi
    long int gpu_freq = get_first_gpu_frequency_from_nvidia_smi();

    if (gpu_freq != -1) {
        // Successfully obtained GPU frequency from nvidia-smi
        return gpu_freq;
    }

    DIR *dir;
    struct dirent *entry;

    // Open the /sys/class/devfreq/ directory
    dir = opendir(DEVFREQ_PATH);
    if (dir == NULL) {
        perror("Error opening directory");
        exit(EXIT_FAILURE);
    }

    // Find the GPU frequency file dynamically
    while ((entry = readdir(dir)) != NULL) {
        if (strstr(entry->d_name, "gpu") != NULL) {
            char gpu_freq_file_path[512];  // Increased buffer size

            // Use sprintf with caution, ensure buffer size is sufficient
            sprintf(gpu_freq_file_path, "%s%s/max_freq", DEVFREQ_PATH, entry->d_name);
//            sprintf(gpu_freq_file_path, "%s%s/cur_freq", DEVFREQ_PATH, entry->d_name);

            // Read GPU frequency from the file
            FILE *file = fopen(gpu_freq_file_path, "r");
            if (file != NULL) {
                int gpu_freq;
                fscanf(file, "%d", &gpu_freq);
                fclose(file);

                // Close the directory
                closedir(dir);

                return gpu_freq;
            }
        }
    }

    // If we reach here, GPU frequency file was not found
    printf("Error: GPU frequency file not found.\n");

    // Close the directory
    closedir(dir);

    // Return an error value, you might want to handle this appropriately
    return -1;
}

/**
 * Program main
 */
 
int main(int argc, char **argv) {
    unsigned int grid, blk, nitocycles;
    long int freq;
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

    freq=freq_now(); // Get current frequency to compute time from cycles
//    printf("GPU frequency: %lu Hz\n", freq);
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

        int deviceId;
        cudaGetDevice(&deviceId);

        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, deviceId);

        int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;


        if (blk > maxThreadsPerBlock) {
            printf("Error: User is launching more threads in a block than supported by the GPU: %d!\n", maxThreadsPerBlock);
	    return 0;
        }


    }
    else
      printf ("FAIL: blk\n");

    time=false;
    // Kernel time
    if (checkCmdLineFlag(argc, (const char **)argv, "time")) {
        long_nitocycles = ((long int) (freq * getCmdLineArgumentFloat(argc, (const char **)argv, "time")));
        nitocycles=(unsigned int) (long_nitocycles >> BITSNOSIGNIFICATIVOS);
        time=true;
    }
    else // Number of iterations
        if (checkCmdLineFlag(argc, (const char **)argv, "nit")) {
            nitocycles = getCmdLineArgumentInt(argc, (const char **)argv, "nit");
        }
        else
            printf ("FAIL:nit and/or time\n");

//    printf("microKernel=%s, grid: %u, blk: %u, nit o cycles: %u\n", bench, grid, blk, nitocycles);

    int kernel_result = launch_kernel(bench, grid, blk, nitocycles, time);

//    printf("Launch result: %d\n", kernel_result);

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


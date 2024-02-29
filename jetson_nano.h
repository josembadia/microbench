/***************************************************************************
*  micro -- GPU reliability microbenchmarks                               *
*                                                                         *
*  Copyright 2023-24 Jose M. Badia <barrachi@uji.es> and                  *
*                    German Leon <leon@uji.es>                            *
*                                                                         *
*  jetson_nano.h is part of micro                                         *
*                                                                         *
*  micro is free software: you can redistribute it and/or modify          *
*  it under the terms of the GNU General Public License as published by   *
*  the Free Software Foundation; either version 3 of the License, or      *
*  (at your option) any later version.                                    *
*                                                                         *
*  micro is distributed in the hope that it will be useful, but           *
*  WITHOUT ANY WARRANTY; without even the implied warranty of             *
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU      *
*  General Public License for more details.                               *
*                                                                         *
*  You should have received a copy of the GNU General Public License      *
*  along with this program.  If not, see <http://www.gnu.org/licenses/>   *
*                                                                         *
***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <string.h>

#define DEVFREQ_PATH "/sys/class/devfreq/"
#define NVIDIA_SMI_COMMAND "nvidia-smi --query-gpu=clocks.current.sm --format=csv,noheader,nounits"

long int get_first_gpu_frequency_from_nvidia_smi() {
    FILE *nvidia_smi_pipe = popen(NVIDIA_SMI_COMMAND, "r");

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

//    printf("Current GPU freq: %ld Hz\n", gpu_freq_hz);

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
            sprintf(gpu_freq_file_path, "%s%s/cur_freq", DEVFREQ_PATH, entry->d_name);

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

/*
long int frec_now();

// Returns the current frequency of the GPU
long int frec_now() {
  char buffer[80];
  long int valor;
  FILE * fdf;
  sprintf(buffer,"/sys/devices/57000000.gpu/devfreq/57000000.gpu/cur_freq");
  fdf=fopen(buffer,"r");
  fscanf (fdf,"%lu",&valor);
  return (valor);
}	
*/

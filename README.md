# microbench
Microbenchmarks used to test the neutron radiation reliability of the Jetson Nano board

Results presented in the "RADiation and its Effects on Components and Systems Conference (RADECS2023)".

Code is implemented in C and CUDA.

## Usage

micro -bench=bench -grid=grid -blk=blk -time=time or  -nit=nit

where

bench = {"reg", "shm" or "glb"} is the type of memory used to store the results computed on each thread

grid = size of the CUDA grid

blk = thread block size

time = time in seconds of every test

nit = number of iterations of every test

To fix the time of every test we compute the number of clock cycles based on the GPU frequency.



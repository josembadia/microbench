#include <stdio.h>
#include <stdlib.h>

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

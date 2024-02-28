/*
###########################################################################
#  micro -- GPU reliability microbenchmarks                               #
#                                                                         #
#  Copyright 2023-24 Jose M. Badia <barrachi@uji.es> and                  #
#                    German Leon <leon@uji.es>                            #
#                                                                         #
#  jetson_nano.h is part of micro                                         #
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
*/

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

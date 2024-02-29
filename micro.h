/**************************************************************************
*  micro -- GPU reliability microbenchmarks                               *
*                                                                         *
*  Copyright 2023-24 Jose M. Badia <barrachi@uji.es> and                  *
*                    German Leon <leon@uji.es>                            *
*                                                                         *
*  micro.cu is part of micro                                              *
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

#ifndef _MICRO_H_
#define _MICRO_H_

#define FREQ 921600000
#define T 1
#define BITSNOSIGNIFICATIVOS 16
#define CYCLES (T*(FREQ) >> BITSNOSIGNIFICATIVOS)
#define QUATUMINTERACIONES 1000
#define SIZEROW 1
#define myclock() (int) (clock64() >> BITSNOSIGNIFICATIVOS)

//*define SIM_ERROR

typedef int btype;
typedef btype *btypePtr;

extern int launch_kernel(char *bench, int grid, int blk, unsigned int nitocycles, int time);


#endif // _MICRO_H_

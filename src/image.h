#ifndef IMAGE_H
#define IMAGE_H

#include <stdio.h>

#include "data.h"


/* ========================================================================== */
/*  DATA TYPES                                                                */
/* ========================================================================== */
/* container for image data */
struct Img_pan {
  int nw, nh; // dimensions
  double **img_d; // distances
  double **img_h; // heights
  double **img_u; // x step
  double **img_v; // y step
  int **img_n; // step counts
  double z; // relative height
};

/* container for image data */
struct Img_loc {
  int nw, nh; // dimensions
  double **X; // eastings
  double **Y; // nothings
  double **Z; // heights
  int i0, j0; // viewpoint location
};


/* ========================================================================== */
/*   FUNCTION DECLARATIONS                                                    */
/* ========================================================================== */
/* takes the panormam data and writes it to the file whose name is the string
   pointed to by path. Returns an error code:
     0 - success
     1 - failed to allocate png struct or info
     2 - failed to open file
     3 - subsequent write failure
     4 - color palette too large
  */
int panorama_write(const char * restrict path, struct Img_pan *img, struct Panorama *p);

/* takes the location data and writes it to the file whose name is the string
   pointed to by path. Returns an error code:
     0 - success
     1 - failed to allocate png struct or info
     2 - failed to open file
     3 - subsequent write failure
     4 - color palette too large
  */
int location_write(const char * restrict path, struct Img_loc *loc, struct Panorama *p);


#endif

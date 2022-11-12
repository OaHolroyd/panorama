#ifndef IMAGE_H
#define IMAGE_H

#include <stdio.h>

#include "data.h"


/* ========================================================================== */
/*  DATA TYPES                                                                */
/* ========================================================================== */
/* container for image data */
struct Image {
  int nw, nh; // dimensions
  double **img_d; // distances
  double **img_h; // heights
  double **img_u; // x step
  double **img_v; // y step
  int **img_n; // step counts
  double z; // relative height
};


/* ========================================================================== */
/*   FUNCTION DECLARATIONS                                                    */
/* ========================================================================== */
/* takes the image data and writes it to the specified stream */
int image_write(const char * restrict path, struct Image *img, struct Panorama *p);


#endif

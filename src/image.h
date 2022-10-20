#ifndef IMAGE_H
#define IMAGE_H

#include <stdio.h>


/* ========================================================================== */
/*  DATA TYPES                                                                */
/* ========================================================================== */
/* container for image data */
struct Image {
  int nw, nh; // dimensions
  int *wlim; // width limits
  int *hlim; // height limits
  double **img_d; // distances
  double **img_h; // heights
  int **img_n; // step counts
  double dmax; // maximum distance
};


/* ========================================================================== */
/*   FUNCTION DECLARATIONS                                                    */
/* ========================================================================== */
/* takes the image data and writes it to the specified stream */
int image_write(const char * restrict path, struct Image *img);


#endif

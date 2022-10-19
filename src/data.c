#include "data.h"

#include <stdlib.h>
#include <stdio.h>

#include "utils.h"


/* ========================================================================== */
/*   FUNCTION DEFINITIONS                                                     */
/* ========================================================================== */
/* sets ny and nx to the number of columns and rows in a block from the given
   data source */
void get_block_dims(data_source source, int *ny, int *nx) {
  switch (source) {
    case OST50:
      *ny = 200;
      *nx = 200;
      break;
    case SWISSALTI3D:
      *ny = 500;
      *nx = 500;
      break;
    default :
      ERROR("data source not recognised");
  }
}

/* allocates *Bx and *By (with length nbx, nby respectively) and fills them
   with the x and y range of the data source */
void get_data_extent(data_source source, int *nbx, int *nby, int **Bx, int **By) {
  switch (source) {
    case OST50:
      *nbx = 51;
      *Bx = malloc(*nbx*sizeof(int));
      for (int i = 0; i < *nbx; i++) {
        (*Bx)[i] = 0 + 10*i;
      } // i end

      *nby = 123;
      *By = malloc(*nby*sizeof(int));
      for (int i = 0; i < *nby; i++) {
        (*By)[i] = 0 + 10*i;
      } // i end
      break;
    case SWISSALTI3D:
      ERROR("SWISSALTI3D get_data_extent not implemented");
      *nbx = 51;
      *Bx = malloc(*nbx*sizeof(int));
      for (int i = 0; i < *nbx; i++) {
        (*Bx)[i] = 0 + 10*i;
      } // i end

      *nby = 123;
      *By = malloc(*nby*sizeof(int));
      for (int i = 0; i < *nby; i++) {
        (*By)[i] = 0 + 10*i;
      } // i end
      break;
    default :
      ERROR("data source not recognised");
  }
}

/* get the block coords containing the point (x0, y0) */
void block_index(data_source source, double y0, int *I0, double x0, int *J0) {
  switch (source) {
    case OST50:
      *I0 = 10*(((int)y0)/10000);
      *J0 = 10*(((int)x0)/10000);
      break;
    case SWISSALTI3D:
      *I0 = ((int)y0)/1000;
      *J0 = ((int)x0)/1000;
      break;
    default :
      ERROR("data source not recognised");
  }
}

/* fills X, Y, and Z with the eastings, northings, and heights from the data
   block (I0, J0) from the given data source. If the file containing the data
   is not found, it defaults to returning a height of zero for the entire block.
   Returns an integer error code:
      0 on success
     -1 failed to read correct number of entries
   The data must be saved in row-major format from the lower-left corner to the
   upper-right corner */
int get_block(int I0, int J0, data_source source, double **X, double **Y, double **Z) {
  char file[256]; // filepath

  /* get filename, lower left coords, and grid spacing */
  int ny, nx;
  double x0, y0, cw, ch;
  switch (source) {
    case OST50:
      sprintf(file, "./data/ost50/OST50-200-200-%04d-%04d.gzd", I0, J0);
      ny = 200;
      nx = 200;
      x0 = 1000*J0;
      y0 = 1000*I0;
      cw = 50;
      ch = 50;
      break;
    case SWISSALTI3D:
      ERROR("SWISSALTI3D get_block not implemented");
      sprintf(file, "./data/swissalti3d/SWISSALTI3D-500-500-%04d-%04d.gzd", I0, J0);
      ny = 500;
      nx = 500;
      x0 = 1000*J0;
      y0 = 1000*I0;
      cw = 2;
      ch = 2;
      break;
    default :
      ERROR("data source not recognised");
  }

  /* read data from file */
  FILE *fp = fopen(file, "rb");
  if (!fp) {
    /* fill X, Y, and Z values */
    for (int i = 0; i < ny; i++) {
      for (int j = 0; j < nx; j++) {
        Y[i][j] = y0 + (i+0.5)*ch;
        X[i][j] = x0 + (j+0.5)*cw;
        Z[i][j] = 0.0;
      } // j end
    } // i end
    return 0;
  }
  size_t err = fread(*Z, sizeof(double), ny*nx, fp);
  fclose(fp);

  /* return if error has occured */
  if (err != (size_t)ny*nx) {
    return -1;
  }

  /* fill X and Y values */
  for (int i = 0; i < ny; i++) {
    for (int j = 0; j < nx; j++) {
      Y[i][j] = y0 + (i+0.5)*ch;
      X[i][j] = x0 + (j+0.5)*cw;
    } // j end
  } // i end

  return 0;
}

/* given an ny-by-nx base grid Zb, allocates storage for the multigrid
   specified by levels and contstructs the grid ***Z. Note that the top level of
   the grid uses the memory allocated to Zb */
void create_multigrid(int ny, int nx, int nlevels, int *levels, double **Zb, double ****pZ) {
  *pZ = malloc(nlevels*sizeof(double**));

  /* just copy the pointer for the top level */
  (*pZ)[0] = Zb;

  /* allocate space for the remaining levels */
  for (int i = 1; i < nlevels; i++) {
    ny /= levels[i];
    nx /= levels[i];
    (*pZ)[i] = malloc_2d(ny, nx, sizeof(double));
  } // i end
}

/* frees all of the memory allocated by create_multigrid (no including the
   zeroth layer). */
void free_multigrid(int nlevels, double ***Z) {
  /* free internal data */
  for (int i = 1; i < nlevels; i++) {
    free_2d(Z[i]);
  } // i end

  free(Z);
}

/* fills the multigrid with coarsened values going up the levels (see mipmap).
   Note that this assumes that Z[0] = Zb */
void fill_multigrid(int ny, int nx, int nlevels, int *levels, double ***Z) {
  int l, im0, jm0;
  for (int k = 1; k < nlevels; k++) {
    l = levels[k];
    ny /= l;
    nx /= l;
    for (int i = 0; i < ny; i++) {
      im0 = l*i;
      for (int j = 0; j < nx; j++) {
        jm0 = l*j;

        /* find max in l-by-l grid in layer k-1 */
        Z[k][i][j] = Z[k-1][im0][jm0];
        for (int im = im0; im < im0+l; im++) {
          for (int jm = jm0; jm < jm0+l; jm++) {
            if (Z[k][i][j] < Z[k-1][im][jm]) { Z[k][i][j] = Z[k-1][im][jm]; }
          } // jm end
        } // im end
      } // j end
    } // i end
  } // k end
}

/* checks if the level structure is compatable with the grid structure: at each
   refinement the coarsening factor (in levels) must divide the grid dimensions.
   Returns 0 if valid, 1 otherwise */
int validate_levels(int ny, int nx, int nlevels, int *levels) {
  /* there must be at least one level */
  if (nlevels < 1) {
    return 1;
  }

  /* the first level must be the raw data */
  if (levels[0] != 1) {
    return 1;
  }

  /* check that the coarsening factors divide the grid */
  int l = 1;
  for (int i = 1; i < nlevels; i++) {
    l = levels[i];
    if (l*(ny/l)-ny != 0 || l*(nx/l)-nx != 0) {
      return 1;
    }
    ny /= l;
    nx /= l;
  } // i end

  return 0;
}

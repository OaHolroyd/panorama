/*    PANORMA - multi-block, multi-level, single-source                       */
/*                                                                            */
/* Generates a panorama image from multiple blocks of DEM data. Uses a        */
/* pyramid to reduce ray collision checks. Currently only supports            */
/* single-source data in UTM formats from the following sources:              */
/*   OS Terrain 50  (with significant pre-processing to remove metadata and   */
/*                   alter the format from ascii to binary)                   */
/*       SwissTopo  (as-is xyz data)                                          */
/*                                                                            */
/*  TODO                                                                      */
/* - start at the maximum level possible                                      */
/* - correct for missing data blocks (eg holes over the sea for OST50)        */
/* - allow use of multiple UTM sources                                        */
/* - allow use of lat/lon data (via interpolation)                            */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <stdint.h>

#include "utils.h"
#include "data.h"


/* flags to toggle features */
#define LOG_FLAG 1 // display logs

#define DROP 6.544431587294458e-08 // height drop (metre/metre^2)


/* container for panorama data */
struct Panorama {
  /* viewpoint */
  double x0, y0; // easting and northing
  double z0; // relative height above surface
  double d0; // minimum ray collision distance

  /* panorama */
  int wlim[2], hlim[2]; // panorama image limits (in degrees)
  int res; // resolution (pixels per degree)

  /* data */
  data_source source; // data source (with associated projection/datum)
  int blockmax; // number of blocks to cover
  double dmax; // maximum ray distance
  int nlevels; // number of levels
  int *levels; // level structure
};

/* block exit side */
enum Exit_side {
  EXIT_LEFT,
  EXIT_RIGHT,
  EXIT_TOP,
  EXIT_BOTTOM,
};


/* ========================================================================== */
/*   AUXILIARY FUNCTION DEFINITIONS                                           */
/* ========================================================================== */
/* sets the default parameters */
void setup_default(struct Panorama *p) {
  /* Sgurr Fhuaran summit (NG 978 167) */
  p->x0 = 197800.0;
  p->y0 = 816700.0;
  p->z0 = 50.0;
  p->d0 = 100.0;

  p->wlim[0] = 0; p->wlim[1] = 360;
  p->hlim[0] = -6; p->hlim[1] = 2;
  p->res = 16;

  p->source = OST50;
  p->blockmax = 9;
  p->dmax = 82000;
  p->nlevels = 2;
  p->levels = malloc(p->nlevels*sizeof(int));
  p->levels[0] = 1;
  p->levels[1] = 10;
  // p->levels[2] = 2;
}

/* reads inputs from the command line */
int read_intputs(struct Panorama *p, int argc, char const *argv[]) {
  setup_default(p);
  // TODO: read inputs (with getopt)
  debug("TODO: read inputs with getopt");
  return 0;
}


/* ========================================================================== */
/*   MAIN                                                                     */
/* ========================================================================== */
int main(int argc, char const *argv[]) {
  /* ====================================================================== */
  /*   Set Up                                                               */
  /* ====================================================================== */
  /* get inputs from command line */
  struct Panorama p; // wrapper for input data
  read_intputs(&p, argc, argv);

  /* extract from struct */
  double x0 = p.x0;
  double y0 = p.y0;
  double z0 = p.z0;
  double d0 = p.d0;
  int wlim[2] = {p.wlim[0], p.wlim[1]};
  int hlim[2] = {p.hlim[0], p.hlim[1]};
  int res = p.res;
  data_source source = p.source;
  int blockmax = p.blockmax;
  double dmax = p.dmax;
  int nlevels = p.nlevels;
  int *levels = p.levels;


  /* ====================================================================== */
  /*   Image Initialisation                                                 */
  /* ====================================================================== */
  /* dimensions */
  double dpx = 1.0/((double)res);
  int nw = (wlim[1]-wlim[0])*res;
  int nh = (hlim[1]-hlim[0])*res + 1;

  // note that the image is stored rotated so that we can iterate over the
  // columns for speed
  double **img_d = malloc_2d(nw, nh, sizeof(double));
  double **img_h = malloc_2d(nw, nh, sizeof(double));
  int **img_n = malloc_2d(nw, nh, sizeof(int));
  double **img_x = malloc_2d(nw, nh, sizeof(double));
  double **img_y = malloc_2d(nw, nh, sizeof(double));
  for (int j = 0; j < nw; j++) {
    for (int i = 0; i < nh; i++) {
      img_d[j][i] = -DBL_MAX;
      img_h[j][i] = -DBL_MAX;
      img_n[j][i] = 0;
      img_x[j][i] = wlim[0] + j*dpx;
      img_y[j][i] = hlim[0] + i*dpx;
    } // i end
  } // j end


  /* ====================================================================== */
  /*   Block Initialisation                                                 */
  /* ====================================================================== */
  /* dimensions */
  int ny, nx;
  get_block_dims(source, &ny, &nx);

  double **X = malloc_2d(ny, nx, sizeof(double)); // eastings grid
  double **Y = malloc_2d(ny, nx, sizeof(double)); // northings grid
  double **Zb = malloc_2d(ny, nx, sizeof(double)); // height grid

  /* read origin block */
  int I0, J0;
  block_index(source, y0, &I0, x0, &J0);
  int err = get_block(I0, J0, source, X, Y, Zb);
  if (err != 0) {
    ERROR("failed to read block [%d, %d] (returned %d)", I0, J0, err);
  }

  /* grid spacing */
  double cw = X[0][1] - X[0][0];
  double ch = Y[1][0] - Y[0][0];

  /* check levels are valid */
  err = validate_levels(ny, nx, nlevels, levels);
  if (err != 0) {
    ERROR("level structure invalid");
  }

  /* set up multigrid */
  double ***Z;
  create_multigrid(ny, nx, nlevels, levels, Zb, &Z);

  /* compute block distances */
  int nbx, nby;
  int *Bx, *By;
  get_data_extent(source, &nbx, &nby, &Bx, &By);
  int nblocks = nbx*nby;
  double *blocks_d = malloc(nblocks*sizeof(double));
  for (int k = 0; k < nblocks; k++) {
    int i = k/nbx;
    int j = k-i*nbx;
    blocks_d[k] = (I0-By[i])*(I0-By[i]) + (J0-Bx[j])*(J0-Bx[j]);
  } // k end

  // TODO: use better sorting algorithm
  int *blocks = malloc(nblocks*sizeof(int)); // block ordering
  int *iblocks = malloc(nblocks*sizeof(int));
  for (int k = 0; k < nblocks; k++) {
    // find the kth minimum
    int imin = 0;
    double dmin = blocks_d[imin];
    for (int i = 0; i < nblocks; i++) {
      if (dmin > blocks_d[i]) {
        imin = i;
        dmin = blocks_d[i];
      }
    } // i end

    // "remove" from list
    blocks_d[imin] = DBL_MAX;

    // place in blocks and iblocks
    blocks[k] = imin;
    iblocks[imin] = k;
  } // k end


  /* ====================================================================== */
  /*   Ray Initialisation                                                   */
  /* ====================================================================== */
  int nrays = nw*nh;
  int **rays = malloc_2d(blockmax, nrays+1, sizeof(int));

  /* the number of rays in the kth block is stored in rays[k][nrays] */
  for (int k = 0; k < blockmax; k++) {
    rays[k][nrays] = 0;
  } // k end

  /* all of the rays begin in the 0th block */
  rays[0][nrays] = nrays;
  for (int i = 0; i < nrays; i++) {
    rays[0][i] = i;
  } // i end

  /* ray block entry side (0, 1, 2, 3 - top, bottom, right, left) */
  enum Exit_side *rays_entry = malloc(nrays*sizeof(enum Exit_side));


  /* ====================================================================== */
  /*   Ray Tracing                                                          */
  /* ====================================================================== */
  /* absolute viewpoint height */
  int i0 = (y0 - Y[0][0] + 0.5*ch)/ch;
  int j0 = (x0 - X[0][0] + 0.5*cw)/cw;
  z0 += Zb[i0][j0];

  /* loop over the data blocks in order of distance from the origin block */
  for (int block = 0; block < blockmax; block++) {
    /* ======================== */
    /*  Fill elevation pyramid  */
    /* ======================== */
    int IJb = blocks[block]; // single index
    int Ib = IJb/nbx; // row index
    int Jb = IJb-Ib*nbx; // column index
    err = get_block(By[Ib], Bx[Jb], source, X, Y, Zb);
    if (err != 0) {
      /* skip data blocks that don't exist */
      // TODO: try returning a zero block instead or computing the intersections exactly
      fprintf(stderr, "  WARNING: skipping block [%d, %d]\n", By[Ib], Bx[Jb]);
      continue;
    }
    fill_multigrid(ny, nx, nlevels, levels, Z);

    /* log progress */
    if (LOG_FLAG) {
      fprintf(stderr, "block %d [%d, %d] of %d (%d rays)\n", block+1, Ib, Jb, blockmax, rays[block][nrays]);
    }

    /* ================================== */
    /*  Propagate rays through the block  */
    /* ================================== */
    int Jprev = -1; // previous column
    for (int k = 0; k < rays[block][nrays]; k++) {
      /* ============== */
      /* create the ray */
      /* ============== */
      /* index */
      int IJ = rays[block][k];
      int J = IJ/nh; // column index (recalling that img is column-major)

      /* variables */
      double theta = M_PI*(0.5 - img_x[0][IJ]/180.0); // heading
      double u = cos(theta); // ray direction vector
      double v = sin(theta); // TODO: maybe try using u = sqrt(1 - v*v)
      int stepy = (v>0.0) - (v<0.0); // step directions
      int stepx = (u>0.0) - (u<0.0);
      double dty = fabs(ch/v); // dt to cross one cell
      double dtx = fabs(cw/u);
      double dz = tan(img_y[0][IJ]*M_PI/180.0);

      /* =============================== */
      /* start position, index, and edge */
      /* =============================== */
      int i00, j00; // start index
      int i, j; // current index
      double y00, x00, z00; // start position
      int block1; // next block
      int dprev = -1; // previous step (0 for vertical, 1 for horizontal)
      double ty, tx; // vertical and horizontal distance

      /* if we're in the same column as the previous ray we can just continue
         from where it finished (provided we go up the columns of the image) */
      if (J == Jprev) {
        /* check if the previous ray left the block */
        if (i == -1) {
          /* left block, entered new one */
          rays_entry[IJ] = rays_entry[IJ-1];
          rays[block1][rays[block1][nrays]] = IJ;
          rays[block1][nrays] += 1;
          continue;
        } else if (i == -2) {
          /* left available data */
          continue;
        }
      } else {
        /* compute ray start position from scratch */
        double t1;
        if (block == 0) {
          /* just start from the viewpoint */
          y00 = y0; x00 = x0;
          t1 = 0.0;
        } else {
          /* compute ray entry point be edge intersection */
          double y1, x1, dy1, dx1;
          switch (rays_entry[IJ]) {
            case EXIT_BOTTOM: // enter top
              x1 = X[0][0] - 0.5*cw;
              y1 = Y[0][0] + ch*(ny-0.5);
              dx1 = 1.0;
              dy1 = 0.0;
              dprev = 0;
              break;
            case EXIT_TOP: // enter bottom
              x1 = X[0][0] - 0.5*cw;
              y1 = Y[0][0] - 0.5*ch;
              dx1 = 1.0;
              dy1 = 0.0;
              dprev = 0;
              break;
            case EXIT_LEFT: // enter right
              x1 = X[0][0] + cw*(nx-0.5);
              y1 = Y[0][0] - 0.5*ch;
              dx1 = 0.0;
              dy1 = 1.0;
              dprev = 1;
              break;
            case EXIT_RIGHT: // enter left
              x1 = X[0][0] - 0.5*cw;
              y1 = Y[0][0] - 0.5*ch;
              dx1 = 0.0;
              dy1 = 1.0;
              dprev = 1;
              break;
          }
          if (v != 0.0) {
            t1 = (x0-x1 + (u/v)*(y1-y0)) / (dx1 - (u/v)*dy1);
          } else {
            t1 = (y0-y1) / dy1;
          }
          y00 = y1 + t1*dy1; x00 = x1 + t1*dx1;
        }

        /* start index */
        i00 = (y00 - Y[0][0] + 0.5*ch)/ch;
        j00 = (x00 - X[0][0] + 0.5*cw)/cw;
        i = i00; j = j00;

        /* handle corner cases */
        // TODO: handle corner cases better
        if (i < 0) {
          i = 0;
        } else if (i >= ny) {
          i = ny-1;
        }
        if (j < 0) {
          j = 0;
        } else if (j >= nx) {
          j = nx-1;
        }

        /* advance to first edge */
        ty = ((double)stepy*(Y[i][j]-y00)/ch + 0.5)*dty;
        tx = ((double)stepx*(X[i][j]-x00)/cw + 0.5)*dtx;
      }

      /* start height */
      double dy00 = y00-y0;
      double dx00 = x00-x0;
      double d00 = sqrt(dy00*dy00+dx00*dx00); // distance to start
      z00 = z0 + d00*dz + d00*d00*DROP;

      /* ================== */
      /* set starting level */
      /* ================== */
      int level = 0;
      int l = levels[0];
      int refined = 0;
      int modstepy = -1*(stepy==-1); // modulo at which to increase the level
      int modstepx = -1*(stepx==-1);

      /* ============= */
      /* propagate ray */
      /* ============= */
      while (1) {
        /* start with cell (i, j) just crossed, having travelled t metres along
           the ray from (y00, x00), now at height z */
        img_n[0][IJ] += 1;
        double t = (ty < tx) ? ty : tx;
        double z = z00 + t*(dz + t*DROP);

        /* check if that will collide with the current cell */
        if (t > d0 && Z[level][i/l][j/l] > z) {
          if (level == 0) { // collision
            /* set image and record final position for next ray */
            Jprev = J;
            img_d[0][IJ] = t + d00;
            img_h[0][IJ] = Z[0][i][j];
            break;
          } else { // potential collision at lower level
            /* go down a level */
            int l1 = l;
            l /= levels[level];
            level -= 1;
            refined = 1;

            /* find the correct index */
            if (dprev == 0) {
              /* rewind ty */
              ty -= l1*dty;

              /* find correct i index */
              if (stepy == 1) {
                i = l1*(i/l1);
              } else {
                i = l1*(i/l1) + (l1-1);
              }

              /* find correct j index */
              j = ((x00+ty*u) - X[0][0] + 0.5*cw)/cw;

              /* handle corner cases */
              // TODO: handle corner cases better
              if (i < 0) {
                i = 0;
              } else if (i >= ny) {
                i = ny-1;
              }
              if (j < 0) {
                j = 0;
              } else if (j >= nx) {
                j = nx-1;
              }

              /* rewind tx */
              tx -= rint((tx-(X[i][j]-x00+stepx*0.5*l*cw)/u)/(l*dtx))*l*dtx;

              /* move back forwards again (at the lower level) */
              ty += l*dty;
            } else if (dprev == 1) {
              /* rewind tx */
              tx -= l1*dtx;

              /* find correct j index */
              if (stepx == 1) {
                j = l1*(j/l1);
              } else {
                j = l1*(j/l1) + (l1-1);
              }

              /* find correct i index */
              i = ((y00+tx*v) - Y[0][0] + 0.5*ch)/ch;

              /* handle corner cases */
              // TODO: handle corner cases better
              if (i < 0) {
                i = 0;
              } else if (i >= ny) {
                i = ny-1;
              }
              if (j < 0) {
                j = 0;
              } else if (j >= nx) {
                j = nx-1;
              }

              /* rewind ty */
              ty -= rint((ty-(Y[i][j]-y00+stepy*0.5*l*ch)/v)/(l*dty))*l*dty;

              /* move back forwards again (at the lower level) */
              tx += l*dtx;
            } else {
              ERROR("(unreachable) go down level at start");
            }
            continue;
          }
        } // end collision check

        /* move forwards one cell */
        if (ty < tx) {
          dprev = 0;
          ty += l*dty;
          i += l*stepy;
        } else {
          dprev = 1;
          tx += l*dtx;
          j += l*stepx;
        }

        /* check if we've left the block */
        if (i < 0 || i >= ny || j < 0 || j >= nx) {
          /* track next entry side and find next block */
          int Ib1 = Ib; int Jb1 = Jb;
          if (i < 0) {
            Ib1 -= 1;
            rays_entry[IJ] = EXIT_BOTTOM;
          } else if (i >= ny) {
            Ib1 += 1;
            rays_entry[IJ] = EXIT_TOP;
          } else if (j < 0) {
            Jb1 -= 1;
            rays_entry[IJ] = EXIT_LEFT;
          } else {
            Jb1 += 1;
            rays_entry[IJ] = EXIT_RIGHT;
          }

          /* record information for next ray */
          Jprev = J;
          i = -1; j = -1; // -1: exited block (to prevent unnecessary restart)

          /* check if we have also left available data */
          if (Ib1 < 0 || Ib1 >= nby || Jb1 < 0 || Jb1 >= nbx) {
            i = -2; j = -2; // -2: exited data (to prevent unnecessary restart)
            break;
          }

          /* place ray in new block */
          block1 = Ib1*nbx + Jb1; // absolute index of next block
          block1 = iblocks[block1]; // order index of next block
          if (block1 >= blockmax) {
            /* don't move ray */
            i = -2; j = -2; // -2: exited data (to prevent unnecessary restart)
            break;
          }
          rays[block1][rays[block1][nrays]] = IJ;
          rays[block1][nrays] += 1;
          break;
        }

        /* go up a level if possible */
        if (level < nlevels-1) {
          int l1 = l * levels[level+1]; // next level
          int modstepy1 = (modstepy < 0) ? modstepy+l1 : modstepy;
          int modstepx1 = (modstepx < 0) ? modstepx+l1 : modstepx;
          /* step up a level if we are about to cross a one-level-up boundary,
             unless we've recently refined */
          if (i%l1 == modstepy1 && dprev == 0) {
            if (refined == 0) {
              /* adjust ty to align with larger grid */
              ty += (l1-l)*dty;

              /* adjust tx to align with larger grid */
              if (stepx == 1) {
                tx += ((l1-l) - l*((j % l1)/l)) * dtx;
              } else if (stepx == -1) { // TODO: could this just be an 'else' ?
                tx += l*((j % l1)/l) * dtx;
              }

              /* go up a level */
              level += 1;
              l = l1;
            } else {
              /* reset refined to let us go up a level next time */
              refined = 0;
            }
          } else if (j%l1 == modstepx1 && dprev == 1) {
            if (refined == 0) {
              /* adjust tx to align with larger grid */
              tx += (l1-l)*dtx;

              /* adjust ty to align with larger grid */
              if (stepy == 1) {
                ty += ((l1-l) - l*((i % l1)/l)) * dty;
              } else if (stepy == -1) { // TODO: could this just be an 'else' ?
                ty += l*((i % l1)/l) * dty;
              }

              /* go up a level */
              level += 1;
              l = l1;
            } else {
              /* reset refined to let us go up a level next time */
              refined = 0;
            }
          }
        }
      } // while end (loop across cells)
    } // k end (loop over all rays in block)
  } // block end (loop over all data blocks)

  // output the data to a file
  FILE *fp = fopen("out/img_d.dat", "w");
  for (int j = 0; j < nw; j++) {
    for (int i = 0; i < nh; i++) {
      fprintf(fp, "%lf ", img_d[j][i]);
    } // i end
    fprintf(fp, "\n");
  } // j end
  fclose(fp);

  fp = fopen("out/img_h.dat", "w");
  for (int j = 0; j < nw; j++) {
    for (int i = 0; i < nh; i++) {
      fprintf(fp, "%lf ", img_h[j][i]);
    } // i end
    fprintf(fp, "\n");
  } // j end
  fclose(fp);

  fp = fopen("out/img_n.dat", "w");
  for (int j = 0; j < nw; j++) {
    for (int i = 0; i < nh; i++) {
      fprintf(fp, "%d ", img_n[j][i]);
    } // i end
    fprintf(fp, "\n");
  } // j end
  fclose(fp);

  // TODO: generate image

  // TODO: output profiling

  /* ====================================================================== */
  /*   Clean Up                                                             */
  /* ====================================================================== */
  free(levels);
  free_2d(img_d);
  free_2d(img_h);
  free_2d(img_n);
  free_2d(img_x);
  free_2d(img_y);
  free_2d(X);
  free_2d(Y);
  free_2d(Zb);
  free_multigrid(nlevels, Z);
  free(Bx);
  free(By);
  free(blocks_d);
  free(blocks);
  free(iblocks);
  free_2d(rays);
  free(rays_entry);

  return EXIT_SUCCESS;
}

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
/* - exit gracefully                                                          */
/* - figure out optimum level structure                                       */
/* - make nicer images                                                        */
/* - allow use of multiple UTM sources                                        */
/* - allow use of lat/lon data (via interpolation)                            */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <unistd.h>
#include <string.h>

#include "utils.h"
#include "data.h"
#include "image.h"


/* flags to toggle features */
#define LOG_FLAG 1 // display logs

#define DROP 6.544431587294458e-08 // height drop (metre/metre^2)


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
  /* Ben Nevis summit (NN16677128) */
  p->x0 = 216670.0;
  p->y0 = 771280.0;
  p->z0 = 50.0;
  p->d0 = 200.0;

  p->wlim[0] = -22.5; p->wlim[1] = 360.0-22.5;
  p->hlim[0] = -5; p->hlim[1] = 1;
  p->res = 96;

  p->split = 1;
  p->labels = 0;

  p->source = OST50;
  p->blockmax = 500;
  p->dmax = 90000;
  p->nlevels = 3;
  p->levels = malloc(p->nlevels*sizeof(int));
  p->levels[0] = 1;
  p->levels[1] = 5;
  p->levels[2] = 2;
}

/* generate an example input file at filename containing the given data p.
   returns a non-zero exit code on failure. */
int generate_input_file(struct Panorama *p, const char *filename) {
  FILE *fp = fopen(filename, "w");
  if (!fp) { return 1; }

  char gridref[32];
  switch (p->source) {
    case OST50:
      ne_to_osng(p->y0, p->x0, gridref, 10, 0);
      fprintf(fp, "-S %s\n", "ost50");
      fprintf(fp, "-v %s\n", gridref);
      break;
    case SWT02:
      ne_to_swgr(p->y0, p->x0, gridref, 0);
      fprintf(fp, "-S %s\n", "swt02");
      fprintf(fp, "-v %s\n", gridref);
      break;
    case NZT08:
      ne_to_swgr(p->y0, p->x0, gridref, 0);
      fprintf(fp, "-S %s\n", "nzt08");
      fprintf(fp, "-v %s\n", gridref);
      break;
    default :
      ERROR("bad data source");
  }
  fprintf(fp, "-z %lf\n", p->z0);
  fprintf(fp, "-r %d\n", p->res);
  fprintf(fp, "-c %lf\n", p->d0);
  fprintf(fp, "-d %lf\n", p->dmax);
  fprintf(fp, "-b %d\n", p->blockmax);
  fprintf(fp, "-u %lf\n", p->hlim[1]);
  fprintf(fp, "-l %lf\n", -p->hlim[0]);
  fprintf(fp, "-o %lf\n", p->wlim[0]);
  fprintf(fp, "-w %lf\n", p->wlim[1]-p->wlim[0]);
  fprintf(fp, "-t\n");
  fprintf(fp, "-s\n");
  fprintf(fp, "-L\n");

  fclose(fp);
  return 0;
}

/* reads inputs from the command line into the struct p. Alows reading inputs
   from a file only if readfile is non-zero. Returns an exit code
     0 - success
     1 - option read failure
     2 - success but exit anyway */
int read_inputs(struct Panorama *p, int argc, char * const *argv, int readfile, int *loc) {
  /* inputs (set to rubbish) */
  struct Panorama p0;
  p0.x0 = DBL_MAX;
  p0.y0 = DBL_MAX;
  p0.z0 = DBL_MAX;
  p0.d0 = DBL_MAX;
  p0.hlim[0] = DBL_MAX;
  p0.hlim[1] = DBL_MAX;
  p0.res = -1;
  p0.split = -1;
  p0.labels = -1;
  p0.source = DATA_NULL;
  p0.blockmax = -1;
  p0.dmax = DBL_MAX;

  /* set to defaults */
  setup_default(p);

  int c, err;
  double w0 = 337.5, w = 360.0; // TODO: should be read from defaults
  char viewpoint[32];
  int viewset = 0;
  char filename[256];
  int fromfile = 0;
  int tofile = 0;
  const char *argstring = "thsr:v:d:c:b:z:S:l:u:o:w:f:FL";
  const int max_args = (int)strlen(argstring);
  while ((c = getopt(argc, argv, argstring)) != -1) {
    switch (c) {
      case 't':
        /* add text to the image */
        p0.labels = 1;
        break;
      case 's':
        /* strip (ie not split) image */
        p0.split = 0;
        break;
      case 'r':
        /* resolution */
        err = sscanf(optarg, "%d", &p0.res);
        if (err != 1) {
          fprintf(stderr, "failed to read resolution argument '%s'\n", optarg);
          return 1;
        }
        break;
      case 'S':
        /* data source */
        if (!strncmp(optarg, "ost50", 5)) {
          p0.source = OST50;
        } else if (!strncmp(optarg, "swt02", 5)) {
          p0.source = SWT02;
        } else if (!strncmp(optarg, "nzt08", 5)) {
          p0.source = NZT08;
        } else {
          fprintf(stderr, "failed to read source argument '%s'\n", optarg);
          return 1;
        }
        break;
      case 'v':
        /* viewpoint */
        strcpy(viewpoint, optarg);
        viewset = 1;
        break;
      case 'd':
        /* dmax */
        err = sscanf(optarg, "%lf", &p0.dmax);
        if (err != 1) {
          if (!strncmp(optarg, "auto", 4)) {
            p->dmax = -1.0;
          } else {
            fprintf(stderr, "failed to read dmax argument '%s'\n", optarg);
            return 1;
          }
        }
        p0.dmax *= 1000.0;
        break;
      case 'c':
        /* minimum cutoff */
        err = sscanf(optarg, "%lf", &p0.d0);
        if (err != 1) {
          fprintf(stderr, "failed to read cutoff argument '%s'\n", optarg);
          return 1;
        }
        break;
      case 'b':
        /* blockmax */
        err = sscanf(optarg, "%d", &p0.blockmax);
        if (err != 1) {
          fprintf(stderr, "failed to read blockmax argument '%s'\n", optarg);
          return 1;
        }
        break;
      case 'u':
        /* upper edge */
        err = sscanf(optarg, "%lf", &p0.hlim[1]);
        if (err != 1) {
          fprintf(stderr, "failed to read upper argument '%s'\n", optarg);
          return 1;
        }
        break;
      case 'l':
        /* lower edge */
        err = sscanf(optarg, "%lf", &p0.hlim[0]);
        p0.hlim[0] *= -1;
        if (err != 1) {
          fprintf(stderr, "failed to read lower argument '%s'\n", optarg);
          return 1;
        }
        break;
      case 'o':
        /* origin */
        err = sscanf(optarg, "%lf", &w0);
        if (err != 1) {
          fprintf(stderr, "failed to read origin argument '%s'\n", optarg);
          return 1;
        }
        break;
      case 'w':
        /* width */
        err = sscanf(optarg, "%lf", &w);
        if (err != 1) {
          fprintf(stderr, "failed to read width argument '%s'\n", optarg);
          return 1;
        }
        break;
      case 'z':
        /* viewpoint height */
        err = sscanf(optarg, "%lf", &p0.z0);
        if (err != 1) {
          fprintf(stderr, "failed to read height argument '%s'\n", optarg);
          return 1;
        }
        break;
      case 'f':
        /* input file */
        fromfile = 1;
        strcpy(filename, optarg);
        break;
      case 'F':
        /* output example input file */
        tofile = 1;
        break;
      case 'L':
        /* output location detail */
        *loc = 1;
        break;
      case 'h':
      default :
        fprintf(stderr,
          "Usage: panorama [options]\n"
          "Options:\n"
          "  -h             Display this information.\n"
          "  -S <source>    Specify the source of the elevation data. Available options\n"
          "                 are: ost50, swt02. (Defaults to ost50)\n"
          "  -v <gridref>   Specify a grid reference for the viewpoint location. Must\n"
          "                 match the specified data source and have spaces removed.\n"
          "                 (Defaults to NN16677128/E2623451N1100503 for ost50/swt02\n"
          "                 respectively)\n"
          "  -z <height>    Place the viewpoint at a height of <height> metres above the\n"
          "                 ground. (Defaults to 50)\n"
          "  -r <res>       Output an image with a resolution of <res> pixels per degree.\n"
          "                 (Defaults to 96)\n"
          "  -c <cutoff>    Neglect terrain  under <cutoff> metres from the viewpoint.\n"
          "                 (Defaults to 200)\n"
          "  -d <dmax>      Scales the colors between 0 and <dmax> kms from the viewpoint.\n"
          "                 (Defaults to 90)\n"
          "  -b <blockmax>  Limit the raytracing to <blockmax> data blocks. (Defaults to\n"
          "                 500)\n"
          "  -u <upper>     Set the upper edge of the image to <upper> degrees above the\n"
          "                 horizon. Must be greater than 0. (Defaults to 1)\n"
          "  -l <lower>     Set the upper edge of the image to <upper> degrees below the\n"
          "                 horizon. Must be greater than 0. (Defaults to 5)\n"
          "  -o <origin>    Set the leftmost edge to <origin> degrees. Must be between 0\n"
          "                 and 359. (Defaults to 337.5)\n"
          "  -w <width>     Set the width of the panorama to <width> degrees. Must be\n"
          "                 between 1 and 360. (Defaults to 360)\n"
          "  -t             Include labels on selected summits.\n"
          "  -s             Output the image as a single strip rather than divided into 8\n"
          "                 sections.\n"
          "  -L             Generate an additional image detailing the location of the\n"
          "                 viewpoint. Useful for debugging.\n"
          "  -f <file>      Pass inputs to panorama from the file <file>. Each input must\n"
          "                 be on a separate line and be of the format\n"
          "                     -<char> [argument]\n"
          "                 matching the available options listed here. Any options\n"
          "                 specified separately override those in the file.\n"
          "  -F             Generate an input file with the inputs given then exit without\n"
          "                 generating an image.\n");
        return 1;
    }
  }

  p->wlim[0] = w0-360.0;
  p->wlim[1] = p->wlim[0]+w;

  /* set variables from file by calling recursively */
  if (fromfile && readfile) {
    int num_args = 1;
    char *args[max_args];
    args[0] = argv[0];

    /* read file contents */
    FILE *fp = fopen(filename, "r");
    if (!fp) { return 1; }
    fseek(fp, 0L, SEEK_END);
    size_t size = ftell(fp);
    fseek(fp, 0L, SEEK_SET);
    char* contents = malloc(size);
    if (!contents) { fclose(fp); return 1; }
    fread(contents, 1, size, fp);

    /* put into args */
    size_t i = 0;
    while (i < size) {
      /* spaces/newlines become string ends */
      if (contents[i] == ' ' || contents[i] == '\n') {
        contents[i] = '\0';
      } else if (i == 0) { // catch the first term
        args[num_args] = &(contents[i]);
        num_args++;
      } else if (contents[i-1] == '\0') {
        args[num_args] = &(contents[i]);
        num_args++;
      }

      i++;
    }

    /* read args */
    optind = 1;
    read_inputs(p, num_args, args, 0, loc);

    fclose(fp);
  }

  /* update variables that have been set on the command line */
  if (p0.source != DATA_NULL) {
    p->source = p0.source;
    /* also update default location */
    switch (p->source) {
      case OST50:
        // Ben Nevis
        p->x0 = 216670.0;
        p->y0 = 771280.0;
        break;
      case SWT02:
        // Mettlehorn
        p->x0 = 2623451.0;
        p->y0 = 1100503.0;
        break;
      case NZT08:
        // Mount Cook
        p->x0 = 1369350.0;
        p->y0 = 5169150.0; // TODO this is 18000 off
        break;
      default :
        ERROR("bad data source");
    }
  }
  if (viewset) {
    switch (p->source) {
      case OST50:
        err = osng_to_ne(viewpoint, &p->y0, &p->x0);
        break;
      case SWT02:
        err = swgr_to_ne(viewpoint, &p->y0, &p->x0);
        break;
      case NZT08:
        err = swgr_to_ne(viewpoint, &p->y0, &p->x0);
        break;
      default :
        ERROR("bad data source");
        err = 1;
    }
    if (err) {
      fprintf(stderr, "failed to read viewpoint argument '%s'\n", optarg);
      return 1;
    }
  }
  if (p0.z0 < DBL_MAX) {
    p->z0 = p0.z0;
  }
  if (p0.d0 < DBL_MAX) {
    p->d0 = p0.d0;
  }
  if (p0.hlim[0] < DBL_MAX) {
    p->hlim[0] = p0.hlim[0];
  }
  if (p0.hlim[1] < DBL_MAX) {
    p->hlim[1] = p0.hlim[1];
  }
  if (p0.res >= 0) {
    p->res = p0.res;
  }
  if (p0.split >= 0) {
    p->split = p0.split;
  }
  if (p0.labels >= 0) {
    p->labels = p0.labels;
  }
  if (p0.blockmax >= 0) {
    p->blockmax = p0.blockmax;
  }
  if (p0.dmax < DBL_MAX) {
    p->dmax = p0.dmax;
  }

  /* generate input file if requested */
  if (tofile) {
    generate_input_file(p, "inputs.input");
    return 2;
  }

  return 0;
}

/* ensures that the panorama data is valid, returning non-zero error. */
int validate_inputs(struct Panorama *p) {
  /* width must be an integer number of degrees */
  p->wlim[1] = p->wlim[0]+rint(p->wlim[1]-p->wlim[0]);

  return 0;
}

/* takes the block distances and fills blocks with the indices of the blocks
   in the order that they should be covered, and iblocks with the positions of
   each block (so iblocks[blocks[i]] = i and blocks[iblocks[j]] = j). Uses a
   breadth-first search of von Neumann neighbours */
void order_blocks(int *blocks, int *iblocks, int nby, int nbx, int I0, int J0, int *By, int *Bx) {
  /* indicate that none of the blocks have been visited */
  int nblocks = nby*nbx;
  for (int i = 0; i < nblocks; i++) {
    iblocks[i] = -1;
  } // i end

  /* find the origin index */
  for (int i = 0; i < nby; i++) {
    if (By[i] == I0) {
      I0 = i;
      break;
    }
  } // i end
  for (int j = 0; j < nbx; j++) {
    if (Bx[j] == J0) {
      J0 = j;
      break;
    }
  } // i end
  int K0 = I0*nbx + J0;

  /* start with the root index */
  int q0 = 0;
  int q1 = 0;
  int k = K0;
  blocks[q1] = k;
  iblocks[k] = q1;
  q1++;

  /* breadth-first traversal of von Neumann neighbours. blocks is the queue,
     iblocks records visited nodes */
  while (q1-q0 != 0) {
    /* finish if we've added all of the blocks */
    if (q1 == nblocks) {
      break;
    }

    /* get next index from start of queue */
    k = blocks[q0];
    q0++;

    /* convert to double index */
    int ik = k/nbx;
    int jk = k - ik*nbx;

    /* attempt to add [ik-1, jk] */
    if (ik > 0) {
      int k1 = k - nbx;
      if (iblocks[k1] == -1) {
        blocks[q1] = k1;
        iblocks[k1] = q1;
        q1++;
      }
    }

    /* attempt to add [ik, jk-1] */
    if (jk > 0) {
      int k1 = k - 1;
      if (iblocks[k1] == -1) {
        blocks[q1] = k1;
        iblocks[k1] = q1;
        q1++;
      }
    }

    /* attempt to add [ik+1, jk] */
    if (ik < nby-1) {
      int k1 = k + nbx;
      if (iblocks[k1] == -1) {
        blocks[q1] = k1;
        iblocks[k1] = q1;
        q1++;
      }
    }

    /* attempt to add [ik, jk+1] */
    if (jk < nbx-1) {
      int k1 = k + 1;
      if (iblocks[k1] == -1) {
        blocks[q1] = k1;
        iblocks[k1] = q1;
        q1++;
      }
    }
  } // end while
}


/* ========================================================================== */
/*   MAIN                                                                     */
/* ========================================================================== */
int main(int argc, char * const *argv) {
  int exit_code = EXIT_SUCCESS;

  /* ====================================================================== */
  /*   Set Up                                                               */
  /* ====================================================================== */
  /* basic profiling */
  clock_t ttotal = 0;
  clock_t tsetup = 0;
  clock_t tread = 0;
  clock_t tgrid = 0;
  clock_t tray = 0;
  clock_t tpng = 0;
  clock_t t0;
  clock_t tstart = clock();

  /* get inputs from command line */
  struct Panorama p; // wrapper for input data
  int loc; // whether to produce location info
  int err = read_inputs(&p, argc, argv, 1, &loc);
  if (err == 1) {
    fprintf(stderr, "ERROR: failed to read options\n");
    exit_code = EXIT_FAILURE;
    goto cleanup0;
  } else if (err == 2) {
    exit_code = EXIT_SUCCESS;
    goto cleanup0;
  }

  err = validate_inputs(&p);
  if (err) {
    fprintf(stderr, "ERROR: inputs not valid\n");
    exit_code = EXIT_FAILURE;
    goto cleanup0;
  }

  /* extract from struct */
  double x0 = p.x0;
  double y0 = p.y0;
  double z0 = p.z0;
  double d0 = p.d0;
  double wlim[2] = {p.wlim[0], p.wlim[1]};
  double hlim[2] = {p.hlim[0], p.hlim[1]};
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
  int nw = (int)((wlim[1]-wlim[0])*res);
  int nh = (int)((hlim[1]-hlim[0])*res) + 1;

  // note that the image is stored rotated so that we can iterate over the
  // columns for speed
  double **img_d = malloc_2d(nw, nh, sizeof(double)); // distance
  double **img_h = malloc_2d(nw, nh, sizeof(double)); // height (and water cover)
  int **img_n = malloc_2d(nw, nh, sizeof(int)); // number of steps to collision
  double **img_u = malloc_2d(nw, nh, sizeof(double)); // step x-direction
  double **img_v = malloc_2d(nw, nh, sizeof(double)); // step y-direction
  double **img_dz = malloc_2d(nw, nh, sizeof(double)); // step z-direction
  for (int j = 0; j < nw; j++) {
    for (int i = 0; i < nh; i++) {
      img_d[j][i] = -DBL_MAX;
      img_h[j][i] = -DBL_MAX;
      img_n[j][i] = 0;

      double theta = M_PI*(0.5 - (wlim[0]+j*dpx)/180.0);
      img_u[j][i] = cos(theta);
      img_v[j][i] = sin(theta);
      img_dz[j][i] = tan((hlim[0]+i*dpx)*M_PI/180.0);
    } // i end
  } // j end


  /* ====================================================================== */
  /*   Block Initialisation                                                 */
  /* ====================================================================== */
  /* dimensions */
  int ny, nx;
  err = get_block_dims(source, &ny, &nx);
  if (err) {
    fprintf(stderr, "ERROR: failed to recognise data source\n");
    exit_code = EXIT_FAILURE;
    goto cleanup1;
  }

  double **X = malloc_2d(ny, nx, sizeof(double)); // eastings grid
  double **Y = malloc_2d(ny, nx, sizeof(double)); // northings grid
  double **Zb = malloc_2d(ny, nx, sizeof(double)); // height grid

  /* read origin block */
  int I0, J0;
  err = block_index(source, y0, &I0, x0, &J0);
  if (err) {
    fprintf(stderr, "ERROR: block index for [%lf, %lf] not found\n", x0, y0);
    exit_code = EXIT_FAILURE;
    goto cleanup2;
  }
  err = get_block(I0, J0, source, X, Y, Zb);
  if (err) {
    fprintf(stderr, "ERROR: failed to read block [%d, %d] (returned %d)\n", I0, J0, err);
    exit_code = EXIT_FAILURE;
    goto cleanup2;
  }

  /* grid spacing */
  double cw = X[0][1] - X[0][0];
  double ch = Y[1][0] - Y[0][0];

  /* check levels are valid */
  err = validate_levels(ny, nx, nlevels, levels);
  if (err) {
    fprintf(stderr, "ERROR: level structure invalid\n");
    exit_code = EXIT_FAILURE;
    goto cleanup3;
  }

  /* set up multigrid */
  double ***Z;
  create_multigrid(ny, nx, nlevels, levels, Zb, &Z);

  /* order of block traversal */
  int nbx, nby;
  int *Bx, *By;
  err = get_data_extent(source, &nbx, &nby, &Bx, &By);
  if (err) {
    fprintf(stderr, "ERROR: failed to get data extent\n");
    exit_code = EXIT_FAILURE;
    goto cleanup4;
  }
  int nblocks = nbx*nby;
  int *blocks = malloc(nblocks*sizeof(int)); // block order
  int *iblocks = malloc(nblocks*sizeof(int)); // position of block in order
  order_blocks(blocks, iblocks, nby, nbx, I0, J0, By, Bx);

  /* origin block indices */
  int IJb0 = blocks[0]; // single index
  int Ib0 = IJb0/nbx; // row index
  int Jb0 = IJb0-Ib0*nbx; // column index

  /* location map (if required) */
  double **Xloc = NULL;
  double **Yloc = NULL;
  double **Zloc = NULL;
  int loc_I = 0;
  int loc_J = 0;
  int loc_nblocks = 1;
  int *loc_blocks = NULL;
  if (loc) {
    /* how many blocks are required for sufficient detail */
    const double loc_size = 5000.0;
    while (loc_nblocks*cw*nx < loc_size || loc_nblocks*ch*ny < loc_size) {
      loc_nblocks += 2;
      loc_I += 1;
      loc_J += 1;
    }

    /* decide which blocks to store */
    loc_blocks = malloc(loc_nblocks*loc_nblocks*sizeof(int));
    int ib = 0, jb = 0;
    while (ib < loc_nblocks*loc_nblocks) {
      int IJb = blocks[jb]; // single index
      int Ib = IJb/nbx; // row index
      int Jb = IJb-Ib*nbx; // column index
      if (abs(Ib-Ib0) <= loc_nblocks/2 && abs(Jb-Jb0) <= loc_nblocks/2) {
        loc_blocks[ib] = blocks[jb];
        ib++;
      }
      jb++;
    }

    Xloc = malloc_2d(loc_nblocks*ny, loc_nblocks*nx, sizeof(double));
    Yloc = malloc_2d(loc_nblocks*ny, loc_nblocks*nx, sizeof(double));
    Zloc = malloc_2d(loc_nblocks*ny, loc_nblocks*nx, sizeof(double));
  }


  /* ====================================================================== */
  /*   Ray Initialisation                                                   */
  /* ====================================================================== */
  int nrays = nw*nh;
  int **rays = malloc_2d(blockmax, nrays+1, sizeof(int));
  if (!rays) {
    fprintf(stderr, "ERROR: failed to allocate ray memory\n");
    exit_code = EXIT_FAILURE;
    goto cleanup5;
  }

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
  tsetup = clock() - tstart;


  /* ====================================================================== */
  /*   Ray Tracing                                                          */
  /* ====================================================================== */
  /* absolute viewpoint height */
  int i0 = (int)((y0 - Y[0][0])/ch + 0.5);
  int j0 = (int)((x0 - X[0][0])/cw + 0.5);
  double zrel = z0; // store relative height
  z0 += Zb[i0][j0];
  int loc_i = 0; // keep track of stored blocks

  /* loop over the data blocks in order of distance from the origin block */
  for (int block = 0; block < blockmax; block++) {
    /* ======================== */
    /*  Fill elevation pyramid  */
    /* ======================== */
    t0 = clock();
    int IJb = blocks[block]; // single index
    int Ib = IJb/nbx; // row index
    int Jb = IJb-Ib*nbx; // column index
    err = get_block(By[Ib], Bx[Jb], source, X, Y, Zb);
    if (err != 0) {
      /* skip data blocks that don't exist */
      continue;
    }
    tread += clock() - t0;

    t0 = clock();
    fill_multigrid(ny, nx, nlevels, levels, Z);
    tgrid += clock() - t0;

    /* store block for location image if required */
    if (loc && loc_i < loc_nblocks*loc_nblocks) {
      if (IJb == loc_blocks[loc_i]) {
        int ii = (loc_I+(Ib-Ib0))*ny;
        int jj = (loc_J+(Jb-Jb0))*nx;
        for (int i = 0; i < ny; i++) {
          for (int j = 0; j < nx; j++) {
            Xloc[ii+i][jj+j] = X[i][j];
            Yloc[ii+i][jj+j] = Y[i][j];
            Zloc[ii+i][jj+j] = Z[0][i][j];
          } // j end
        } // i end
        loc_i++;
      }
    }

    /* ================================== */
    /*  Propagate rays through the block  */
    /* ================================== */
    int i00, j00; // start index
    int i = -4, j = -4; // current index
    double y00, x00, z00; // start position
    int block1; // next block
    int dprev = -1; // previous step (0 for vertical, 1 for horizontal)
    double ty, tx; // vertical and horizontal distance

    t0 = clock();
    int Jprev = -1; // previous column
    for (int k = 0; k < rays[block][nrays]; k++) {
      /* ============== */
      /* create the ray */
      /* ============== */
      /* index */
      int IJ = rays[block][k];
      int J = IJ/nh; // column index (recalling that img is column-major)

      /* variables */
      double u = img_u[0][IJ]; // ray direction vector
      double v = img_v[0][IJ];
      int stepy = (v>0.0) - (v<0.0); // step directions
      int stepx = (u>0.0) - (u<0.0);
      double dty = fabs(ch/v); // dt to cross one cell
      double dtx = fabs(cw/u);
      double dz = img_dz[0][IJ];

      /* =============================== */
      /* start position, index, and edge */
      /* =============================== */
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
            default:
              fprintf(stderr, "ERROR: exit side not valid\n");
              exit_code = EXIT_FAILURE;
              goto cleanup6;
          }
          if (stepy != 0) { // if stepy == 0 then v == 0 and u/v is not defined
            t1 = (x0-x1 + (u/v)*(y1-y0)) / (dx1 - (u/v)*dy1);
          } else {
            t1 = (y0-y1) / dy1;
          }
          y00 = y1 + t1*dy1; x00 = x1 + t1*dx1;
        }

        /* start index */
        i00 = (int)((y00 - Y[0][0])/ch + 0.5);
        j00 = (int)((x00 - X[0][0])/cw + 0.5);
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
        if (t+d00 > d0 && Z[level][i/l][j/l] > z) {
          if (level == 0) { // collision
            /* set image and record final position for next ray */
            Jprev = J;
            img_d[0][IJ] = t + d00;
            img_h[0][IJ] = Z[0][i][j];

            /* check for water */
            double zc = Z[0][i][j];
            const double watertol = 0.0;
            if (i <= 1) {
              if (fabs(Z[0][i+1][j]-zc) > watertol) {
                break;
              }
              if (fabs(Z[0][i+2][j]-zc) > watertol) {
                break;
              }
              if (j <= 1) {
                if (fabs(Z[0][i][j+1]-zc) > watertol) {
                  break;
                }
                if (fabs(Z[0][i][j+2]-zc) > watertol) {
                  break;
                }
              } else if (j >= nx-2) {
                if (fabs(Z[0][i][j-1]-zc) > watertol) {
                  break;
                }
                if (fabs(Z[0][i][j-2]-zc) > watertol) {
                  break;
                }
              } else {
                if (fabs(Z[0][i][j+1]-zc) > watertol) {
                  break;
                }
                if (fabs(Z[0][i][j-1]-zc) > watertol) {
                  break;
                }
                if (fabs(Z[0][i][j+2]-zc) > watertol) {
                  break;
                }
                if (fabs(Z[0][i][j-2]-zc) > watertol) {
                  break;
                }
              }
            } else if (i >= ny-2) {
              if (fabs(Z[0][i-1][j]-zc) > watertol) {
                break;
              }
              if (fabs(Z[0][i-2][j]-zc) > watertol) {
                break;
              }
              if (j <= 1) {
                if (fabs(Z[0][i][j+1]-zc) > watertol) {
                  break;
                }
                if (fabs(Z[0][i][j+2]-zc) > watertol) {
                  break;
                }
              } else if (j >= nx-2) {
                if (fabs(Z[0][i][j-1]-zc) > watertol) {
                  break;
                }
                if (fabs(Z[0][i][j-2]-zc) > watertol) {
                  break;
                }
              } else {
                if (fabs(Z[0][i][j+1]-zc) > watertol) {
                  break;
                }
                if (fabs(Z[0][i][j-1]-zc) > watertol) {
                  break;
                }
                if (fabs(Z[0][i][j+2]-zc) > watertol) {
                  break;
                }
                if (fabs(Z[0][i][j-2]-zc) > watertol) {
                  break;
                }
              }
            } else {
              if (fabs(Z[0][i+1][j]-zc) > watertol) {
                break;
              }
              if (fabs(Z[0][i+2][j]-zc) > watertol) {
                break;
              }
              if (fabs(Z[0][i-1][j]-zc) > watertol) {
                break;
              }
              if (fabs(Z[0][i-2][j]-zc) > watertol) {
                break;
              }
              if (j <= 1) {
                if (fabs(Z[0][i][j+1]-zc) > watertol) {
                  break;
                }
                if (fabs(Z[0][i][j+2]-zc) > watertol) {
                  break;
                }
              } else if (j >= nx-2) {
                if (fabs(Z[0][i][j-1]-zc) > watertol) {
                  break;
                }
                if (fabs(Z[0][i][j-2]-zc) > watertol) {
                  break;
                }
              } else {
                if (fabs(Z[0][i][j+1]-zc) > watertol) {
                  break;
                }
                if (fabs(Z[0][i][j-1]-zc) > watertol) {
                  break;
                }
                if (fabs(Z[0][i][j+2]-zc) > watertol) {
                  break;
                }
                if (fabs(Z[0][i][j-2]-zc) > watertol) {
                  break;
                }
              }
            }

            img_h[0][IJ] = -DBL_MAX; // leave as -DBL_MAX if it's water
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
              i = l1*(i/l1) + (stepy!=1)*(l1-1);

              /* find correct j index */
              j = (int)(((x00+ty*u) - X[0][0])/cw + 0.5);

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

              /* rewind tx (used to use rint) */
              tx -= ((tx-(X[i][j]-x00+stepx*0.5*l*cw)/u)/(l*dtx))*l*dtx;

              /* move back forwards again (at the lower level) */
              ty += l*dty;
            } else if (dprev == 1) {
              /* rewind tx */
              tx -= l1*dtx;

              /* find correct j index */
              j = l1*(j/l1) + (stepx!=1)*(l1-1);

              /* find correct i index */
              i = (int)(((y00+tx*v) - Y[0][0])/ch + 0.5);

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

              /* rewind ty (used to use rint) */
              ty -= ((ty-(Y[i][j]-y00+stepy*0.5*l*ch)/v)/(l*dty))*l*dty;

              /* move back forwards again (at the lower level) */
              tx += l*dtx;
            } else {
              fprintf(stderr, "ERROR: (unreachable) go down level at start\n");
              exit_code = EXIT_FAILURE;
              goto cleanup6;
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
    tray += clock() - t0;
  } // block end (loop over all data blocks)


  /* ====================================================================== */
  /*   GENERATE IMAGE                                                       */
  /* ====================================================================== */
  /* check dmax */
  if (dmax < 0.0) {
    /* find maximum */
    for (int IJ = 0; IJ < nrays; IJ++) {
      if (img_d[0][IJ] > dmax && img_h[0][IJ] > -DBL_MAX) {
        dmax = img_d[0][IJ];
      }
    } // IJ end
    p.dmax = dmax;
  }

  /* pack data into struct */
  struct Img_pan img;
  img.nw = nw;
  img.nh = nh;
  img.img_d = img_d;
  img.img_h = img_h;
  img.img_n = img_n;
  img.img_u = img_u;
  img.img_v = img_v;
  img.z = zrel;
  p.z0 = z0 - zrel;

  /* write to file */
  t0 = clock();
  err = panorama_write("out/panorama.png", &img, &p);
  tpng = clock() - t0;
  if (err) {
    fprintf(stderr, "ERROR: failed to write image\n");
    exit_code = EXIT_FAILURE;
  }

  /* location write */
  if (loc) {
    struct Img_loc img_loc;
    img_loc.nh = loc_nblocks*ny;
    img_loc.nw = loc_nblocks*nx;
    img_loc.X = Xloc;
    img_loc.Y = Yloc;
    img_loc.Z = Zloc;
    img_loc.i0 = ny*loc_I+i0;
    img_loc.j0 = nx*loc_J+j0;
    location_write("out/location.png", &img_loc, &p);
  }


  /* ====================================================================== */
  /*   Basic Profiling                                                      */
  /* ====================================================================== */
  ttotal = clock() - tstart;

  double stotal = ((double)ttotal)/CLOCKS_PER_SEC;
  double ssetup = ((double)tsetup)/CLOCKS_PER_SEC;
  double sread = ((double)tread)/CLOCKS_PER_SEC;
  double sgrid = ((double)tgrid)/CLOCKS_PER_SEC;
  double sray = ((double)tray)/CLOCKS_PER_SEC;
  double spng = ((double)tpng)/CLOCKS_PER_SEC;
  double sother = stotal - (ssetup+sread+sgrid+sray+spng);

  fprintf(stderr, "setup: %6.3lf [%6.2lf%%]\n", ssetup, 100.0*ssetup/stotal);
  fprintf(stderr, " read: %6.3lf [%6.2lf%%]\n", sread, 100.0*sread/stotal);
  fprintf(stderr, " grid: %6.3lf [%6.2lf%%]\n", sgrid, 100.0*sgrid/stotal);
  fprintf(stderr, "  ray: %6.3lf [%6.2lf%%]\n", sray, 100.0*sray/stotal);
  fprintf(stderr, "  png: %6.3lf [%6.2lf%%]\n", spng, 100.0*spng/stotal);
  fprintf(stderr, "other: %6.3lf [%6.2lf%%]\n", sother, 100.0*sother/stotal);
  fprintf(stderr, "total: %6.3lf [%6.2lf%%]\n", stotal, 100.0*stotal/stotal);


  /* ====================================================================== */
  /*   Clean Up                                                             */
  /* ====================================================================== */
  cleanup6:
  free(rays_entry);

  cleanup5:
  free(blocks);
  free(iblocks);
  free_2d(Xloc);
  free_2d(Yloc);
  free_2d(Zloc);
  free(loc_blocks);
  free_2d(rays);

  cleanup4:
  free(Bx);
  free(By);

  cleanup3:
  free_multigrid(nlevels, Z);

  cleanup2:
  free_2d(X);
  free_2d(Y);
  free_2d(Zb);

  cleanup1:
  free_2d(img_d);
  free_2d(img_h);
  free_2d(img_n);
  free_2d(img_u);
  free_2d(img_v);
  free_2d(img_dz);

  cleanup0:
  free(p.levels);

  return exit_code;
}

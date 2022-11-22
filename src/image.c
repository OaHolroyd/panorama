#include "image.h"

#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <png.h>
// #include <zlib.h>

#include "data.h"
#include "gazetteer.h"
#include "colors.h"
#include "font.h"
#include "utils.h"

/* lower bound for feasible height (lowest point visible is -430m) */
#define MIN_H (-500.0)

struct img_writer {
  /* file writer info */
  png_structp png;
  png_infop info;
  FILE *fp;

  /* image details */
  int w, h; // width/height
};


/* ========================================================================== */
/*   AUXILIARY FUNCTION DEFINITIONS                                           */
/* ========================================================================== */
/* sets up the image writer (wrapper for libpng boilerplate). Returns nonzero
   exit code on failure. */
int img_writer_setup(struct img_writer *w, const char * restrict path) {
  w->png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if (!w->png) {
    return 1;
  }
  w->info = png_create_info_struct(w->png);
  if (!w->info) {
    png_destroy_write_struct(&w->png, NULL);
    return 1;
  }

  /* open the file */
  w->fp = fopen(path, "wb");
  if (!w->png) {
    png_destroy_write_struct(&w->png, &w->info);
    return 2;
  }

  /* set failure point */
  if (setjmp(png_jmpbuf(w->png))) {
    png_destroy_write_struct(&w->png, &w->info);
    fclose(w->fp);
    return 3;
  }

  /* set output */
  png_init_io(w->png, w->fp);

  /* set filtering options */
  // TODO: see if setting filters is useful
  // png_set_filter(0, /* filters */);

  /* set the zlib compression level */
  // TODO: see if setting compression level is useful
  // png_set_compression_level(png, Z_BEST_COMPRESSION);

  /* fill the image header chunk */
  png_set_IHDR(
    w->png, w->info,
    w->w, w->h, // image dimensions
    8, // color bit-depth
    PNG_COLOR_TYPE_PALETTE, // index into a defined color palette
    PNG_INTERLACE_NONE, // disable interlacing
    PNG_COMPRESSION_TYPE_DEFAULT, // fixed
    PNG_FILTER_TYPE_DEFAULT // fixed
  );

  // TODO: fill tIME

  /* fill text chunk */
  // const int num_text = 1;
  // png_text text[num_text];
  // char key0[] = "Title";
  // char text0[] = "Panorama";
  // text[0].key = key0;
  // text[0].text = text0;
  // text[0].compression = PNG_TEXT_COMPRESSION_NONE;
  // text[0].itxt_length = 0;
  // text[0].lang = NULL;
  // text[0].lang_key = NULL;
  // png_set_text(w->png, w->info, text, num_text);

  // TODO: set gAMA: png_set_gAMA(png_ptr, info_ptr, gamma);

  /* set the color palette (max 256 colors) */
  if (NUM_COLORS > 256) {
    png_destroy_write_struct(&w->png, &w->info);
    fclose(w->fp);
    return 4;
  }
  png_set_PLTE(w->png, w->info, palette, NUM_COLORS);

  /* write chunks */
  png_write_info(w->png, w->info);

  return 0;
}

/* writes the image in image (which must have the dimensions specified in w)
   to the file. */
void img_writer_image(struct img_writer *w, png_byte **image) {
  /* set up rows of pixels */
  // TODO: should this use png_malloc?
  png_bytep *rows = malloc(w->h*sizeof(png_bytep));
  for (int i = 0; i < w->h; i++) {
    rows[i] = image[w->h-i-1];
  } // i end

  /* write the image data */
  png_write_image(w->png, rows);

  free(rows);
}

/* finished the image writing process */
void img_writer_end(struct img_writer *w) {
  /* finish writing */
  png_write_end(w->png, w->info);
  png_destroy_write_struct(&w->png, &w->info);
  fclose(w->fp);
}

/* adds the characters in the string str to the image, scaled by size. (i0, j0)
   denotes the upper left corner or the upper central point of the string,
   depending if centred is true. Uses fcol (>=0) as the foreground color and
   bcol as the background color (or blank if bcol = 0). Returns the
   x coordinate of the next blank space. The length of the string is given by
   len. If it is negative this is computed with strlen. */
int add_text(const char *str, int len, int i0, int j0, int size, png_byte **image, int centred, png_byte fcol, png_byte bcol) {
  const unsigned char *c;

  if (len < 0) {
    len = (int)strlen(str);
  }

  /* check for newlines */
  for (int i = 0; i < len; i++) {
    if (str[i] == '\n') {
      int j1 = add_text(str, i, i0, j0, size, image, centred, fcol, bcol);
      int j2 = add_text(&(str[i+1]), -1, i0-size*CHAR_H, j0, size, image, centred, fcol, bcol);
      return (j1 > j2) ? j1 : j2;
    }
  } // i end

  /* start x coordinate */
  if (centred) {
    j0 -= (len*size*CHAR_W)/2;
  }
  int j1 = j0;

  for (int k = 0; k < len; k++) {
    int i1 = i0;
    c = CHAR_8BIT(str[k]);

    /* print character */
    for (int i = CHAR_H-1; i >= 0; i--) {
      j1 = j0 + k*CHAR_W*size;
      for (int j = CHAR_W-1; j >= 0; j--) {
        png_byte set = (c[i] & 1 << j) ? fcol : bcol;

        /* scale pixels by size */
        for (int l0 = 0; l0 < size; l0++) {
          for (int l1 = 0; l1 < size; l1++) {
            if (set != COLOR_NULL) {
              image[i1-l0][j1+l1] = set;
            }
          } // l1 end
        } // l0 end
        j1 += size;
      } // j end
      i1 -= size;
    } // i end
  } // k end

  return j1;
}

/* checks if str, scaled by size, will overlap an existing string. (i0, j0)
   denotes the upper left corner or the upper central point of the string,
   depending if centred is true. Uses fcol (>=0) as the foreground color and
   bcol as the background color (or blank if bcol = 0). The length of the string
   is given by len. If it is negative this is computed with strlen.Returns 1 if
   the space is free, 0 otherwise. */
// TODO: add const back by including a len argument
int check_text(const char *str, int len, int i0, int j0, int size, png_byte **image, int centred, int fcol, int bcol) {
  if (len < 0) {
    len = (int)strlen(str);
  }

  /* check for newlines */
  for (int i = 0; i < len; i++) {
    if (str[i] == '\n') {
      int j1 = check_text(str, i, i0, j0, size, image, centred, fcol, bcol);
      int j2 = check_text(&(str[i+1]), -1, i0-size*CHAR_H, j0, size, image, centred, fcol, bcol);
      return (j1 && j2);
    }
  } // i end

  /* start x coordinate */
  if (centred) {
    j0 -= (len*size*CHAR_W)/2;
  }
  int j1 = j0;

  int all_sky = 1;
  int no_sky = 1;

  for (int k = 0; k < len; k++) {
    int i1 = i0;

    /* print character */
    for (int i = CHAR_H-1; i >= 0; i--) {
      j1 = j0 + k*CHAR_W*size;
      for (int j = CHAR_W-1; j >= 0; j--) {
        /* scale pixels by size */
        for (int l0 = 0; l0 < size; l0++) {
          for (int l1 = 0; l1 < size; l1++) {
            if (image[i1-l0][j1+l1] == fcol || image[i1-l0][j1+l1] == bcol) {
              return 0;
            }

            if (image[i1-l0][j1+l1] == COLOR_SKY) {
              no_sky = 0;
            }

            if (image[i1-l0][j1+l1] >= COLORMAP(0.0)) {
              all_sky = 0;
            }
          } // l1 end
        } // l0 end
        j1 += size;
      } // j end
      i1 -= size;
    } // i end
  } // k end

  /* if the label obscures the horizon, the label is not well-placed */
  if (all_sky || no_sky) {
    return 1;
  } else {
    return 0;
  }
}

/* if the string is more than tol chars long, add a newline near the centre */
void split_string(char *str, int tol) {
  int len = (int)strlen(str);
  if (len <= tol) {
    return;
  }

  int i = len/2;

  if (str[i] == ' ') {
    str[i] = '\n';
    return;
  }

  int j = 1;
  while (j/2 < i) {
    if (str[i+j] == ' ') {
      str[i+j] = '\n';
      return;
    }

    if (str[i-j] == ' ') {
      str[i-j] = '\n';
      return;
    }

    j++;
  }
}


/* ========================================================================== */
/*   FUNCTION DEFINITIONS                                                     */
/* ========================================================================== */
/* takes the image data and writes it to the file whose name is the string
   pointed to by path. Returns an error code:
     0 - success
     1 - failed to allocate png struct or info
     2 - failed to open file
     3 - subsequent write failure
     4 - color palette too large
  */
int panorama_write(const char * restrict path, struct Img_pan *img, struct Panorama *p) {
  /* extract data from container */
  int nw = img->nw;
  int nh = img->nh;
  double *wlim = p->wlim;
  double **img_d = img->img_d;
  double **img_h = img->img_h;
  double *img_u = img->img_u;
  double *img_v = img->img_v;
  int split = p->split;


  /* ====================== */
  /*   SET UP PNG WRITING   */
  /* ====================== */
  struct img_writer writer;

  /* compute size of output image */
  const int pad = nw/625; // padding between split rows
  int head = nw/125; // space for header
  int foot = nw/250; // space for footer

  /* text sizes */
  int clarge = (int)(head/(2.0*CHAR_H));
  int cmed = (int)(head/(4.0*CHAR_H));
  int csmall = (int)(head/(6.0*CHAR_H));
  int ctiny = (int)(head/(8.0*CHAR_H));
  clarge = (clarge == 0) ? 1 : clarge;
  cmed = (cmed == 0) ? 1 : cmed;
  csmall = (csmall == 0) ? 1 : csmall;
  ctiny = (ctiny == 0) ? 1 : ctiny;

  /* ensure header and footer are large enough to contain text */
  head = (head < clarge*CHAR_H+pad) ? clarge*CHAR_H+pad : head;
  foot = (foot < clarge*CHAR_H+pad) ? clarge*CHAR_H+pad : foot;

  if (split) {
    writer.w = nw/8 + 2*pad;
    writer.h = nh*8 + head + foot + 7*pad;
  } else {
    writer.w = nw + 2*pad;
    writer.h = nh + head + foot;
  }

  int err = img_writer_setup(&writer, path);
  if (err) {
    return err;
  }


  /* ========================= */
  /*   CONVERT DATA TO IMAGE   */
  /* ========================= */
  /* image (represented as 8-bit indices into the palette) */
  // TODO: should this use png_malloc?
  png_byte **image = malloc_2d(writer.h, writer.w, sizeof(png_byte));

  /* find edges -- 0 for no edge, 1 for major, 2 for minor */
  int **edges = malloc_2d(nw, nh, sizeof(int));
  const double alpha = 2.5;
  const double beta = 2.0;
  for (int j = 0; j < nw; j++) {
    for (int i = 1; i < nh; i++) {
      if (img_h[j][i] < MIN_H && img_h[j][i-1] < MIN_H) {
        /* water has no edges */
        edges[j][i] = 0;
      } else if (fabs(1.0 - img_d[j][i-1]/img_d[j][i]) > 0.05) {
        /* big distance jump */
        edges[j][i] = 1;
      } else if (img_h[j][i] < MIN_H && img_h[j][i-1] > MIN_H) {
        /* turn from land to water */
        edges[j][i] = 1;
      } else if (img_h[j][i] > MIN_H && img_h[j][i-1] < MIN_H) {
        /* turn from water to land */
        edges[j][i] = 1;
      } else if (img_d[j][i] < 0.0) {
        /* hit the sky */
        edges[j][i] = 0;
        break;
      } else if (floor(pow(img_d[j][i]/alpha, 1.0/beta)) - floor(pow(img_d[j][i-1]/alpha, 1.0/beta)) > 0.1) {
        /* minor edge */
        edges[j][i] = 2;
      } else {
        edges[j][i] = 0;
      }
    } // i end
  } // j end

  /* white background */
  for (int i = 0; i < writer.h; i++) {
    for (int j = 0; j < writer.w; j++) {
      image[i][j] = COLOR_WHITE;
    } // j end
  } // i end

  /* write distance data */
  for (int i = 0; i < nh; i++) {
    for (int j = 0; j < nw; j++) {
      int k = (j*8)/nw;
      int ii = i + foot + split*((7-k)*(nh+pad));
      int jj = j + pad - split*(k*(nw/8));
      if (edges[j][i] == 1) {
        image[ii][jj] = COLOR_BLACK;
      } else if (edges[j][i] == 2) {
        image[ii][jj] = COLOR_GREY;
      } else if (img_d[j][i] < 0.0) {
        image[ii][jj] = COLOR_SKY;
      } else if (img_h[j][i] < MIN_H) {
        image[ii][jj] = COLOR_WATER;
      } else {
        double d = img_d[j][i]/p->dmax;
        if (d > 1.0) { d = 1.0; }
        image[ii][jj] = COLORMAP(d);
      }
    } // j end
  } // i end

  /* recompute dmax to true max */
  double dmax = 0.0;
  for (int j = 0; j < nw; j++) {
    for (int i = 0; i < nh; i++) {
      if (img_d[j][i] > dmax) {
        dmax = img_d[j][i];
      }
    } // i end
  } // j end


  /* ============ */
  /*   ADD TEXT   */
  /* ============ */
  char name[64];
  char str[128];

  /* top left */
  int i0 = writer.h-pad-1;
  int j0 = pad;

  /* name */
  init_gazetteer(p->source);
  double derr = gazetteer_nearest(p->y0, p->x0, name);

  if (derr > 200.0) {
    /* if the error is too large, mark as unknown */
    strcpy(name, "Unknown location");
  }
  if (head-pad >= 2*cmed*CHAR_H) {
    add_text("view ", -1, i0, j0, cmed, image, 0, COLOR_BLACK, COLOR_NULL);
    j0 = add_text("from ", -1, i0-(CHAR_H+1)*cmed, j0, cmed, image, 0, COLOR_BLACK, COLOR_NULL);
  } else {
    j0 = add_text("view from ", -1, i0, j0, cmed, image, 0, COLOR_BLACK, COLOR_NULL);
  }
  sprintf(str, "%s, ", name);
  j0 = add_text(str, -1, i0, j0, clarge, image, 0, COLOR_BLACK, COLOR_NULL);

  /* grid reference */
  err = 0;
  switch (p->source) {
    case OST50:
      err = ne_to_osng(p->y0, p->x0, str, 8, 1);
      break;
    case SWT02:
      err = ne_to_swgr(p->y0, p->x0, str, 1);
      break;
    case NZT08:
      // TODO: although this is correct it would be more normal to refer to the correct 1:50 map code
      err = ne_to_swgr(p->y0, p->x0, str, 1);
      break;
    default :
      ERROR("data source not recognised");
      break;
  }
  if (err) {
    strcpy(str, "unknown gridref");
  }
  j0 = add_text(str, -1, i0, j0, clarge, image, 0, COLOR_BLACK, COLOR_NULL);
  j0 = add_text(" ", -1, i0, j0, clarge, image, 0, COLOR_BLACK, COLOR_NULL);

  /* other details */
  if (head-pad >= 2*cmed*CHAR_H) {
    sprintf(str, "  altitude: %.0lf m", p->z0);
    add_text(str, -1, i0, j0, cmed, image, 0, COLOR_BLACK, COLOR_NULL);
    sprintf(str, "eye height: %.0lf m", img->z);
    add_text(str, -1, i0-(CHAR_H+1)*cmed, j0, cmed, image, 0, COLOR_BLACK, COLOR_NULL);
  } else {
    sprintf(str, "  altitude: %.0lf m, ", p->z0);
    j0 = add_text(str, -1, i0, j0, cmed, image, 0, COLOR_BLACK, COLOR_NULL);
    sprintf(str, "eye height: %.0lf m", img->z);
    add_text(str, -1, i0, j0, cmed, image, 0, COLOR_BLACK, COLOR_NULL);
  }

  /* bottom left */
  i0 = foot-pad;
  j0 = pad;

  /* attributions */
  switch (p->source) {
    case OST50:
      sprintf(str, "generated from Ordnance Survey Terrain 50 data (Crown copyright)");
      break;
    case SWT02:
      sprintf(str, "generated from Federal Office of Topography swisstopo swissALTI3D data (copyright swisstopo)");
      break;
    case NZT08:
      sprintf(str, "generated from Land Information New Zealand 8m DEM (CC BY 4.0)");
      break;
    default :
      ERROR("data source not recognised");
      break;
  }
  split_string(str, writer.w/(CHAR_W*csmall));
  add_text(str, -1, i0, j0, csmall, image, 0, COLOR_BLACK, COLOR_NULL);


  /* ================ */
  /*   IMAGE LABELS   */
  /* ================ */
  /* zones */
  int radius = nh/12;
  if (split) {
    for (int k = 0; k < 8; k++) {
      i0 = foot + (7-k)*(nh+pad) + pad + radius;
      j0 = 2*pad + radius;
      double w0 = M_PI*(0.5 - (wlim[0]+k*(nw/(8*p->res)))/180.0);
      double w1 = M_PI*(0.5 - (wlim[0]+(k+1)*(nw/(8*p->res)))/180.0);

      /* draw circle */
      for (int i = -radius; i < radius; i++) {
        for (int j = -radius; j < radius; j++) {
          int d = i*i + j*j;
          if (d < radius*radius) {
            image[i0 + i][j0 + j] = COLOR_WHITE;

            /* draw zone */
            double a = atan2(i, j);
            if (a > w1 && a <= w0) {
              image[i0 + i][j0 + j] = COLOR_RED;
              continue;
            }
            a -= 2.0*M_PI; // account for equivalent angles
            if (a > w1 && a <= w0) {
              image[i0 + i][j0 + j] = COLOR_RED;
              continue;
            }
            a += 4.0*M_PI; // account for equivalent angles
            if (a > w1 && a <= w0) {
              image[i0 + i][j0 + j] = COLOR_RED;
              continue;
            }
          }
        } // j end
      } // i end
    } // k end
  } else {
    i0 = foot + pad + radius;
    j0 = 2*pad + radius;
    double w0 = M_PI*(0.5 - wlim[0]/180.0);
    double w1 = M_PI*(0.5 - wlim[1]/180.0);

    /* draw circle */
    for (int i = -radius; i < radius; i++) {
      for (int j = -radius; j < radius; j++) {
        int d = i*i + j*j;
        if (d < radius*radius) {
          image[i0 + i][j0 + j] = COLOR_WHITE;

          /* draw zone */
          double a = atan2(i, j);
          if (a > w1 && a <= w0) {
            image[i0 + i][j0 + j] = COLOR_RED;
            continue;
          }
          a -= 2.0*M_PI; // account for equivalent angles
          if (a > w1 && a <= w0) {
            image[i0 + i][j0 + j] = COLOR_RED;
            continue;
          }
          a += 4.0*M_PI; // account for equivalent angles
          if (a > w1 && a <= w0) {
            image[i0 + i][j0 + j] = COLOR_RED;
            continue;
          }
        }
      } // j end
    } // i end
  }

  /* cardinal points and bearings */
  for (int j = 0; j < nw; j++) {
    int k = (j*8)/nw;
    int ii = nh-2 + foot + split*((7-k)*(nh+pad));
    int jj = j + pad - split*(k*(nw/8));

    /* label 8 main compass directions and then every dd degrees */
    const double dd = 10.0;
    double d = fmod(wlim[0]+(double)j/p->res + 1440.0, 360.0);
    if (fabs(d-0.0) < 0.5/p->res) {
      add_text("NORTH", -1, ii, jj, csmall, image, 1, COLOR_RED, COLOR_BACK);
    } else if (fabs(d-180.0) < 0.5/p->res) {
      add_text("SOUTH", -1, ii, jj, csmall, image, 1, COLOR_RED, COLOR_BACK);
    } else if (fabs(d-90.0) < 0.5/p->res) {
      add_text("EAST", -1, ii, jj, csmall, image, 1, COLOR_RED, COLOR_BACK);
    } else if (fabs(d-270.0) < 0.5/p->res) {
      add_text("WEST", -1, ii, jj, csmall, image, 1, COLOR_RED, COLOR_BACK);
    } else if (fabs(d-45.0) < 0.5/p->res) {
      add_text("NE", -1, ii, jj, csmall, image, 1, COLOR_RED, COLOR_BACK);
    } else if (fabs(d-135.0) < 0.5/p->res) {
      add_text("SE", -1, ii, jj, csmall, image, 1, COLOR_RED, COLOR_BACK);
    } else if (fabs(d-225.0) < 0.5/p->res) {
      add_text("SW", -1, ii, jj, csmall, image, 1, COLOR_RED, COLOR_BACK);
    } else if (fabs(d-315.0) < 0.5/p->res) {
      add_text("NW", -1, ii, jj, csmall, image, 1, COLOR_RED, COLOR_BACK);
    } else if (fabs(fmod(d, dd)) < 0.5/p->res) {
      sprintf(str, "%03.0lf", fmod(dd*rint(d/dd)+360.0, 360.0));
      add_text(str, -1, ii, jj, csmall, image, 1, COLOR_RED, COLOR_BACK);
    }
  } // j end

  /* hill labels */
  if (p->labels) {
    for (int l = 0; l < gaz_n; l++) {
      double y = gaz_y[l];
      double x = gaz_x[l];

      /* skip ones that are too far away */
      double dy = p->y0 - y;
      double dx = p->x0 - x;
      double d = dx*dx + dy*dy;
      const double dcut = dmax*dmax*5;
      if (d > dcut) {
        continue;
      }

      /* find direction index to reduce search area */
      double a = atan2(-dx, -dy) * 180.0/M_PI;
      if (a < wlim[0]) {
        a += 360.0;
      } else if (a > wlim[1]) {
        a -= 360.0;
      }
      int ja = (int)((a - wlim[0]) * p->res);
      int ja0 = ja - p->res;
      int ja1 = ja + p->res;
      if (ja0 < 0) {
        ja0 = 0;
      }
      if (ja1 > nw) {
        ja1 = nw;
      }

      /* find closest point to hill */
      int imin = -1;
      int jmin = -1;
      double dmin = DBL_MAX;
      int count = 0;
      double imean = 0.0;
      double jmean = 0.0;
      const double dtol = 200;
      for (int j = ja0; j < ja1; j++) {
        for (int i = 0; i < nh; i++) {
          if (img_d[j][i] < 0.0) {
            break;
          }

          dy = p->y0 + img_d[j][i]*img_v[j] - y;
          dx = p->x0 + img_d[j][i]*img_u[j] - x;
          d = dx*dx + dy*dy;

          if (d < dtol*dtol) {
            imean += i;
            jmean += j;
            count++;
          }

          if (d < dmin) {
            dmin = d;
            imin = i;
            jmin = j;
          }
        } // j end
      } // i end

      /* skip if we didn't see it */
      if (count == 0) {
        continue;
      }

      /* use the average point to centre the label */
      imean /= count;
      jmean /= count;
      imin = (int)imean;
      jmin = (int)jmean;

      /* skip if it's not a real point */
      if (img_d[jmin][imin] < 0.0) {
        continue;
      }

      /* if the peak satisfies some desirability threshold, label it */
      dmin = sqrt(dmin);
      double label1 = 0.8*img_d[jmin][imin]*count/(1000.0*p->res*p->res);
      double label2 = img_h[jmin][imin]/1500.0;
      double label3 = 1.5*img_d[jmin][imin]/dmax;
      double label4 = (img_h[jmin][imin] > 918.0);
      double label = label1 + label2 + label3 + label4;

      // if (dmin < dtol) {
      //   fprintf(stderr, "%35s %8.5lf < %8.5lf %8.5lf %8.5lf %8.5lf\n", gaz_names[l], label, label1, label2, label3, label4);
      // }

      if (dmin < dtol && label > 2.25) {
        int k = (jmin*8)/nw;
        int ii = imin + foot + split*((7-k)*(nh+pad)) + 2*CHAR_H*ctiny;
        int jj = jmin + pad - split*(k*(nw/8));
        sprintf(str, "%s %.0lf", gaz_names[l], img_d[jmin][imin]/1000.0);
        split_string(str, 10);
        if (ii < writer.h-head) {
          /* try placing the label as close to the point as possible by going
             round in a spiral */
          int direction = 0; // 0 for u/l, 1 for d/r
          int j = 1;
          while (1) {
            int stop = 0;

            if (direction) {
              for (int i = 0; i < j; i++) {
                ii -= 1;
                if (ii < 0) {
                  continue;
                }
                if (check_text(str, -1, ii, jj, ctiny, image, 1, COLOR_TEXT, COLOR_BACK)) {
                  stop = 1;
                  break;
                }
              } // i end
              if (stop) {
                break;
              }
              for (int i = 0; i < 2*j; i++) {
                jj -= 1;
                if (jj < 0) {
                  continue;
                }
                if (check_text(str, -1, ii, jj, ctiny, image, 1, COLOR_TEXT, COLOR_BACK)) {
                  stop = 1;
                  break;
                }
              } // i end
            } else {
              for (int i = 0; i < j; i++) {
                ii += 1;
                if (ii >= writer.h) {
                  continue;
                }
                if (check_text(str, -1, ii, jj, ctiny, image, 1, COLOR_TEXT, COLOR_BACK)) {
                  stop = 1;
                  break;
                }
              } // i end
              if (stop) {
                break;
              }
              for (int i = 0; i < 2*j-1; i++) {
                jj += 1;
                if (jj >= writer.w) {
                  continue;
                }
                if (check_text(str, -1, ii, jj, ctiny, image, 1, COLOR_TEXT, COLOR_BACK)) {
                  stop = 1;
                  break;
                }
              } // i end
            }

            if (stop) {
              break;
            }
            direction = !direction;
            j++;
          }
          if (ii >= 0 && ii < writer.h-1 && jj >= 0 && jj < writer.w-1) {
            add_text(str, -1, ii+1, jj, ctiny, image, 1, COLOR_TEXT, COLOR_NULL);
          }
        }
      }
    } // l end
  }


  /* ================= */
  /*   WRITE TO FILE   */
  /* ================= */
  img_writer_image(&writer, image);
  img_writer_end(&writer);

  /* clean up */
  free_gazetteer();
  free_2d(edges);
  free_2d(image);
  return 0;
}

/* takes the location data and writes it to the file whose name is the string
   pointed to by path. Returns an error code:
     0 - success
     1 - failed to allocate png struct or info
     2 - failed to open file
     3 - subsequent write failure
     4 - color palette too large
  */
int location_write(const char * restrict path, struct Img_loc *loc, struct Panorama *p) {
  /* extract information */
  int nh = loc->nh;
  int nw = loc->nw;
  double *X = loc->X;
  double *Y = loc->Y;
  double **Z = loc->Z;
  int I0 = loc->i0;
  int J0 = loc->j0;
  double dy = Y[1] - Y[0];
  double dx = X[1] - X[0];

  const int max_zoom = 3;
  char str[128];

  /* ====================== */
  /*   SET UP PNG WRITING   */
  /* ====================== */
  struct img_writer writer;

  /* compute size of output image */
  const int pad = nw/50; // padding between split rows
  int head = nw/12; // space for header
  int foot = nw/25; // space for footer

  /* text sizes */
  int clarge = (int)(head/(2.0*CHAR_H));
  int cmed = (int)(head/(4.0*CHAR_H));
  int csmall = (int)(head/(6.0*CHAR_H));
  int ctiny = (int)(head/(8.0*CHAR_H));
  clarge = (clarge == 0) ? 1 : clarge;
  cmed = (cmed == 0) ? 1 : cmed;
  csmall = (csmall == 0) ? 1 : csmall;
  ctiny = (ctiny == 0) ? 1 : ctiny;

  /* ensure header and footer are large enough to contain text */
  head = (head < clarge*CHAR_H+pad) ? clarge*CHAR_H+pad : head;
  foot = (foot < clarge*CHAR_H+pad) ? clarge*CHAR_H+pad : foot;

  writer.w = max_zoom*nw + max_zoom*pad + pad;
  writer.h = nh + head + foot;

  int err = img_writer_setup(&writer, path);
  if (err) {
    return err;
  }


  /* ========================= */
  /*   CONVERT DATA TO IMAGE   */
  /* ========================= */
  /* image (represented as 8-bit indices into the palette) */
  // TODO: should this use png_malloc?
  png_byte **image = malloc_2d(writer.h, writer.w, sizeof(png_byte));

  /* white background */
  for (int i = 0; i < writer.h; i++) {
    for (int j = 0; j < writer.w; j++) {
      image[i][j] = COLOR_WHITE;
    } // j end
  } // i end


  /* heightmap at increasing zoom */
  int scale = 1;
  for (int zoom = 0; zoom < max_zoom; zoom++) {
    /* origin shift */
    int di = I0-nh/(2*scale);
    int dj = J0-nw/(2*scale);
    if (di > nh-nh/scale) { di = nh-nh/scale; }
    if (di < 0) { di = 0; }
    if (dj > nw-nw/scale) { dj = nw-nw/scale; }
    if (dj < 0) { dj = 0; }

    /* find min and max values */
    double zmin = DBL_MAX;
    double zmax = -DBL_MAX;
    for (int i = 0; i < nh/scale; i++) {
      for (int j = 0; j < nw/scale; j++) {
        double z = Z[di+i][dj+j];
        if (z > zmax) {
          zmax = z;
        }
        if (z < zmin) {
          zmin = z;
        }
      } // j end
    } // i end
    double zrange = zmax-zmin;

    /* height data */
    for (int i = 0; i < nh/scale; i++) {
      for (int j = 0; j < nw/scale; j++) {
        int ii = scale*i + foot;
        int jj = scale*j + pad + zoom*(nw+pad);

        double z = (Z[di+i][dj+j]-zmin)/zrange;
        if (z > 1.0) { z = 1.0; }

        for (int k = 0; k < scale; k++) {
          for (int l = 0; l < scale; l++) {
            image[ii+k][jj+l] = COLORMAP(z);
          } // l end
        } // k end
      } // j end
    } // i end

    /* gridlines */
    for (int i = 0; i < nh/scale; i++) {
      for (int j = 0; j < nw/scale; j++) {
        int ii = scale*i + foot;
        int jj = scale*j + pad + zoom*(nw+pad);

        /* gridlines */
        if (fmod(Y[di+i],1000.0) <= dy/2) {
          for (int k = 0; k < scale; k++) {
            image[ii][jj+k] = COLOR_GREY;
          } // k end

          if (j == 0) {
            sprintf(str, "%.0lf", Y[di+i]/1000.0);
            int w = (int)(ctiny*CHAR_W*strlen(str));
            int h = ctiny*CHAR_H;
            add_text(str, -1, ii+h/2, pad+zoom*(nw+pad), ctiny, image, 0, COLOR_BLACK, COLOR_BACK);
            add_text(str, -1, ii+h/2, pad+zoom*(nw+pad)+nw-w, ctiny, image, 0, COLOR_BLACK, COLOR_BACK);
          }
        }
        if (fmod(X[dj+j],1000.0) <= dx/2) {
          for (int k = 0; k < scale; k++) {
            image[ii+k][jj] = COLOR_GREY;
          } // k end

          if (i == 0) {
            sprintf(str, "%.0lf", X[dj+j]/1000.0);
            int w = (int)(ctiny*CHAR_W*strlen(str));
            int h = ctiny*CHAR_H;
            add_text(str, -1, foot+h, jj-w/2, ctiny, image, 0, COLOR_BLACK, COLOR_BACK);
            add_text(str, -1, foot+nh, jj-w/2, ctiny, image, 0, COLOR_BLACK, COLOR_BACK);
          }
        }
      } // j end
    } // i end

    /* viewpoint location */
    int ii = scale*(I0-di)+(scale/2)+foot;
    int jj = scale*(J0-dj)+(scale/2)+pad+zoom*(nw+pad);
    for (int k = 0; k <= scale; k++) {
      image[ii+k][jj] = COLOR_BLACK;
      image[ii-k][jj] = COLOR_BLACK;
      image[ii][jj+k] = COLOR_BLACK;
      image[ii][jj-k] = COLOR_BLACK;
    } // k end

    int w = ctiny*CHAR_W;
    int h = ctiny*CHAR_H;
    add_text("X", -1, ii+h/2, jj-w/2, ctiny, image, 0, COLOR_BLACK, COLOR_NULL);

    scale *= 2;
  } // zoom end


  /* ============ */
  /*   ADD TEXT   */
  /* ============ */
  char name[64];

  /* top left */
  int i0 = writer.h-pad-1;
  int j0 = pad;

  /* name */
  init_gazetteer(p->source);
  double derr = gazetteer_nearest(p->y0, p->x0, name);

  if (derr > 200.0) {
    /* if the error is too large, mark as unknown */
    strcpy(name, "Unknown location");
    derr = NAN;
  }
  sprintf(str, "%s, ", name);
  j0 = add_text(str, -1, i0, j0, clarge, image, 0, COLOR_BLACK, COLOR_NULL);

  /* grid reference */
  err = 0;
  switch (p->source) {
    case OST50:
      err = ne_to_osng(p->y0, p->x0, str, 8, 1);
      break;
    case SWT02:
      err = ne_to_swgr(p->y0, p->x0, str, 1);
      break;
    case NZT08:
      // TODO: although this is correct it would be more normal to refer to the correct 1:50 map code
      err = ne_to_swgr(p->y0, p->x0, str, 1);
      break;
    default :
      ERROR("data source not recognised");
      break;
  }
  if (err) {
    strcpy(str, "unknown gridref");
  }
  j0 = add_text(str, -1, i0, j0, clarge, image, 0, COLOR_BLACK, COLOR_NULL);
  j0 = add_text(" ", -1, i0, j0, clarge, image, 0, COLOR_BLACK, COLOR_NULL);

  /* other details */
  if (head-pad >= 2*cmed*CHAR_H) {
    sprintf(str, "  altitude: %.0lf m", Z[I0][J0]);
    add_text(str, -1, i0, j0, cmed, image, 0, COLOR_BLACK, COLOR_NULL);
    sprintf(str, "summit distance: %.0lf m", derr);
    add_text(str, -1, i0-(CHAR_H+1)*cmed, j0, cmed, image, 0, COLOR_BLACK, COLOR_NULL);
  } else {
    sprintf(str, "  altitude: %.0lf m, ", Z[I0][J0]);
    j0 = add_text(str, -1, i0, j0, cmed, image, 0, COLOR_BLACK, COLOR_NULL);
    sprintf(str, "summit distance: %.0lf m", derr);
    add_text(str, -1, i0, j0, cmed, image, 0, COLOR_BLACK, COLOR_NULL);
  }


  /* ================= */
  /*   WRITE TO FILE   */
  /* ================= */
  img_writer_image(&writer, image);
  img_writer_end(&writer);

  free_2d(image);
  return 0;
}

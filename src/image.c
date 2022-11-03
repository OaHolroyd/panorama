#include "image.h"

#include <stdlib.h>
#include <math.h>
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


/* ========================================================================== */
/*   AUXILIARY FUNCTION DEFINITIONS                                           */
/* ========================================================================== */
/* adds the characters in the string str to the image, scaled by size. (i0, j0)
   denotes the upper left corner or the upper central point of the string,
   depending if centred is true. Uses fcol (>=0) as the foreground color and
   bcol as the background color (or blank if bcol = 0). Returns the
   x coordinate of the next blank space. */
int add_text(const char *str, int i0, int j0, int size, png_byte **image, int centred, int fcol, int bcol) {
  int len = (int)strlen(str);
  const unsigned char *c;

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
              image[i1+l0][j1+l1] = set;
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
int image_write(const char * restrict path, struct Image *img, struct Panorama *p) {
  /* extract data from container */
  int nw = img->nw;
  int nh = img->nh;
  int *wlim = p->wlim;
  int *hlim = p->hlim;
  double **img_d = img->img_d;
  double **img_h = img->img_h;
  int **img_n = img->img_n;
  double dmax = p->dmax;
  int split = p->split;


  /* ====================== */
  /*   SET UP PNG WRITING   */
  /* ====================== */
  /* allocate space for the image and the write info */
  png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if (!png) {
    return 1;
  }
  png_infop info = png_create_info_struct(png);
  if (!info) {
    png_destroy_write_struct(&png, NULL);
    return 1;
  }

  /* open the file */
  FILE *fp = fopen(path, "wb");
  if (!png) {
    png_destroy_write_struct(&png, &info);
    return 2;
  }

  /* set failure point */
  if (setjmp(png_jmpbuf(png))) {
    png_destroy_write_struct(&png, &info);
    fclose(fp);
    return 3;
  }

  /* set output */
  png_init_io(png, fp);

  /* set filtering options */
  // TODO: see if setting filters is useful
  // png_set_filter(0, /* filters */);

  /* set the zlib compression level */
  // TODO: see if setting compression level is useful
  // png_set_compression_level(png, Z_BEST_COMPRESSION);

  /* compute size of output image */
  const int pad = nh/20; // padding between split rows
  const int head = nh/4; // space for header
  const int foot = nh/4; // space for footer
  int w, h;
  if (split) {
    w = nw/8 + 2*pad;
    h = nh*8 + head + foot + 7*pad;
  } else {
    w = nw + 2*pad;
    h = nh + head + foot;
  }

  /* fill the image header chunk */
  png_set_IHDR(
    png, info,
    w, h, // image dimensions
    8, // color bit-depth
    PNG_COLOR_TYPE_PALETTE, // index into a defined color palette
    PNG_INTERLACE_NONE, // disable interlacing
    PNG_COMPRESSION_TYPE_DEFAULT, // fixed
    PNG_FILTER_TYPE_DEFAULT // fixed
  );

  // TODO: fill tIME

  /* fill text chunk */
  const int num_text = 1;
  png_text text[num_text];
  char key0[] = "Title";
  char text0[] = "Panorama";
  text[0].key = key0;
  text[0].text = text0;
  text[0].compression = PNG_TEXT_COMPRESSION_NONE;
  text[0].itxt_length = 0;
  text[0].lang = NULL;
  text[0].lang_key = NULL;
  png_set_text(png, info, text, num_text);


  // TODO: set gAMA: png_set_gAMA(png_ptr, info_ptr, gamma);

  /* set the color palette (max 256 colors) */
  if (NUM_COLORS > 256) {
    png_destroy_write_struct(&png, &info);
    fclose(fp);
    return 4;
  }
  png_set_PLTE(png, info, palette, NUM_COLORS);

  /* write chunks */
  png_write_info(png, info);


  /* ========================= */
  /*   CONVERT DATA TO IMAGE   */
  /* ========================= */
  /* image (represented as 8-bit indices into the palette) */
  // TODO: should this use png_malloc?
  png_byte **image = malloc_2d(h, w, sizeof(png_byte));

  /* find edges */
  int **edges = malloc_2d(nw, nh, sizeof(int));
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
      } else {
        edges[j][i] = 0;
      }
    } // i end
  } // j end

  /* white background */
  for (int i = 0; i < h; i++) {
    for (int j = 0; j < w; j++) {
      image[i][j] = COLOR_WHITE;
    } // j end
  } // i end

  /* write distance data */
  for (int i = 0; i < nh; i++) {
    for (int j = 0; j < nw; j++) {
      int k = j/(nw/8);
      int ii = i + foot + split*((7-k)*(nh+pad));
      int jj = j + pad - split*(k*(nw/8));
      if (edges[j][i]) {
        image[ii][jj] = COLOR_BLACK;
      } else if (img_d[j][i] < 0.0) {
        image[ii][jj] = COLOR_SKY;
      } else if (img_h[j][i] < MIN_H) {
        image[ii][jj] = COLOR_WATER;
      } else {
        double d = img_d[j][i]/dmax;
        if (d > 1.0) { d = 1.0; }
        image[ii][jj] = COLORMAP(d);
      }
    } // j end
  } // i end


  /* ============ */
  /*   ADD TEXT   */
  /* ============ */
  char name[64];
  char str[128];

  /* sizes */
  int clarge = head/(1.2*CHAR_H);
  int cmed = head/(2.4*CHAR_H);
  int csmall = head/(4.8*CHAR_H);

  /* top left */
  int i0 = h-pad-1;
  int j0 = pad;

  /* name */
  init_gazetteer(p->source);
  double derr = gazetteer_nearest(p->y0, p->x0, name);
  free_gazetteer();

  if (derr > 200.0) {
    /* if the error is too large, mark as unknown */
    strcpy(name, "Unknown location");
  }
  add_text("view ", i0, j0, cmed, image, 0, COLOR_BLACK, COLOR_NULL);
  j0 = add_text("from ", i0-(CHAR_H+1)*cmed, j0, cmed, image, 0, COLOR_BLACK, COLOR_NULL);
  sprintf(str, "%s, ", name);
  j0 = add_text(str, i0, j0, clarge, image, 0, COLOR_BLACK, COLOR_NULL);

  /* grid reference */
  int err = ne_to_osng(p->y0, p->x0, str, 8, 1);
  if (err) {
    strcpy(str, "unknown gridref");
  }
  j0 = add_text(str, i0, j0, clarge, image, 0, COLOR_BLACK, COLOR_NULL);
  j0 = add_text(" ", i0, j0, clarge, image, 0, COLOR_BLACK, COLOR_NULL);

  /* other details */
  sprintf(str, "  altitude: %.0lf m", p->z0);
  add_text(str, i0, j0, cmed, image, 0, COLOR_BLACK, COLOR_NULL);
  sprintf(str, "eye height: %.0lf m", img->z);
  add_text(str, i0-(CHAR_H+1)*cmed, j0, cmed, image, 0, COLOR_BLACK, COLOR_NULL);

  /* bottom left */
  i0 = foot-pad;
  j0 = pad;

  /* attributions */
  switch (p->source) {
    case OST50:
      sprintf(str, "generated from OS Terrain 50 data (Crown copyright)");
      break;
    case SWISSALTI3D:
      ERROR("SWISSALTI3D not supported");
      break;
    default :
      ERROR("data source not recognised");
      break;
  }
  add_text(str, i0, j0, csmall, image, 0, COLOR_BLACK, COLOR_NULL);


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
        }
      } // j end
    } // i end
  }

  /* cardinal points and bearings */
  for (int j = 0; j < nw; j++) {
    int k = j/(nw/8);
    int ii = nh-2 + foot + split*((7-k)*(nh+pad));
    int jj = j + pad - split*(k*(nw/8));

    /* label 8 main compass directions and then every dd degrees */
    const double dd = 10.0;
    double d = wlim[0]+(double)j/p->res;
    if (fabs(d-0.0) < 0.5/p->res) {
      add_text("NORTH", ii, jj, csmall, image, 1, COLOR_RED, COLOR_WHITE);
    } else if (fabs(d-180.0) < 0.5/p->res) {
      add_text("SOUTH", ii, jj, csmall, image, 1, COLOR_RED, COLOR_WHITE);
    } else if (fabs(d-90.0) < 0.5/p->res) {
      add_text("EAST", ii, jj, csmall, image, 1, COLOR_RED, COLOR_WHITE);
    } else if (fabs(d-270.0) < 0.5/p->res) {
      add_text("WEST", ii, jj, csmall, image, 1, COLOR_RED, COLOR_WHITE);
    } else if (fabs(d-45.0) < 0.5/p->res) {
      add_text("NE", ii, jj, csmall, image, 1, COLOR_RED, COLOR_WHITE);
    } else if (fabs(d-135.0) < 0.5/p->res) {
      add_text("SE", ii, jj, csmall, image, 1, COLOR_RED, COLOR_WHITE);
    } else if (fabs(d-225.0) < 0.5/p->res) {
      add_text("SW", ii, jj, csmall, image, 1, COLOR_RED, COLOR_WHITE);
    } else if (fabs(d-315.0) < 0.5/p->res) {
      add_text("NW", ii, jj, csmall, image, 1, COLOR_RED, COLOR_WHITE);
    } else if (fabs(fmodf(d, dd)) < 0.5/p->res) {
      sprintf(str, "%03.0lf", fmodf(dd*rint(d/dd)+360.0, 360.0));
      add_text(str, ii, jj, csmall, image, 1, COLOR_RED, COLOR_WHITE);
    }
  } // j end


  /* ================= */
  /*   WRITE TO FILE   */
  /* ================= */
  /* set up rows of pixels */
  // TODO: should this use png_malloc?
  png_bytep *rows = malloc(h*sizeof(png_bytep));
  for (int i = 0; i < h; i++) {
    rows[i] = image[h-i-1];
  } // i end

  /* write the image data */
  png_write_image(png, rows);

  /* finish writing */
  png_write_end(png, info);


  /* clean up */
  free_2d(edges);
  free_2d(image);
  free(rows);
  png_destroy_write_struct(&png, &info);
  fclose(fp);
  return 0;
}

#include "image.h"

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <png.h>
// #include <zlib.h>

#include "data.h"
#include "colors.h"
#include "font.h"
#include "utils.h"

/* lower bound for feasible height (lowest point visible is -430m) */
#define MIN_H (-500.0)


/* ========================================================================== */
/*   AUXILIARY FUNCTION DEFINITIONS                                           */
/* ========================================================================== */
/* adds the characters in the string str to the image, scaled by size. (i0, j0)
   denotes the upper left corner of the string. Returns the x coordinate of the
   next blank space. */
int add_text(const char *str, int i0, int j0, int size, png_byte **image) {
  int len = (int)strlen(str);
  const unsigned char *c;

  int j1 = j0;
  for (int k = 0; k < len; k++) {
    int i1 = i0;
    c = CHAR_8BIT(str[k]);

    /* print character */
    for (int i = CHAR_H-1; i >= 0; i--) {
      j1 = j0 + k*CHAR_W*size;
      for (int j = CHAR_W-1; j >= 0; j--) {
        png_byte set = (c[i] & 1 << j) ? COLOR_BLACK : COLOR_WHITE;

        /* scale pixels by size */
        for (int l0 = 0; l0 < size; l0++) {
          for (int l1 = 0; l1 < size; l1++) {
            image[i1+l0][j1+l1] = set;
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
  if (split) {
    for (int i = 0; i < nh; i++) {
      for (int j = 0; j < nw; j++) {
        int k = j/(nw/8);
        int ii = i + foot + k*(nh+pad);
        int jj = j + pad - k*(nw/8);
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
  } else {
    for (int i = 0; i < nh; i++) {
      for (int j = 0; j < nw; j++) {
        int ii = i + foot;
        int jj = j + pad;
        if (img_d[j][i] < 0.0) {
          image[ii][jj] = COLOR_SKY;
        } else if (img_h[j][i] < 0.0) {
          image[ii][jj] = COLOR_WATER;
        } else {
          double d = img_d[j][i]/dmax;
          if (d > 1.0) { d = 1.0; }
          image[ii][jj] = COLORMAP(d);
        }
      } // j end
    } // i end
  }


  /* ============ */
  /*   ADD TEXT   */
  /* ============ */
  char str[128];

  /* sizes */
  int clarge = head/(2*CHAR_H);
  int cmed = head/(4*CHAR_H);
  int csmall = head/(8*CHAR_H);

  /* top left */
  int i0 = h-pad-1;
  int j0 = pad+1;

  /* name */
  // TODO: find with gazetteer
  strcpy(str, "Unknown: ");
  j0 = add_text(str, i0, j0, clarge, image);

  /* grid reference */
  int err = ne_to_osng(p->y0, p->x0, str, 8, 1);
  if (err) {
    strcpy(str, "unknown gridref");
  }
  j0 = add_text(str, i0, j0, clarge, image);
  j0 = add_text(" ", i0, j0, clarge, image);

  /* other details */
  sprintf(str, "terrain height: %.0lf m", p->z0);
  add_text(str, i0, j0, cmed, image);
  sprintf(str, "    eye height: %.0lf m", img->z);
  add_text(str, i0-(CHAR_H+1)*cmed, j0, cmed, image);



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

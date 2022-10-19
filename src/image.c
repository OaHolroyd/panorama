#include "image.h"

#include <stdlib.h>
#include <png.h>
// #include <zlib.h>

#include "colors.h"
#include "utils.h"


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
int image_write(const char * restrict path, struct Image *img) {
  /* extract data from container */
  int nw = img->nw;
  int nh = img->nh;
  double **img_x = img->img_x;
  double **img_y = img->img_y;
  double **img_d = img->img_d;
  double **img_h = img->img_h;
  int **img_n = img->img_n;
  double dmax = img->dmax;


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

  /* fill the image header chunk */
  png_set_IHDR(
    png, info,
    nw, nh, // image dimensions
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
  png_byte **image = malloc_2d(nh, nw, sizeof(png_byte));

  // TODO: write this properly
  for (int i = 0; i < nh; i++) {
    for (int j = 0; j < nw; j++) {
      if (img_d[j][i] < 0.0) {
        image[i][j] = COLOR_WHITE;
      } else {
        double d = img_d[j][i]/dmax;
        if (d > 1.0) { d = 1.0; }
        image[i][j] = COLORMAP(d);
      }
    } // j end
  } // i end


  /* ================= */
  /*   WRITE TO FILE   */
  /* ================= */
  /* set up rows of pixels */
  // TODO: should this use png_malloc?
  png_bytep *rows = malloc(nh*sizeof(png_bytep));
  for (int i = 0; i < nh; i++) {
    rows[i] = image[nh-i-1];
  } // i end

  /* write the image data */
  png_write_image(png, rows);

  /* finish writing */
  png_write_end(png, info);


  /* clean up */
  free_2d(image);
  free(rows);
  png_destroy_write_struct(&png, &info);
  fclose(fp);
  return 0;
}

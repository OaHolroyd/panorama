#include "gazetteer.h"

#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <float.h>
#include <math.h>
#include <string.h>

#include "utils.h"


#define GAZ_NAME_LEN 35


/* ========================================================================== */
/*   GAZETTEER VARIABLES                                                      */
/* ========================================================================== */
int gaz_n; // number of entries
char **gaz_names; // names
int32_t *gaz_y; // y coord
int32_t *gaz_x; // x coord


/* ========================================================================== */
/*   FUNCTION DEFINITIONS                                                     */
/* ========================================================================== */
/* read the gazetteer appropriate for the data source */
void init_gazetteer(data_source source) {
  /* get the filename */
  char file[128];
  switch (source) {
    case OST50:
      sprintf(file, "./data/gazetteers/dobih.gaz");
      break;
    case SWISSALTI3D:
      ERROR("SWISSALTI3D not supported");
      break;
    default :
      ERROR("data source not recognised");
      break;
  }

  /* use the fact that the names are 35 char strings and the x and y coords are
     int16s to convert the length of the file to the number of entries */
  FILE *fp = fopen(file, "rb");
  fseek(fp, 0L, SEEK_END);
  long int fsize = ftell(fp);
  rewind(fp);

  long int size = 35*sizeof(char) + 2*sizeof(int32_t); // size of data per entry
  gaz_n = (int)(fsize/size);

  /* allocate the required memory */
  gaz_names = malloc_2d(gaz_n, 35, sizeof(char));
  gaz_x = malloc(gaz_n*sizeof(int32_t));
  gaz_y = malloc(gaz_n*sizeof(int32_t));

  /* read from the file */
  fread(gaz_names[0], sizeof(char), 35*gaz_n, fp);
  fread(gaz_y, sizeof(int32_t), gaz_n, fp);
  fread(gaz_x, sizeof(int32_t), gaz_n, fp);
  fclose(fp);
}

/* frees all the gazetteer information */
void free_gazetteer(void) {
  free_2d(gaz_names);
  free(gaz_x);
  free(gaz_y);
}

/* finds the closest entry in the gazetteer to a point (py, px), filling its
   name into name. Returns the distance to to the entry. Name should have space
   for up to 35 chars. */
double gazetteer_nearest(int py, int px, char *name) {
  double dmin = DBL_MAX;
  int imin = -1;
  for (int i = 0; i < gaz_n; i++) {
    double dy = py-gaz_y[i];
    double dx = px-gaz_x[i];

    double d = dx*dx + dy*dy;
    if (d < dmin) {
      imin = i;
      dmin = d;
    }
  } // i end

  strcpy(name, gaz_names[imin]);

  return sqrt(dmin);
}

#ifndef GAZETEER_H
#define GAZETEER_H

#include "data.h"


/* ========================================================================== */
/*   FUNCTION DECLARATIONS                                                    */
/* ========================================================================== */
/* read the gazetteer appropriate for the data source */
void init_gazetteer(data_source source);

/* frees all the gazetteer information */
void free_gazetteer(void);

/* finds the closest entry in the gazetteer to a point (py, px), filling its
   name into name. Returns the distance to to the entry. Name should have space
   for up to 35 chars. */
double gazetteer_nearest(int py, int px, char *name);

#endif

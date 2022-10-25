#ifndef DATA_H
#define DATA_H


/* ========================================================================== */
/*  DATA TYPES                                                                */
/* ========================================================================== */
/* where the base data comes from */
typedef enum data_source {
  OST50, // OS Terrain 50
  SWISSALTI3D, // swissALTI3D
} data_source;


/* ========================================================================== */
/*   FUNCTION DECLARATIONS                                                    */
/* ========================================================================== */
/* sets ny and nx to the number of columns and rows in a block from the given
   data source */
void get_block_dims(data_source source, int *ny, int *nx);

/* allocates *Bx and *By (with length nbx, nby respectively) and fills them
   with the x and y range of the data source */
void get_data_extent(data_source source, int *nbx, int *nby, int **Bx, int **By);

/* get the block coords containing the point (x0, y0) */
void block_index(data_source source, double y0, int *I0, double x0, int *J0);

/* fills X, Y, and Z with the eastings, northings, and heights from the data
   block (I0, J0) from the given data source. If the file containing the data
   is not found, it defaults to returning a height of zero for the entire block.
   Returns an integer error code:
      0 on success
     -1 failed to read correct number of entries
   The data must be saved in row-major format from the lower-left corner to the
   upper-right corner */
int get_block(int I0, int J0, data_source source, double **X, double **Y, double **Z);

/* given an ny-by-nx base grid Zb, allocates storage for the multigrid
   specified by levels and contstructs the grid ***Z. Note that the top level of
   the grid uses the memory allocated to Zb */
void create_multigrid(int ny, int nx, int nlevels, int *levels, double **Zb, double ****pZ);

/* frees all of the memory allocated by create_multigrid (no including the
   zeroth layer). */
void free_multigrid(int nlevels, double ***Z);

/* fills the multigrid with coarsened values going up the levels (see mipmap).
   Note that this assumes that Z[0] = Zb */
void fill_multigrid(int ny, int nx, int nlevels, int *levels, double ***Z);

/* checks if the level structure is compatable with the grid structure: at each
   refinement the coarsening factor (in levels) must divide the grid dimensions.
   Returns 1 if valid, 0 otherwise */
int validate_levels(int ny, int nx, int nlevels, int *levels);

/* converts an OS National Grid coordinate string gridref (which must begin with
   a valid two-letter grid identifier, followed by a 6, 8, or 10 figure numeric
   grid reference) to a Northing y and Easting x. Returns an error code:
      0 on success
      1 if invalid length
      2 if invalid leading character code
      3 if invalid grid reference */
int osng_to_ne(char *gridref, double *y, double *x);


#endif

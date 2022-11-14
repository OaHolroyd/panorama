#include "utils.h"

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>


/* aborts with an error message */
__attribute__((__format__ (__printf__, 1, 0)))
void ERROR(const char *format, ...) {
  va_list args;
  va_start(args, format);

  fprintf(stderr, "%s:%d: ERROR: ", __FILE__, __LINE__);
  vfprintf(stderr, format, args);
  fprintf(stderr, "\n");

  va_end(args);

  exit(1);
}

/* prints a debugging message */
#if DEBUG
__attribute__((__format__ (__printf__, 1, 0)))
void debug(const char *format, ...) {
  va_list args;
  va_start(args, format);

  fprintf(stderr, "DEBUG: ");
  vfprintf(stderr, format, args);
  fprintf(stderr, "\n");

  va_end(args);
}
#else
void debug(const char *format, ...) { UNUSED(format); }
#endif

/* Allocate memory for a 2D array of arbitrary type with a given size, matching
   row indices to the corresponding memory locations. */
void* malloc_2d(size_t ni, size_t nj, size_t size) {
  /* allocate row memory */
  void **p_2arr = malloc(ni*sizeof(void*));
  if (!p_2arr) { return NULL; }

  /* allocate main memory */
  char *mem = malloc(ni*nj*size);
  if (!mem) { free(p_2arr); return NULL; }

  /* match rows to memory */
  for (size_t i = 0; i < ni; i++) {
    p_2arr[i] = &(mem[i*nj*size]);
  } // i end

  return p_2arr;
}

/* Frees memory associated with a 2D array */
void internal_free_2d(void** arr) {
  if (arr) { free(*arr); }
  free(arr);
}

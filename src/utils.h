#ifndef UTILS_H
#define UTILS_H

#include <stddef.h>

#ifdef DEBUG
  #undef DEBUG
  #define DEBUG 1
#else
  #define DEBUG 0
#endif

/* marks a parameter as unused (to prevent triggering Wunused-parameter) */
#define UNUSED(x) do { (void)(x); } while (0)

/* aborts with an error message */
void ERROR(const char *format, ...);

/* prints a debugging message */
void debug(const char *format, ...);

/* Allocate memory for a 2D array of arbitrary type with a given size, matching
   row indices to the corresponding memory locations. */
void* malloc_2d(int ni, int nj, size_t size);

/* Frees memory associated with a 2D array */
#define free_2d(A) internal_free_2d((void **)A)
void internal_free_2d(void** arr);


#endif

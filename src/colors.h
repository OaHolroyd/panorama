#ifndef COLORS_H
#define COLORS_H

#include <png.h>

/* number of colors */
#define NUM_COLORS ((png_byte)(sizeof(palette)/sizeof(png_color)))
#define NUM_INDIVIDUAL ((png_byte)(sizeof(individual_colors)/sizeof(png_color)))
#define NUM_COLORMAP ((png_byte)(sizeof(colormap_colors)/sizeof(png_color)))

/* convert x in [0, 1] to a colormap index */
#define COLORMAP(x) (NUM_INDIVIDUAL + ((png_byte)(x*((double)NUM_COLORMAP-1.0))))

/* individual color indices */
#define COLOR_NULL 0
#define COLOR_BLACK 1
#define COLOR_WHITE 2
#define COLOR_GREY 3
#define COLOR_WATER 4
#define COLOR_SKY 5
#define COLOR_RED 6

/* individual colors */
#define INDIVIDUAL_COLORS \
  {  0,   0,   0}, /* "null" */ \
  {  0,   0,   0}, /* black */ \
  {255, 255, 255}, /* white */ \
  {128, 128, 128}, /* grey */ \
  { 51, 153, 255}, /* water */ \
  {175, 225, 255}, /* sky */ \
  {255,   0,   0}  /* red */

/* colormap colors */
#define COLORMAP_COLORS \
  { 34, 119,  55}, \
  { 32, 125,  51}, \
  { 29, 131,  48}, \
  { 27, 137,  44}, \
  { 24, 142,  40}, \
  { 21, 148,  35}, \
  { 19, 154,  31}, \
  { 16, 160,  27}, \
  { 14, 166,  24}, \
  { 11, 172,  20}, \
  {  9, 178,  16}, \
  { 14, 181,  15}, \
  { 22, 183,  14}, \
  { 30, 185,  13}, \
  { 38, 188,  13}, \
  { 46, 191,  13}, \
  { 54, 193,  13}, \
  { 62, 196,  12}, \
  { 71, 198,  11}, \
  { 79, 201,  11}, \
  { 86, 203,  10}, \
  { 94, 206,  10}, \
  {102, 208,   9}, \
  {111, 211,   9}, \
  {119, 213,   8}, \
  {127, 216,   8}, \
  {135, 218,   8}, \
  {143, 220,   7}, \
  {151, 223,   7}, \
  {159, 225,   6}, \
  {167, 228,   5}, \
  {175, 231,   5}, \
  {183, 233,   4}, \
  {191, 235,   4}, \
  {199, 238,   3}, \
  {207, 240,   3}, \
  {215, 242,   3}, \
  {223, 245,   2}, \
  {231, 248,   2}, \
  {239, 250,   1}, \
  {247, 253,   1}, \
  {255, 255,   0}, \
  {255, 244,   0}, \
  {255, 234,   0}, \
  {255, 223,   0}, \
  {255, 212,   0}, \
  {255, 201,   0}, \
  {255, 190,   0}, \
  {255, 180,   0}, \
  {255, 169,   0}, \
  {255, 159,   0}, \
  {255, 148,   0}, \
  {255, 138,   0}, \
  {255, 127,   0}, \
  {255, 116,   0}, \
  {255, 105,   0}, \
  {255,  94,   0}, \
  {255,  84,   0}, \
  {255,  73,   0}, \
  {255,  63,   0}, \
  {255,  52,   0}, \
  {255,  41,   0}, \
  {255,  30,   0}, \
  {255,  20,   0}, \
  {255,   9,   0}, \
  {255,   1,   2}, \
  {254,   5,  13}, \
  {254,   9,  25}, \
  {253,  12,  37}, \
  {252,  16,  48}, \
  {251,  20,  60}, \
  {250,  24,  72}, \
  {250,  28,  84}, \
  {249,  32,  95}, \
  {249,  35, 107}, \
  {248,  40, 119}, \
  {247,  44, 131}, \
  {246,  48, 142}, \
  {245,  51, 154}, \
  {245,  55, 166}, \
  {244,  59, 177}, \
  {244,  63, 189}, \
  {243,  67, 201}, \
  {242,  71, 213}, \
  {241,  74, 224}, \
  {240,  79, 236}, \
  {241,  88, 241}, \
  {242, 101, 242}, \
  {243, 114, 243}, \
  {244, 127, 244}, \
  {245, 140, 245}, \
  {247, 153, 247}, \
  {248, 166, 248}, \
  {248, 178, 248}, \
  {249, 191, 249}, \
  {250, 203, 250}, \
  {251, 216, 251}, \
  {253, 229, 253}, \
  {254, 242, 254}, \
  {255, 255, 255}

/* full color palette */
const png_color palette[] = {
  INDIVIDUAL_COLORS,
  COLORMAP_COLORS,
};

/* partial color palettes */
const png_color individual_colors[] = { INDIVIDUAL_COLORS };
const png_color colormap_colors[] = { COLORMAP_COLORS };


#endif

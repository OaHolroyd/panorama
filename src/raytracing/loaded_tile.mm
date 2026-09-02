#include "loaded_tile.h"

#include "metal_tile.h"

#include <stdexcept>

namespace panorama {

LoadedTile LoadedTile::load(const std::filesystem::path &path) {
  if (!is_metal_tile_path(path)) {
    throw std::invalid_argument(
        "Terrain tracing requires a prepared .ptile file: " + path.string()
    );
  }

  const MetalTileHeader header = read_metal_tile_header(path);
  return {
      Crs::from_epsg(header.epsg_code),
      header.maximum_elevation,
      header.cell_count,
      header.lower_left_x,
      header.lower_left_y,
      header.cell_size,
      header.level_count,
  };
}

} // namespace panorama

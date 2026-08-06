#pragma once

#include <cstdint>

namespace panorama {

// Geographic coordinates use WGS 84 degrees. Keeping the field names explicit
// prevents accidentally supplying the common (longitude, latitude) order.
struct LatLon {
  double lat;
  double lon;
};

// Projected coordinates are metres east and north in the CRS represented by a
// Crs instance: x is easting and y is northing.
struct Coord {
  double x;
  double y;
};

// The supported projected coordinate reference systems for terrain tiles.
enum class CrsId : uint32_t {
  SwissLv95 = 2056,
  FrenchLambert93 = 2154,
  BritishNationalGrid = 27700,
};

// A small, deliberately restricted CRS value type. Transformations are
// performed through GDAL/PROJ, rather than approximate hand-written formulas.
class Crs {
public:
  /// Construct one of the explicitly supported projected coordinate systems.
  explicit Crs(CrsId id);

  /// Return the supported CRS identified by `epsg_code`, or throw if unknown.
  [[nodiscard]] static Crs from_epsg(uint32_t epsg_code);

  /// Return this CRS's strongly typed identifier.
  [[nodiscard]] CrsId id() const;

  /// Return this CRS's EPSG authority code.
  [[nodiscard]] uint32_t epsg_code() const;

  /// Return a stable human-readable name for this CRS.
  [[nodiscard]] const char *name() const;

  /// Transform WGS 84 latitude/longitude degrees into projected metres.
  [[nodiscard]] Coord from_lat_lon(LatLon coordinate) const;

  /// Transform projected metres in this CRS into WGS 84 latitude/longitude
  /// degrees.
  [[nodiscard]] LatLon to_lat_lon(Coord coordinate) const;

private:
  CrsId id_;
};

} // namespace panorama

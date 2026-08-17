#pragma once

#include <cstdint>

namespace panorama {

/// Geographic WGS 84 coordinates in the explicit `(latitude, longitude)` order.
struct LatLon {
  double lat;
  double lon;
};

/// Projected metre coordinates where x is easting and y is northing.
struct Coord {
  double x;
  double y;
};

/// Projected coordinate reference systems supported by terrain tiles.
enum class CrsId : uint32_t {
  SwissLv95 = 2056,
  FrenchLambert93 = 2154,
  BritishNationalGrid = 27700,
};

/// Restricted terrain CRS value type whose transforms are delegated to GDAL/PROJ.
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

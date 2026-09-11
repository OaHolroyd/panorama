#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

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

/// Reusable GDAL/PROJ transformation for catalogue-sized coordinate batches.
class CoordinateTransform {
public:
  CoordinateTransform(uint32_t source_epsg, uint32_t destination_epsg);
  CoordinateTransform(uint32_t source_epsg, LatLon local_aeqd_anchor);
  CoordinateTransform(LatLon local_aeqd_anchor, uint32_t destination_epsg);
  ~CoordinateTransform();
  CoordinateTransform(CoordinateTransform &&) noexcept;
  CoordinateTransform &operator=(CoordinateTransform &&) noexcept;
  CoordinateTransform(const CoordinateTransform &) = delete;
  CoordinateTransform &operator=(const CoordinateTransform &) = delete;

  [[nodiscard]] std::vector<Coord> apply(std::span<const Coord> coordinates) const;

private:
  struct State;
  std::unique_ptr<State> state_;
};

/// Transform conventional GIS `(x, y)` coordinates between arbitrary EPSG
/// systems. Geographic coordinates therefore use `(longitude, latitude)`.
[[nodiscard]] std::vector<Coord> transform_coordinates(
    uint32_t source_epsg,
    uint32_t destination_epsg,
    std::span<const Coord> coordinates
);

/// Transform to and from an azimuthal-equidistant metre frame centred on a
/// WGS 84 anchor. Local x is east and local y is north at the anchor.
[[nodiscard]] std::vector<Coord> transform_coordinates_to_local_aeqd(
    uint32_t source_epsg,
    LatLon anchor,
    std::span<const Coord> coordinates
);
[[nodiscard]] std::vector<Coord> transform_coordinates_from_local_aeqd(
    uint32_t destination_epsg,
    LatLon anchor,
    std::span<const Coord> coordinates
);

/// Return whether an EPSG CRS is projected with metre horizontal units.
[[nodiscard]] bool epsg_uses_projected_metres(uint32_t epsg_code);

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

  /// Transform a batch while sharing one GDAL/PROJ transformation.
  [[nodiscard]] std::vector<Coord> from_lat_lon(std::span<const LatLon> coordinates) const;

  /// Transform projected metres in this CRS into WGS 84 latitude/longitude
  /// degrees.
  [[nodiscard]] LatLon to_lat_lon(Coord coordinate) const;

private:
  CrsId id_;
};

} // namespace panorama

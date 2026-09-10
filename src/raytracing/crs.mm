#include "crs.h"

#include <ogr_spatialref.h>

#include <memory>
#include <stdexcept>
#include <string>

namespace panorama {
namespace {

constexpr uint32_t kWgs84Epsg = 4326;

/// Build a GDAL spatial reference with this project's explicit GIS axis order.
[[nodiscard]] OGRSpatialReference spatial_reference(uint32_t epsg_code) {
  OGRSpatialReference reference;
  if (reference.importFromEPSG(static_cast<int>(epsg_code)) != OGRERR_NONE) {
    throw std::runtime_error("Could not initialise EPSG:" + std::to_string(epsg_code));
  }

  // GDAL 3 honours authority-defined axis order by default. This application
  // uses the conventional GIS order: (longitude, latitude) and (easting,
  // northing), so make it explicit at the GDAL boundary.
  reference.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
  return reference;
}

/// unique_ptr deleter for an OGR coordinate transformation allocated by GDAL.
struct CoordinateTransformationDeleter {
  /// Release the transformation allocated by OGRCreateCoordinateTransformation.
  void operator()(OGRCoordinateTransformation *transformation) const {
    OCTDestroyCoordinateTransformation(transformation);
  }
};

/// Create an owning source-to-destination coordinate transformation.
[[nodiscard]] std::unique_ptr<OGRCoordinateTransformation, CoordinateTransformationDeleter>
make_transformation(uint32_t source_epsg, uint32_t destination_epsg) {
  OGRSpatialReference source = spatial_reference(source_epsg);
  OGRSpatialReference destination = spatial_reference(destination_epsg);
  OGRCoordinateTransformation *raw = OGRCreateCoordinateTransformation(&source, &destination);
  if (raw == nullptr) {
    throw std::runtime_error(
        "Could not transform from EPSG:" + std::to_string(source_epsg) +
        " to EPSG:" + std::to_string(destination_epsg)
    );
  }
  return std::unique_ptr<OGRCoordinateTransformation, CoordinateTransformationDeleter>(raw);
}

} // namespace

// Crs is intentionally a tiny value type. The potentially expensive GDAL
// transformation is created only for the conversion currently being requested.
Crs::Crs(CrsId id) : id_(id) {}

Crs Crs::from_epsg(uint32_t epsg_code) {
  switch (epsg_code) {
  case static_cast<uint32_t>(CrsId::SwissLv95):
    return Crs(CrsId::SwissLv95);
  case static_cast<uint32_t>(CrsId::FrenchLambert93):
    return Crs(CrsId::FrenchLambert93);
  case static_cast<uint32_t>(CrsId::BritishNationalGrid):
    return Crs(CrsId::BritishNationalGrid);
  default:
    throw std::invalid_argument("Unsupported terrain CRS EPSG:" + std::to_string(epsg_code));
  }
}

CrsId Crs::id() const { return id_; }

uint32_t Crs::epsg_code() const { return static_cast<uint32_t>(id_); }

const char *Crs::name() const {
  switch (id_) {
  case CrsId::SwissLv95:
    return "Swiss LV95";
  case CrsId::FrenchLambert93:
    return "French Lambert-93";
  case CrsId::BritishNationalGrid:
    return "British National Grid";
  }
  throw std::logic_error("Unknown CRS identifier");
}

Coord Crs::from_lat_lon(LatLon coordinate) const {
  // GDAL receives conventional GIS axis order because spatial_reference()
  // selected OAMS_TRADITIONAL_GIS_ORDER: x is longitude, y is latitude.
  auto transformation = make_transformation(kWgs84Epsg, epsg_code());
  double longitude = coordinate.lon;
  double latitude = coordinate.lat;
  if (!transformation->Transform(1, &longitude, &latitude)) {
    throw std::runtime_error("Could not transform WGS 84 coordinate to " + std::string(name()));
  }
  return {longitude, latitude};
}

std::vector<Coord> Crs::from_lat_lon(std::span<const LatLon> coordinates) const {
  if (coordinates.empty())
    return {};
  auto transformation = make_transformation(kWgs84Epsg, epsg_code());
  std::vector<double> longitudes;
  std::vector<double> latitudes;
  longitudes.reserve(coordinates.size());
  latitudes.reserve(coordinates.size());
  for (const LatLon coordinate : coordinates) {
    longitudes.push_back(coordinate.lon);
    latitudes.push_back(coordinate.lat);
  }
  if (!transformation
           ->Transform(coordinates.size(), longitudes.data(), latitudes.data(), nullptr, nullptr)) {
    throw std::runtime_error("Could not transform WGS 84 coordinates to " + std::string(name()));
  }
  std::vector<Coord> result;
  result.reserve(coordinates.size());
  for (size_t index = 0; index < coordinates.size(); ++index)
    result.push_back({longitudes[index], latitudes[index]});
  return result;
}

LatLon Crs::to_lat_lon(Coord coordinate) const {
  // Transform mutates its coordinate arguments in place. Begin with projected
  // easting/northing, then reinterpret the resulting x/y as longitude/latitude.
  auto transformation = make_transformation(epsg_code(), kWgs84Epsg);
  double easting = coordinate.x;
  double northing = coordinate.y;
  if (!transformation->Transform(1, &easting, &northing)) {
    throw std::runtime_error(
        "Could not transform " + std::string(name()) + " coordinate to WGS 84"
    );
  }
  return {northing, easting};
}

} // namespace panorama

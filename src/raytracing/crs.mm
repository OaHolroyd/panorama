#include "crs.h"

#include <ogr_spatialref.h>

#include <cmath>
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

[[nodiscard]] OGRSpatialReference local_aeqd_reference(LatLon anchor) {
  if (!std::isfinite(anchor.lat) || !std::isfinite(anchor.lon) || anchor.lat < -90.0 ||
      anchor.lat > 90.0)
    throw std::invalid_argument("Azimuthal-equidistant anchor is invalid");
  OGRSpatialReference reference;
  if (reference.SetWellKnownGeogCS("WGS84") != OGRERR_NONE ||
      reference.SetAE(anchor.lat, anchor.lon, 0.0, 0.0) != OGRERR_NONE ||
      reference.SetLinearUnits("metre", 1.0) != OGRERR_NONE)
    throw std::runtime_error("Could not initialise local azimuthal-equidistant CRS");
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

[[nodiscard]] std::unique_ptr<OGRCoordinateTransformation, CoordinateTransformationDeleter>
make_transformation(OGRSpatialReference source, OGRSpatialReference destination) {
  OGRCoordinateTransformation *raw = OGRCreateCoordinateTransformation(&source, &destination);
  if (raw == nullptr) {
    throw std::runtime_error("Could not create coordinate transformation");
  }
  return std::unique_ptr<OGRCoordinateTransformation, CoordinateTransformationDeleter>(raw);
}

[[nodiscard]] std::vector<Coord> transform(
    OGRCoordinateTransformation &transformation,
    std::span<const Coord> coordinates,
    const char *failure
) {
  if (coordinates.empty())
    return {};
  std::vector<double> xs, ys;
  xs.reserve(coordinates.size());
  ys.reserve(coordinates.size());
  for (const Coord coordinate : coordinates) {
    xs.push_back(coordinate.x);
    ys.push_back(coordinate.y);
  }
  if (!transformation.Transform(coordinates.size(), xs.data(), ys.data(), nullptr, nullptr))
    throw std::runtime_error(failure);
  std::vector<Coord> result;
  result.reserve(coordinates.size());
  for (size_t index = 0; index < coordinates.size(); ++index)
    result.push_back({xs[index], ys[index]});
  return result;
}

} // namespace

struct CoordinateTransform::State {
  std::unique_ptr<OGRCoordinateTransformation, CoordinateTransformationDeleter> transformation;
};

CoordinateTransform::CoordinateTransform(uint32_t source_epsg, uint32_t destination_epsg)
    : state_(std::make_unique<State>()) {
  state_->transformation =
      make_transformation(spatial_reference(source_epsg), spatial_reference(destination_epsg));
}

CoordinateTransform::CoordinateTransform(uint32_t source_epsg, LatLon local_aeqd_anchor)
    : state_(std::make_unique<State>()) {
  state_->transformation =
      make_transformation(spatial_reference(source_epsg), local_aeqd_reference(local_aeqd_anchor));
}

CoordinateTransform::CoordinateTransform(LatLon local_aeqd_anchor, uint32_t destination_epsg)
    : state_(std::make_unique<State>()) {
  state_->transformation = make_transformation(
      local_aeqd_reference(local_aeqd_anchor),
      spatial_reference(destination_epsg)
  );
}

CoordinateTransform::~CoordinateTransform() = default;
CoordinateTransform::CoordinateTransform(CoordinateTransform &&) noexcept = default;
CoordinateTransform &CoordinateTransform::operator=(CoordinateTransform &&) noexcept = default;

std::vector<Coord> CoordinateTransform::apply(std::span<const Coord> coordinates) const {
  if (state_ == nullptr || state_->transformation == nullptr)
    throw std::logic_error("Coordinate transform has been moved from");
  return transform(*state_->transformation, coordinates, "Could not transform coordinates");
}

std::vector<Coord> transform_coordinates(
    uint32_t source_epsg,
    uint32_t destination_epsg,
    std::span<const Coord> coordinates
) {
  if (coordinates.empty())
    return {};
  if (source_epsg == destination_epsg)
    return {coordinates.begin(), coordinates.end()};
  CoordinateTransform transformation(source_epsg, destination_epsg);
  return transformation.apply(coordinates);
}

std::vector<Coord> transform_coordinates_to_local_aeqd(
    uint32_t source_epsg,
    LatLon anchor,
    std::span<const Coord> coordinates
) {
  CoordinateTransform transformation(source_epsg, anchor);
  return transformation.apply(coordinates);
}

std::vector<Coord> transform_coordinates_from_local_aeqd(
    uint32_t destination_epsg,
    LatLon anchor,
    std::span<const Coord> coordinates
) {
  CoordinateTransform transformation(anchor, destination_epsg);
  return transformation.apply(coordinates);
}

bool epsg_uses_projected_metres(uint32_t epsg_code) {
  OGRSpatialReference reference = spatial_reference(epsg_code);
  const char *unit_name = nullptr;
  const double units = reference.GetLinearUnits(&unit_name);
  return reference.IsProjected() && std::isfinite(units) && std::abs(units - 1.0) < 1e-12;
}

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

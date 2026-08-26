#include "source_catalogue.h"

#include "gdal_utils.h"

#include <cpl_conv.h>
#include <cpl_string.h>
#include <cpl_vsi.h>
#include <gdal_priv.h>
#include <ogr_spatialref.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace panorama::terrain {
namespace {

/// Return a path's lower-case filename extension.
[[nodiscard]] std::string lower_extension(const std::filesystem::path &path) {
  std::string extension = path.extension().string();
  std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char value) {
    return static_cast<char>(std::tolower(value));
  });
  return extension;
}

/// Return whether a loose file is one of the supported DEM representations.
[[nodiscard]] bool has_source_extension(const std::filesystem::path &path) {
  const std::string extension = lower_extension(path);
  return extension == ".tif" || extension == ".tiff" || extension == ".hgt" || extension == ".asc";
}

/// Return a lower-case copy suitable for matching case-sensitive archive members.
[[nodiscard]] std::string lower_string(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
    return static_cast<char>(std::tolower(character));
  });
  return value;
}

/// Return an absolute, lexically normal path without requiring it to exist.
[[nodiscard]] std::filesystem::path normal_path(const std::filesystem::path &path) {
  std::error_code error;
  const std::filesystem::path absolute = std::filesystem::absolute(path, error);
  if (error) {
    throw std::runtime_error("Could not resolve path " + path.string() + ": " + error.message());
  }
  return absolute.lexically_normal();
}

/// Return whether `candidate` is equal to or below `directory`.
[[nodiscard]] bool
is_within(const std::filesystem::path &candidate, const std::filesystem::path &directory) {
  auto candidate_part = candidate.begin();
  for (auto directory_part = directory.begin(); directory_part != directory.end();
       directory_part++) {
    if (candidate_part == candidate.end() || *candidate_part != *directory_part) {
      return false;
    }
    candidate_part++;
  }
  return true;
}

/// Convert a positive GDAL raster dimension to the tool's fixed-width type.
[[nodiscard]] uint32_t checked_dimension(int value, const std::string &name) {
  if (value <= 0 || static_cast<uintmax_t>(value) > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error("DEM source has an invalid dimension: " + name);
  }
  return static_cast<uint32_t>(value);
}

/// Require a coordinate to occupy an integer location on a reference sample grid.
void validate_grid_alignment(
    double coordinate,
    double reference,
    double resolution,
    const char *axis,
    const std::string &name
) {
  const double index = (coordinate - reference) / resolution;
  if (!approximately_equal(index, std::round(index))) {
    throw std::runtime_error(
        "DEM source " + name + " is not aligned with the shared " + axis + " sample grid"
    );
  }
}

/// Read the source's PixelIsArea or PixelIsPoint registration declaration.
[[nodiscard]] RasterRegistration registration_from_dataset(GDALDataset &dataset) {
  const char *area_or_point = dataset.GetMetadataItem(GDALMD_AREA_OR_POINT);
  if (area_or_point != nullptr && std::string(area_or_point) == "Point") {
    return RasterRegistration::PixelIsPoint;
  }
  return RasterRegistration::PixelIsArea;
}

/// Compare CRS definitions independently of GDAL's dataset axis-to-SRS mapping.
///
/// SRTMHGT exposes conventional raster X/Y as longitude/latitude while EPSG:4326
/// declares latitude/longitude authority order. Parsing its WKT alone therefore
/// produces a different data-axis mapping even though the underlying CRS is the
/// same. Terrain arrays consistently use conventional GIS X/Y, so normalise both
/// sides before comparison.
[[nodiscard]] bool
same_spatial_reference(const OGRSpatialReference &candidate, const std::string &shared_wkt) {
  OGRSpatialReference shared;
  if (shared.SetFromUserInput(shared_wkt.c_str()) != OGRERR_NONE) {
    throw std::runtime_error("Could not parse the shared DEM coordinate reference system");
  }
  OGRSpatialReference normalised_candidate(candidate);
  normalised_candidate.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
  shared.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
  return normalised_candidate.IsSame(&shared) != 0;
}

/// Return an EPSG authority code already present in a spatial reference.
[[nodiscard]] std::optional<uint32_t> epsg_code(const OGRSpatialReference &reference) {
  OGRSpatialReference identified(reference);
  identified.AutoIdentifyEPSG();
  for (const char *node : {static_cast<const char *>(nullptr), "PROJCS", "GEOGCS"}) {
    const char *text = identified.GetAuthorityCode(node);
    if (text == nullptr) {
      continue;
    }
    uint32_t code = 0U;
    const std::string_view value(text);
    const auto result = std::from_chars(value.data(), value.data() + value.size(), code);
    if (result.ec == std::errc() && result.ptr == value.data() + value.size()) {
      return code;
    }
  }
  return std::nullopt;
}

/// Read a small text sidecar through GDAL's filesystem abstraction.
[[nodiscard]] std::string read_text_file(const std::string &path) {
  VSIStatBufL status = {};
  if (VSIStatL(path.c_str(), &status) != 0 || status.st_size < 0 ||
      static_cast<uintmax_t>(status.st_size) > std::numeric_limits<size_t>::max()) {
    throw std::runtime_error("Could not inspect CRS sidecar " + path);
  }
  VSILFILE *file = VSIFOpenL(path.c_str(), "rb");
  if (file == nullptr) {
    throw std::runtime_error("Could not open CRS sidecar " + path);
  }
  std::string contents(static_cast<size_t>(status.st_size), '\0');
  const size_t count = VSIFReadL(contents.data(), 1U, contents.size(), file);
  const int close_result = VSIFCloseL(file);
  if (count != contents.size() || close_result != 0) {
    throw std::runtime_error("Could not read CRS sidecar " + path);
  }
  return contents;
}

/// Extract the authoritative EPSG identifier emitted by OS Terrain GML files.
[[nodiscard]] std::optional<uint32_t>
epsg_code_from_gml(const std::optional<std::string> &gml_path) {
  if (!gml_path.has_value()) {
    return std::nullopt;
  }
  const std::string contents = read_text_file(*gml_path);
  constexpr std::string_view marker = "urn:ogc:def:crs:EPSG::";
  std::optional<uint32_t> code;
  size_t position = 0U;
  while ((position = contents.find(marker, position)) != std::string::npos) {
    const char *begin = contents.data() + position + marker.size();
    const char *end = begin;
    while (end != contents.data() + contents.size() &&
           std::isdigit(static_cast<unsigned char>(*end))) {
      end++;
    }
    uint32_t parsed = 0U;
    const auto result = std::from_chars(begin, end, parsed);
    if (begin == end || result.ec != std::errc()) {
      throw std::runtime_error("Invalid EPSG identifier in CRS sidecar " + *gml_path);
    }
    if (code.has_value() && *code != parsed) {
      throw std::runtime_error("Conflicting EPSG identifiers in CRS sidecar " + *gml_path);
    }
    code = parsed;
    position = static_cast<size_t>(end - contents.data());
  }
  return code;
}

/// Export a stable WKT, preferring an authoritative EPSG code from the package.
[[nodiscard]] std::pair<std::string, std::string> projection_description(
    const OGRSpatialReference &dataset_reference,
    const std::optional<uint32_t> package_epsg,
    const std::string &name
) {
  OGRSpatialReference reference(dataset_reference);
  if (package_epsg.has_value()) {
    OGRSpatialReference authoritative;
    if (authoritative.importFromEPSG(static_cast<int>(*package_epsg)) != OGRERR_NONE) {
      throw std::runtime_error("Could not import packaged EPSG code for DEM source " + name);
    }
    reference = authoritative;
  }
  char *projection_text = nullptr;
  if (reference.exportToWkt(&projection_text) != OGRERR_NONE || projection_text == nullptr) {
    throw std::runtime_error("Could not serialise DEM source CRS: " + name);
  }
  const std::string projection_wkt(projection_text);
  CPLFree(projection_text);
  return {
      projection_wkt,
      reference.GetName() == nullptr ? "unknown" : reference.GetName(),
  };
}

/// Open one candidate through the explicitly supported GDAL raster drivers.
[[nodiscard]] GdalDatasetPointer open_source(const SourceLocation &location) {
  const char *drivers[] = {"GTiff", "SRTMHGT", "AAIGrid", nullptr};
  GDALDataset *dataset = static_cast<GDALDataset *>(GDALOpenEx(
      location.dataset.c_str(),
      GDAL_OF_RASTER | GDAL_OF_READONLY | GDAL_OF_VERBOSE_ERROR,
      drivers,
      nullptr,
      nullptr
  ));
  if (dataset == nullptr) {
    throw std::runtime_error(gdal_error("Could not open DEM source " + location.display_name));
  }
  return GdalDatasetPointer(dataset);
}

/// Parse and validate one DEM source without reading its elevation pixels.
[[nodiscard]] SourceRaster
inspect_source(const SourceLocation &location, SourceGrid *shared_grid, bool first_source) {
  const std::string &name = location.display_name;
  GdalDatasetPointer dataset = open_source(location);
  if (dataset->GetRasterCount() != 1) {
    throw std::runtime_error("DEM source must have exactly one raster band: " + name);
  }

  std::array<double, 6> transform = {};
  if (dataset->GetGeoTransform(transform.data()) != CE_None) {
    throw std::runtime_error("DEM source has no affine geotransform: " + name);
  }
  if (transform[1] <= 0.0 || transform[5] >= 0.0 || transform[2] != 0.0 || transform[4] != 0.0 ||
      !approximately_equal(transform[1], -transform[5])) {
    throw std::runtime_error(
        "DEM source must be a north-up, unrotated grid with square pixels: " + name
    );
  }

  const RasterRegistration registration = registration_from_dataset(*dataset);
  if (registration == RasterRegistration::PixelIsPoint) {
    // GDAL always reports an outer pixel-corner geotransform, including for
    // point-registered rasters. Rechunking reasons about sample positions, so
    // move the origin to the centre of sample (0, 0). For N46E007.hgt this
    // converts (6.999861..., 47.000138...) to the stated (7, 47) HGT bounds.
    transform[0] += 0.5 * transform[1] + 0.5 * transform[2];
    transform[3] += 0.5 * transform[4] + 0.5 * transform[5];
  }

  const OGRSpatialReference *reference = dataset->GetSpatialRef();
  if (reference == nullptr || (reference->IsProjected() == 0 && reference->IsGeographic() == 0)) {
    throw std::runtime_error("DEM source must declare a projected or geographic CRS: " + name);
  }
  const std::optional<uint32_t> packaged_epsg = epsg_code_from_gml(location.gml_dataset);
  const std::optional<uint32_t> declared_epsg = epsg_code(*reference);
  if (packaged_epsg.has_value() && declared_epsg.has_value() && *packaged_epsg != *declared_epsg) {
    throw std::runtime_error("DEM raster and GML sidecar declare different CRSs: " + name);
  }
  const std::optional<uint32_t> resolved_epsg =
      packaged_epsg.has_value() ? packaged_epsg : declared_epsg;
  const auto [projection_wkt, coordinate_system_name] =
      projection_description(*reference, packaged_epsg, name);

  GDALRasterBand *band = dataset->GetRasterBand(1);
  const char *unit_text = band->GetUnitType();
  const std::string elevation_unit = unit_text == nullptr ? "" : unit_text;

  if (first_source) {
    shared_grid->projection_wkt = projection_wkt;
    shared_grid->coordinate_system_name = coordinate_system_name;
    shared_grid->elevation_unit = elevation_unit;
    shared_grid->epsg_code = resolved_epsg;
    shared_grid->x_resolution = transform[1];
    shared_grid->y_resolution = -transform[5];
    shared_grid->reference_x = transform[0];
    shared_grid->reference_y = transform[3];
    shared_grid->registration = registration;
  } else {
    const bool matching_epsg = shared_grid->epsg_code.has_value() && resolved_epsg.has_value() &&
                               *shared_grid->epsg_code == *resolved_epsg;
    if (!matching_epsg && !same_spatial_reference(*reference, shared_grid->projection_wkt)) {
      throw std::runtime_error("DEM source CRS does not match the first source: " + name);
    }
    if (!approximately_equal(transform[1], shared_grid->x_resolution) ||
        !approximately_equal(-transform[5], shared_grid->y_resolution)) {
      throw std::runtime_error("DEM source resolution does not match the first source: " + name);
    }
    if (registration != shared_grid->registration) {
      throw std::runtime_error(
          "DEM source pixel registration does not match the first source: " + name
      );
    }
    if (elevation_unit != shared_grid->elevation_unit) {
      throw std::runtime_error(
          "DEM source elevation unit does not match the first source: " + name
      );
    }
    validate_grid_alignment(
        transform[0],
        shared_grid->reference_x,
        shared_grid->x_resolution,
        "X",
        name
    );
    validate_grid_alignment(
        transform[3],
        shared_grid->reference_y,
        shared_grid->y_resolution,
        "Y",
        name
    );
  }

  int has_no_data = false;
  const double no_data_value = band->GetNoDataValue(&has_no_data);
  int has_scale = false;
  const double scale = band->GetScale(&has_scale);
  int has_offset = false;
  const double offset = band->GetOffset(&has_offset);

  return {
      location,
      transform,
      checked_dimension(dataset->GetRasterXSize(), name),
      checked_dimension(dataset->GetRasterYSize(), name),
      static_cast<int>(band->GetRasterDataType()),
      has_no_data != 0 ? std::optional<double>(no_data_value) : std::nullopt,
      has_scale != 0 ? scale : 1.0,
      has_offset != 0 ? offset : 0.0,
      band->GetMaskFlags(),
      registration,
  };
}

/// Describe one loose raster and any adjacent GML CRS sidecar.
[[nodiscard]] SourceLocation loose_source(const std::filesystem::path &path) {
  SourceLocation location = {path.string(), path.string(), std::nullopt};
  if (lower_extension(path) == ".asc") {
    std::filesystem::path gml = path;
    gml.replace_extension(".gml");
    if (std::filesystem::is_regular_file(gml)) {
      location.gml_dataset = gml.string();
    }
  }
  return location;
}

/// Return every ASC member of one ZIP without extracting the archive.
[[nodiscard]] std::vector<SourceLocation> archive_sources(const std::filesystem::path &archive) {
  const std::string root = "/vsizip/" + archive.generic_string();
  char **members = VSIReadDirRecursive(root.c_str());
  if (members == nullptr) {
    throw std::runtime_error("Could not inspect ZIP archive " + archive.string());
  }

  std::vector<std::string> names;
  std::map<std::string, std::string> names_by_lowercase;
  for (size_t index = 0U; members[index] != nullptr; index++) {
    const std::string member(members[index]);
    names.push_back(member);
    names_by_lowercase.emplace(lower_string(member), member);
  }
  CSLDestroy(members);

  std::vector<SourceLocation> sources;
  for (const std::string &member : names) {
    if (lower_extension(std::filesystem::path(member)) != ".asc") {
      continue;
    }
    std::filesystem::path gml_member(member);
    gml_member.replace_extension(".gml");
    const auto gml = names_by_lowercase.find(lower_string(gml_member.generic_string()));
    sources.push_back(
        {
            root + "/" + member,
            archive.string() + "!/" + member,
            gml == names_by_lowercase.end() ? std::nullopt
                                            : std::optional<std::string>(root + "/" + gml->second),
        }
    );
  }
  return sources;
}

} // namespace

SourceCatalogue SourceCatalogue::discover(
    const std::filesystem::path &input_directory,
    const std::filesystem::path &excluded_directory
) {
  register_gdal_drivers();
  const std::filesystem::path input = normal_path(input_directory);
  const std::filesystem::path excluded =
      excluded_directory.empty() ? std::filesystem::path() : normal_path(excluded_directory);
  if (!std::filesystem::is_directory(input)) {
    throw std::invalid_argument("Input path is not a directory: " + input.string());
  }

  std::vector<SourceLocation> locations;
  std::error_code iteration_error;
  std::filesystem::recursive_directory_iterator iterator(
      input,
      std::filesystem::directory_options::skip_permission_denied,
      iteration_error
  );
  const std::filesystem::recursive_directory_iterator end;
  while (iterator != end) {
    if (iteration_error) {
      throw std::runtime_error(
          "Could not scan input directory " + input.string() + ": " + iteration_error.message()
      );
    }

    const std::filesystem::directory_entry &entry = *iterator;
    const std::filesystem::path candidate = normal_path(entry.path());
    if (!excluded.empty() && entry.is_directory() && is_within(candidate, excluded)) {
      iterator.disable_recursion_pending();
    } else if (entry.is_regular_file() && has_source_extension(candidate)) {
      locations.push_back(loose_source(candidate));
    } else if (entry.is_regular_file() && lower_extension(candidate) == ".zip") {
      std::vector<SourceLocation> archive = archive_sources(candidate);
      locations.insert(
          locations.end(),
          std::make_move_iterator(archive.begin()),
          std::make_move_iterator(archive.end())
      );
    }
    iterator.increment(iteration_error);
  }
  if (iteration_error) {
    throw std::runtime_error(
        "Could not scan input directory " + input.string() + ": " + iteration_error.message()
    );
  }
  std::sort(locations.begin(), locations.end(), [](const auto &left, const auto &right) {
    return left.display_name < right.display_name;
  });
  if (locations.empty()) {
    throw std::runtime_error(
        "No GeoTIFF, HGT, ASC, or ZIP-contained ASC files were found below " + input.string()
    );
  }

  SourceGrid grid = {};
  std::vector<SourceRaster> sources;
  sources.reserve(locations.size());
  for (const SourceLocation &location : locations) {
    sources.push_back(inspect_source(location, &grid, sources.empty()));
  }
  return SourceCatalogue(input, std::move(grid), std::move(sources));
}

SourceCatalogue::SourceCatalogue(
    std::filesystem::path input_directory,
    SourceGrid grid,
    std::vector<SourceRaster> sources
)
    : input_directory_(std::move(input_directory)), grid_(std::move(grid)),
      sources_(std::move(sources)) {}

const std::filesystem::path &SourceCatalogue::input_directory() const { return input_directory_; }

const SourceGrid &SourceCatalogue::grid() const { return grid_; }

const std::vector<SourceRaster> &SourceCatalogue::sources() const { return sources_; }

} // namespace panorama::terrain

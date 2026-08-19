#include "arguments.h"

#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>

namespace panorama::arguments {

std::string_view option_value(int argc, const char *argv[], int &index, std::string_view option) {
  if (index + 1 >= argc) {
    throw std::invalid_argument("Missing value for " + std::string(option));
  }
  index++;
  return argv[index];
}

uint64_t parse_uint64(std::string_view text, std::string_view name) {
  uint64_t value = 0U;
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc() || end != text.data() + text.size()) {
    throw std::invalid_argument(
        "Invalid value for " + std::string(name) + ": " + std::string(text)
    );
  }
  return value;
}

uint32_t parse_uint32(std::string_view text, std::string_view name, bool allow_zero) {
  const uint64_t value = parse_uint64(text, name);
  if (value > std::numeric_limits<uint32_t>::max() || (!allow_zero && value == 0U)) {
    throw std::out_of_range(
        "Value for " + std::string(name) + " is outside the permitted uint32 range"
    );
  }
  return static_cast<uint32_t>(value);
}

double parse_finite_double(std::string_view text, std::string_view name) {
  const std::string owned(text);
  char *end = nullptr;
  errno = 0;
  const double value = std::strtod(owned.c_str(), &end);
  if (errno != 0 || end == owned.c_str() || *end != '\0' || !std::isfinite(value)) {
    throw std::invalid_argument("Invalid value for " + std::string(name) + ": " + owned);
  }
  return value;
}

} // namespace panorama::arguments

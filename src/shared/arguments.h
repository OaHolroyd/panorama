#pragma once

#include <cstdint>
#include <string_view>

namespace panorama::arguments {

/// Return the command-line value following an option and advance its index.
[[nodiscard]] std::string_view
option_value(int argc, const char *argv[], int &index, std::string_view option);

/// Parse a non-negative integer representable as an unsigned 64-bit value.
[[nodiscard]] uint64_t parse_uint64(std::string_view text, std::string_view name);

/// Parse a positive or non-negative integer representable as uint32.
[[nodiscard]] uint32_t parse_uint32(std::string_view text, std::string_view name, bool allow_zero);

/// Parse a finite double-precision command-line value.
[[nodiscard]] double parse_finite_double(std::string_view text, std::string_view name);

} // namespace panorama::arguments

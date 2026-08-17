#include "timer.h"

#include <iterator>
#include <stdexcept>
#include <unistd.h>

namespace panorama {
namespace {

constexpr const char *kNormalWhite = "\033[37m";
constexpr const char *kDimWhite = "\033[2;37m";
constexpr const char *kBoldWhite = "\033[1;37m";
constexpr const char *kResetStyle = "\033[0m";

/// Return whether a C stream is connected to an ANSI-capable terminal.
[[nodiscard]] bool supports_color(FILE *stream) {
  return stream != nullptr && isatty(fileno(stream)) != 0;
}

/// Print one timer row, optionally wrapped in an ANSI terminal style.
void print_row(
    FILE *stream,
    int name_width,
    const char *name,
    double milliseconds,
    const char *kind,
    const char *style
) {
  if (style != nullptr) {
    std::fputs(style, stream);
  }
  std::fprintf(stream, "  %-*s: %8.3f ms (%s)\n", name_width, name, milliseconds, kind);
  if (style != nullptr) {
    std::fputs(kResetStyle, stream);
  }
}

} // namespace

Timer::Timer(std::string_view total_name)
    : total_name_(total_name), total_started_(std::chrono::steady_clock::now()) {}

void Timer::start_wall(std::string_view name) { start(name, Kind::Wall); }

void Timer::start_work(std::string_view name) { start(name, Kind::Work); }

void Timer::stop(std::string_view name) {
  std::lock_guard<std::mutex> lock(mutex_);
  const std::thread::id calling_thread = std::this_thread::get_id();
  for (auto iterator = active_measurements_.rbegin(); iterator != active_measurements_.rend();
       ++iterator) {
    if (iterator->name != name || iterator->thread_id != calling_thread) {
      continue;
    }
    const std::chrono::duration<double, std::milli> elapsed =
        std::chrono::steady_clock::now() - iterator->started;
    Measurement &measurement = find_or_create(iterator->name);
    if (iterator->kind == Kind::Wall) {
      measurement.wall += elapsed;
    } else {
      measurement.work += elapsed;
    }
    active_measurements_.erase(std::next(iterator).base());
    return;
  }
  throw std::logic_error("Timer has no active region named: " + std::string(name));
}

void Timer::start(std::string_view name, Kind kind) {
  std::lock_guard<std::mutex> lock(mutex_);
  active_measurements_.push_back(
      {
          std::string(name),
          kind,
          std::this_thread::get_id(),
          std::chrono::steady_clock::now(),
      }
  );
}

void Timer::add_work(std::string_view name, std::chrono::duration<double, std::milli> duration) {
  std::lock_guard<std::mutex> lock(mutex_);
  find_or_create(name).work += duration;
}

void Timer::add_work(std::string_view name, double milliseconds) {
  add_work(name, std::chrono::duration<double, std::milli>(milliseconds));
}

std::chrono::duration<double, std::milli> Timer::total_elapsed() const {
  return std::chrono::steady_clock::now() - total_started_;
}

void Timer::print(FILE *stream) const {
  std::lock_guard<std::mutex> lock(mutex_);
  size_t name_width = total_name_.size();
  for (const Measurement &measurement : measurements_) {
    if (measurement.name.size() > name_width) {
      name_width = measurement.name.size();
    }
  }
  const int print_width = static_cast<int>(name_width);
  const bool color = supports_color(stream);
  for (const Measurement &measurement : measurements_) {
    if (measurement.wall.count() != 0.0) {
      print_row(
          stream,
          print_width,
          measurement.name.c_str(),
          measurement.wall.count(),
          "wall",
          color ? kNormalWhite : nullptr
      );
    }
    if (measurement.work.count() != 0.0) {
      print_row(
          stream,
          print_width,
          measurement.name.c_str(),
          measurement.work.count(),
          "work",
          color ? kDimWhite : nullptr
      );
    }
  }
  print_row(
      stream,
      print_width,
      total_name_.c_str(),
      total_elapsed().count(),
      "wall",
      color ? kBoldWhite : nullptr
  );
}

Timer::Measurement &Timer::find_or_create(std::string_view name) {
  for (Measurement &measurement : measurements_) {
    if (measurement.name == name) {
      return measurement;
    }
  }
  measurements_.push_back({std::string(name), {}, {}});
  return measurements_.back();
}

} // namespace panorama

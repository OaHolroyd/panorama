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

  // A strict per-thread stack is what makes the recorded parent relationships
  // trustworthy. Stopping an outer region while an inner one remains active
  // would otherwise create overlapping siblings in the printed tree.
  for (auto iterator = active_measurements_.rbegin(); iterator != active_measurements_.rend();
       iterator++) {
    if (iterator->thread_id != calling_thread) {
      continue;
    }
    const size_t measurement_index = iterator->measurement;
    if (measurements_[measurement_index].name != name) {
      throw std::logic_error(
          "Timer must stop innermost region " + measurements_[measurement_index].name +
          " before: " + std::string(name)
      );
    }
    const std::chrono::duration<double, std::milli> elapsed =
        std::chrono::steady_clock::now() - iterator->started;
    Measurement &measurement = measurements_[measurement_index];
    if (iterator->kind == Kind::Wall) {
      measurement.wall += elapsed;
    } else {
      measurement.work += elapsed;
    }
    active_measurements_.erase(std::next(iterator).base());
    remember_completed(calling_thread, measurement_index);
    return;
  }
  throw std::logic_error("Timer has no active region on the calling thread: " + std::string(name));
}

void Timer::start(std::string_view name, Kind kind) {
  std::lock_guard<std::mutex> lock(mutex_);
  const std::thread::id calling_thread = std::this_thread::get_id();
  const size_t measurement = find_or_create(name, active_parent(calling_thread));
  active_measurements_.push_back(
      {
          measurement,
          kind,
          calling_thread,
          std::chrono::steady_clock::now(),
      }
  );
}

void Timer::add_work(std::string_view name, std::chrono::duration<double, std::milli> duration) {
  std::lock_guard<std::mutex> lock(mutex_);
  const std::thread::id calling_thread = std::this_thread::get_id();

  // Device work is normally added while its enclosing wall region is active.
  // Prefer that exact node so both values appear on the same tree branch.
  for (auto iterator = active_measurements_.rbegin(); iterator != active_measurements_.rend();
       iterator++) {
    if (iterator->thread_id == calling_thread &&
        measurements_[iterator->measurement].name == name) {
      measurements_[iterator->measurement].work += duration;
      return;
    }
  }

  // An asynchronous operation may finish while a differently named host
  // region publishes its result. In that case place its work below the
  // currently active region, preserving where the completion was consumed
  // instead of attaching every later sample to an older same-name node.
  const size_t parent = active_parent(calling_thread);
  if (parent != kRootMeasurement) {
    measurements_[find_or_create(name, parent)].work += duration;
    return;
  }

  // Some APIs expose device timestamps only after the wall region has been
  // stopped. Attach those values to the most recently completed same-name
  // node on this thread rather than creating an unrelated root measurement.
  for (auto iterator = completed_measurements_.rbegin(); iterator != completed_measurements_.rend();
       iterator++) {
    if (iterator->thread_id == calling_thread && iterator->name == name) {
      measurements_[iterator->measurement].work += duration;
      return;
    }
  }

  measurements_[find_or_create(name, kRootMeasurement)].work += duration;
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

  // Include indentation in the alignment width so values remain in one
  // readable column while the names visibly communicate nesting.
  std::vector<size_t> depths(measurements_.size(), 1U);
  for (size_t index = 0U; index < measurements_.size(); index++) {
    size_t parent = measurements_[index].parent;
    while (parent != kRootMeasurement) {
      depths[index]++;
      parent = measurements_[parent].parent;
    }
    const size_t displayed_width = 2U * depths[index] + measurements_[index].name.size();
    if (displayed_width > name_width) {
      name_width = displayed_width;
    }
  }
  const int print_width = static_cast<int>(name_width);
  const bool color = supports_color(stream);

  // Print the total first so every indented row below it reads naturally as
  // an inclusive child. Measurements retain insertion order among siblings.
  print_row(
      stream,
      print_width,
      total_name_.c_str(),
      total_elapsed().count(),
      "wall",
      color ? kBoldWhite : nullptr
  );

  const auto print_children = [&](auto &&self, size_t parent, size_t depth) -> void {
    for (size_t index = 0U; index < measurements_.size(); index++) {
      const Measurement &measurement = measurements_[index];
      if (measurement.parent != parent) {
        continue;
      }
      const std::string displayed_name(2U * depth, ' ');
      const std::string label = displayed_name + measurement.name;
      if (measurement.wall.count() != 0.0) {
        print_row(
            stream,
            print_width,
            label.c_str(),
            measurement.wall.count(),
            "wall",
            color ? kNormalWhite : nullptr
        );
      }
      if (measurement.work.count() != 0.0) {
        print_row(
            stream,
            print_width,
            label.c_str(),
            measurement.work.count(),
            "work",
            color ? kDimWhite : nullptr
        );
      }
      self(self, index, depth + 1U);
    }
  };
  print_children(print_children, kRootMeasurement, 1U);
}

size_t Timer::find_or_create(std::string_view name, size_t parent) {
  for (size_t index = 0U; index < measurements_.size(); index++) {
    if (measurements_[index].name == name && measurements_[index].parent == parent) {
      return index;
    }
  }
  measurements_.push_back({std::string(name), parent, {}, {}});
  return measurements_.size() - 1U;
}

size_t Timer::active_parent(std::thread::id thread_id) const {
  for (auto iterator = active_measurements_.rbegin(); iterator != active_measurements_.rend();
       iterator++) {
    if (iterator->thread_id == thread_id) {
      return iterator->measurement;
    }
  }
  return kRootMeasurement;
}

void Timer::remember_completed(std::thread::id thread_id, size_t measurement) {
  const std::string &name = measurements_[measurement].name;
  for (CompletedMeasurement &completed : completed_measurements_) {
    if (completed.thread_id == thread_id && completed.name == name) {
      completed.measurement = measurement;
      return;
    }
  }
  completed_measurements_.push_back({name, thread_id, measurement});
}

} // namespace panorama

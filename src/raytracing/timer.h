#pragma once

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace panorama {

/// Collect named elapsed-time measurements within one overall wall-clock run.
///
/// A wall region measures elapsed time between `start_wall` and `stop`.
/// Work measurements are additive, making them suitable for parallel workers
/// whose aggregate effort can exceed the overall wall-clock duration. All
/// methods are safe to call from multiple threads.
class Timer {
public:
  /// Construct and start an overall wall-clock timer with the supplied name.
  explicit Timer(std::string_view total_name);

  /// Start one named wall-clock region on the calling thread.
  void start_wall(std::string_view name);

  /// Start one named additive-work region on the calling thread.
  void start_work(std::string_view name);

  /// Stop the latest active named region started by the calling thread.
  ///
  /// Throws when the calling thread has no active region with this name.
  void stop(std::string_view name);

  /// Add one duration of parallel or device work to a named measurement.
  void add_work(std::string_view name, std::chrono::duration<double, std::milli> duration);

  /// Add a millisecond-valued parallel or device work measurement.
  void add_work(std::string_view name, double milliseconds);

  /// Return the elapsed wall-clock duration since construction.
  [[nodiscard]] std::chrono::duration<double, std::milli> total_elapsed() const;

  /// Print the overall wall-clock time followed by every named measurement.
  void print(FILE *stream = stdout) const;

private:
  /// Distinguish elapsed wall-clock regions from additive parallel work.
  enum class Kind {
    Wall,
    Work,
  };

  /// One named pair of accumulated wall-clock and additive-work measurements.
  struct Measurement {
    std::string name;
    std::chrono::duration<double, std::milli> wall;
    std::chrono::duration<double, std::milli> work;
  };

  /// One active region owned by a particular calling thread.
  struct ActiveMeasurement {
    std::string name;
    Kind kind;
    std::thread::id thread_id;
    std::chrono::steady_clock::time_point started;
  };

  /// Return the measurement with `name`, creating it if needed.
  Measurement &find_or_create(std::string_view name);

  /// Start one named region of `kind` on the calling thread.
  void start(std::string_view name, Kind kind);

  std::string total_name_;
  std::chrono::steady_clock::time_point total_started_;
  mutable std::mutex mutex_;
  std::vector<Measurement> measurements_;
  std::vector<ActiveMeasurement> active_measurements_;
};

} // namespace panorama

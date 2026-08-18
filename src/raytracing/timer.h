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
/// Regions started while another region is active on the same thread become
/// children of that region. The printed hierarchy therefore makes inclusive
/// parent timings explicit instead of presenting nested durations as if they
/// were independent costs. A wall region measures elapsed time between
/// `start_wall` and `stop`. Work measurements are additive, making them
/// suitable for parallel workers whose aggregate effort can exceed the
/// overall wall-clock duration. All methods are safe to call from multiple
/// threads.
class Timer {
public:
  /// Construct and start an overall wall-clock timer with the supplied name.
  explicit Timer(std::string_view total_name);

  /// Start one named wall-clock region on the calling thread.
  void start_wall(std::string_view name);

  /// Start one named additive-work region on the calling thread.
  void start_work(std::string_view name);

  /// Stop the innermost active region started by the calling thread.
  ///
  /// Throws if the calling thread has no active region or `name` is not its
  /// innermost region, preserving a well-formed timing hierarchy.
  void stop(std::string_view name);

  /// Add one duration of parallel or device work to a named measurement.
  void add_work(std::string_view name, std::chrono::duration<double, std::milli> duration);

  /// Add a millisecond-valued parallel or device work measurement.
  void add_work(std::string_view name, double milliseconds);

  /// Return the elapsed wall-clock duration since construction.
  [[nodiscard]] std::chrono::duration<double, std::milli> total_elapsed() const;

  /// Print the overall wall-clock time followed by its nested timing tree.
  void print(FILE *stream = stdout) const;

private:
  /// Distinguish elapsed wall-clock regions from additive parallel work.
  enum class Kind {
    Wall,
    Work,
  };

  /// One named pair of measurements at one fixed position in the timing tree.
  struct Measurement {
    std::string name;
    size_t parent;
    std::chrono::duration<double, std::milli> wall;
    std::chrono::duration<double, std::milli> work;
  };

  /// One active region owned by a particular calling thread.
  struct ActiveMeasurement {
    size_t measurement;
    Kind kind;
    std::thread::id thread_id;
    std::chrono::steady_clock::time_point started;
  };

  /// Remember the last completed node with one name on a calling thread.
  struct CompletedMeasurement {
    std::string name;
    std::thread::id thread_id;
    size_t measurement;
  };

  /// Sentinel identifying a measurement directly below the overall timer.
  static constexpr size_t kRootMeasurement = static_cast<size_t>(-1);

  /// Return a tree node with `name` and `parent`, creating it if needed.
  size_t find_or_create(std::string_view name, size_t parent);

  /// Return the innermost active node on one thread, or the root sentinel.
  [[nodiscard]] size_t active_parent(std::thread::id thread_id) const;

  /// Associate a stopped node with its thread for a subsequent `add_work`.
  void remember_completed(std::thread::id thread_id, size_t measurement);

  /// Start one named region of `kind` on the calling thread.
  void start(std::string_view name, Kind kind);

  std::string total_name_;
  std::chrono::steady_clock::time_point total_started_;
  mutable std::mutex mutex_;
  std::vector<Measurement> measurements_;
  std::vector<ActiveMeasurement> active_measurements_;
  std::vector<CompletedMeasurement> completed_measurements_;
};

} // namespace panorama

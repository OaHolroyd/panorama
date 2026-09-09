#pragma once

#include <atomic>
#include <chrono>
#include <cstdio>

namespace panorama::trace_activity {

inline double seconds() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// The viewer binds only its render worker. The monitor reads these approximate
// samples without touching renderer locks; CLI/tests leave diagnostics unbound.
struct Activity {
  std::atomic<const char *> stage{"idle"};
  std::atomic<double> since{seconds()};
};
inline thread_local Activity *current = nullptr;

struct Binding {
  Activity *previous;
  explicit Binding(Activity *activity) : previous(current) { current = activity; }
  ~Binding() { current = previous; }
  Binding(const Binding &) = delete;
  Binding &operator=(const Binding &) = delete;
};

class Scope {
public:
  explicit Scope(const char *name, double *milliseconds = nullptr)
      : activity_(current), name_(name), milliseconds_(milliseconds) {
    if (activity_ == nullptr && milliseconds_ == nullptr)
      return;
    started_ = seconds();
    if (activity_ != nullptr) {
      previous_ = activity_->stage.load(std::memory_order_relaxed);
      previous_since_ = activity_->since.load(std::memory_order_relaxed);
      activity_->since.store(started_, std::memory_order_relaxed);
      activity_->stage.store(name_, std::memory_order_relaxed);
    }
  }
  ~Scope() {
    if (activity_ == nullptr && milliseconds_ == nullptr)
      return;
    const double elapsed = 1000.0 * (seconds() - started_);
    if (milliseconds_ != nullptr)
      *milliseconds_ += elapsed;
    if (activity_ != nullptr) {
      // Inclusive timings: nested slow stages must not be added together.
      if (elapsed >= 100.0)
        std::printf("Slow stage: %s wall %.3f ms (inclusive)\n", name_, elapsed);
      activity_->since.store(previous_since_, std::memory_order_relaxed);
      activity_->stage.store(previous_, std::memory_order_relaxed);
    }
  }
  Scope(const Scope &) = delete;
  Scope &operator=(const Scope &) = delete;

private:
  Activity *activity_;
  const char *name_, *previous_ = nullptr;
  double *milliseconds_;
  double started_ = 0.0, previous_since_ = 0.0;
};

} // namespace panorama::trace_activity

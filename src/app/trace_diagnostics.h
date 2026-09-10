#pragma once

#include "trace_activity.h"

#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#include <mach/mach.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <thread>

namespace panorama::app::diagnostics {

// Set once before starting the renderer or AppKit callbacks.
inline bool enabled = false;
inline const auto epoch = std::chrono::steady_clock::now();
inline double seconds() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - epoch).count();
}

struct Activity {
  std::atomic<const char *> stage{"idle"};
  std::atomic<double> since{0.0};
  std::atomic<unsigned long long> calls{0}, submitted{0}, completed{0}, presented{0};
  std::atomic<unsigned long long> stale{0}, unavailable{0}, revision{0};
  std::atomic<unsigned long long> refreshed{0};

  void mark(const char *name) {
    if (enabled) {
      since.store(seconds(), std::memory_order_relaxed);
      stage.store(name, std::memory_order_relaxed);
    }
  }
};
inline Activity worker, display, minimap;
inline trace_activity::Activity terrain;

// Scope cleanup also records early returns, exceptions, and pool-drain time.
struct Scope {
  Activity &activity;
  explicit Scope(Activity &value) : activity(value) {
    if (enabled)
      ++activity.calls;
    activity.mark("begin");
  }
  ~Scope() { activity.mark("idle"); }
};

inline void
track_submission(Activity &activity, id<MTLCommandBuffer> command, id<CAMetalDrawable> drawable) {
  if (!enabled)
    return;
  // Capture only the process-lifetime counters, never the drawable or command.
  Activity *counters = &activity;
  ++activity.submitted;
  [command addCompletedHandler:^(id<MTLCommandBuffer>) {
    ++counters->completed;
  }];
  [drawable addPresentedHandler:^(id<MTLDrawable>) {
    ++counters->presented;
  }];
}

// Independent of the render worker and AppKit: a blocked callback must not
// prevent its own diagnostic from being emitted. No renderer locks are taken.
class Monitor {
public:
  explicit Monitor(id<MTLDevice> device) {
    if (!enabled)
      return;
    thread_ = std::thread([this, device] {
      std::unique_lock lock(mutex_);
      while (!changed_.wait_for(lock, std::chrono::seconds(1), [this] { return stopping_; })) {
        @autoreleasepool {
          task_vm_info_data_t vm = {};
          mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
          const bool memory_valid = task_info(
                                        mach_task_self(),
                                        TASK_VM_INFO,
                                        reinterpret_cast<task_info_t>(&vm),
                                        &count
                                    ) == KERN_SUCCESS;
          const double now = seconds();
          std::fprintf(
              stdout,
              "Health t=%.3f: memory %s footprint/resident %.1f/%.1f MiB, "
              "Metal allocated/recommended %.1f/%.1f MiB, thermal=%ld\n",
              now,
              memory_valid ? "ok" : "unavailable",
              double(vm.phys_footprint) / 1048576.0,
              double(vm.resident_size) / 1048576.0,
              double(device.currentAllocatedSize) / 1048576.0,
              double(device.recommendedMaxWorkingSetSize) / 1048576.0,
              static_cast<long>(NSProcessInfo.processInfo.thermalState)
          );
          report("worker", worker, now);
          report("display", display, now);
          report("minimap", minimap, now);
          std::fprintf(
              stdout,
              "Health t=%.3f: terrain stage=%s age=%.1f ms\n",
              now,
              terrain.stage.load(),
              1000.0 * (trace_activity::seconds() - terrain.since.load())
          );
          std::fflush(stdout);
        }
      }
    });
  }
  ~Monitor() {
    {
      std::lock_guard lock(mutex_);
      stopping_ = true;
    }
    changed_.notify_one();
    if (thread_.joinable())
      thread_.join();
  }

private:
  static void report(const char *name, const Activity &activity, double now) {
    // Stage and age are approximate independent samples; counters accumulate.
    std::fprintf(
        stdout,
        "Health t=%.3f: %s stage=%s age=%.1f ms, calls=%llu revision=%llu, "
        "submitted/completed/presented=%llu/%llu/%llu, stale=%llu unavailable=%llu "
        "refreshed=%llu\n",
        now,
        name,
        activity.stage.load(),
        (now - activity.since.load()) * 1000.0,
        activity.calls.load(),
        activity.revision.load(),
        activity.submitted.load(),
        activity.completed.load(),
        activity.presented.load(),
        activity.stale.load(),
        activity.unavailable.load(),
        activity.refreshed.load()
    );
  }
  std::mutex mutex_;
  std::condition_variable changed_;
  bool stopping_ = false;
  std::thread thread_;
};

} // namespace panorama::app::diagnostics

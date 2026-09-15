#include "panorama_controller_private.h"

#include "solar_position.h"
#include "timer.h"
#include "trace_diagnostics.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

@implementation PanoramaController (TextEditing)

- (void)peakLabelsChanged:(id)sender {
  (void)sender;
  _peakLabelMode = static_cast<panorama::app::PeakLabelMode>(_peakLabelControl.indexOfSelectedItem);
  [NSUserDefaults.standardUserDefaults setInteger:static_cast<NSInteger>(_peakLabelMode)
                                           forKey:@"panorama.peak-labels.mode"];
  [_panoramaView setPeakLabelMode:_peakLabelMode];
  _renderer->request_peak_labels_enabled(_peakLabelMode != panorama::app::PeakLabelMode::Off);
}

- (void)controlTextDidChange:(NSNotification *)notification {
  NSTextField *changed = notification.object;
  if (changed == _coordinateInputControl) {
    [self updateCoordinateInputValidation];
  }
}

- (void)controlTextDidEndEditing:(NSNotification *)notification {
  NSTextField *field = notification.object;
  if (field == _resolutionScaleControl) {
    [self commitResolutionControls];
  } else if (field == _minimumControl || field == _maximumControl) {
    [self publishTerrainControls];
  } else if (field == _astronomicalDateControl) {
    [self astronomicalInputChanged:field];
  } else if (field == _groundClearanceControl) {
    [self commitGroundClearanceControl];
  } else if (field == _coordinateInputControl) {
    [self updateCoordinateInputValidation];
  }
}

- (void)useNativeResolution:(id)sender {
  (void)sender;
  const CGFloat scale = _panoramaView.window.backingScaleFactor;
  if (!(scale > 0.0)) {
    NSBeep();
    return;
  }
  _resolutionScaleControl.stringValue = panorama::app::format_range_value(scale);
  [self commitResolutionControls];
}

- (void)zoomWithScrollDelta:(double)delta precise:(bool)precise {
  if (_viewerPaused) {
    return;
  }
  // Exponential scaling makes equal scroll motion feel proportional at wide
  // and narrow fields of view. Trackpads report much finer-grained deltas
  // than traditional mouse wheels and therefore use a gentler coefficient.
  const double sensitivity = precise ? 0.012 : 0.08;
  constexpr double kMinimumFieldOfView = 5.0 * std::numbers::pi / 180.0;
  constexpr double kMaximumFieldOfView = 140.0 * std::numbers::pi / 180.0;
  const double next = std::clamp(
      _verticalFieldOfView * std::exp(-delta * sensitivity),
      kMinimumFieldOfView,
      kMaximumFieldOfView
  );
  if (std::abs(next - _verticalFieldOfView) <= 1e-12) {
    return;
  }
  [self setVerticalFieldOfViewDegrees:next * panorama::app::kRadiansToDegrees];
}

@end

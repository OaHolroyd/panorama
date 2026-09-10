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

- (void)updateAspectLockAppearance {
  const BOOL locked = _aspectLockControl.state == NSControlStateValueOn;
  _aspectLockControl.image = [NSImage
      imageWithSystemSymbolName:locked ? @"lock.fill" : @"lock.open"
       accessibilityDescription:locked ? @"Aspect ratio locked" : @"Aspect ratio unlocked"];
}

- (void)aspectLockChanged:(NSButton *)sender {
  if (sender.state == NSControlStateValueOn) {
    const std::optional<uint32_t> width =
        panorama::app::parse_image_dimension(_imageWidthControl.stringValue);
    const std::optional<uint32_t> height =
        panorama::app::parse_image_dimension(_imageHeightControl.stringValue);
    if (width.has_value() && height.has_value()) {
      _lockedAspectRatio = static_cast<double>(*width) / *height;
    }
  }
  [self updateAspectLockAppearance];
}

/// Maintain the captured aspect ratio while either dimension is edited. The
/// paired field changes immediately; editing completion commits both values.
- (void)controlTextDidChange:(NSNotification *)notification {
  NSTextField *changed = notification.object;
  if (changed == _coordinateInputControl) {
    [self updateCoordinateInputValidation];
    return;
  }
  if (_updatingResolutionControls || _aspectLockControl.state != NSControlStateValueOn) {
    return;
  }
  if (changed != _imageWidthControl && changed != _imageHeightControl) {
    return;
  }
  const std::optional<uint32_t> value = panorama::app::parse_image_dimension(changed.stringValue);
  if (!value.has_value() || !std::isfinite(_lockedAspectRatio) || _lockedAspectRatio <= 0.0) {
    return;
  }

  const double paired_value = changed == _imageWidthControl
                                  ? static_cast<double>(*value) / _lockedAspectRatio
                                  : static_cast<double>(*value) * _lockedAspectRatio;
  if (paired_value < 1.0 || paired_value > std::numeric_limits<uint32_t>::max()) {
    return;
  }
  _updatingResolutionControls = true;
  NSTextField *paired = changed == _imageWidthControl ? _imageHeightControl : _imageWidthControl;
  paired.stringValue =
      [NSString stringWithFormat:@"%u", static_cast<uint32_t>(std::llround(paired_value))];
  _updatingResolutionControls = false;
}

- (void)controlTextDidEndEditing:(NSNotification *)notification {
  NSTextField *field = notification.object;
  if (field == _imageWidthControl || field == _imageHeightControl) {
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

/// Match the render aspect to the available window content by changing only
/// its horizontal pixel count. Vertical resolution and vertical FOV remain
/// untouched.
- (void)matchWindowResolution:(id)sender {
  (void)sender;
  const NSSize available = _aspectFitView.bounds.size;
  const std::optional<uint32_t> height =
      panorama::app::parse_image_dimension(_imageHeightControl.stringValue);
  if (!height.has_value() || available.width <= 0.0 || available.height <= 0.0) {
    NSBeep();
    return;
  }
  const double width = static_cast<double>(*height) * available.width / available.height;
  if (!std::isfinite(width) || width < 1.0 || width > std::numeric_limits<uint32_t>::max()) {
    NSBeep();
    return;
  }
  const uint32_t rounded_width = static_cast<uint32_t>(std::llround(width));
  _updatingResolutionControls = true;
  _imageWidthControl.stringValue = [NSString stringWithFormat:@"%u", rounded_width];
  _updatingResolutionControls = false;
  if (_aspectLockControl.state == NSControlStateValueOn) {
    _lockedAspectRatio = static_cast<double>(rounded_width) / *height;
  }
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

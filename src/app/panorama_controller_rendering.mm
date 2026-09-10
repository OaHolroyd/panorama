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

@implementation PanoramaController (Rendering)

- (void)updateSettingsControlAvailability {
  const BOOL scalarColour = _colourSourceControl.indexOfSelectedItem != 0;
  for (NSView *row in _scalarColourRows) {
    row.hidden = !scalarColour;
  }
  _featureOutlineDetailRow.hidden = _featureOutlinesControl.state != NSControlStateValueOn;
  const BOOL normalLighting = _normalLightingControl.state == NSControlStateValueOn;
  const BOOL manualSun = _sunModeControl.selectedSegment == 0;
  const BOOL hasObserverTimeZone = _observerTimeZone != nil;
  for (NSView *view in _normalLightingRows) {
    view.hidden = !normalLighting;
  }
  for (NSView *row in _manualSunRows) {
    row.hidden = !normalLighting || !manualSun;
  }
  for (NSView *row in _astronomicalSunRows) {
    row.hidden = !normalLighting || manualSun;
  }

  const BOOL astronomicalTimeEnabled = normalLighting && !manualSun && hasObserverTimeZone;
  _astronomicalDateControl.enabled = astronomicalTimeEnabled;
  _astronomicalTimeControl.enabled = astronomicalTimeEnabled;
  _astronomicalTimeDecreaseControl.enabled =
      astronomicalTimeEnabled && _astronomicalTimeControl.doubleValue > 0.0;
  _astronomicalTimeIncreaseControl.enabled =
      astronomicalTimeEnabled && _astronomicalTimeControl.doubleValue < 1439.0;
}

- (void)renderModeChanged:(id)sender {
  (void)sender;
  [self updateSettingsControlAvailability];
  [self publishTerrainControls];
}

/// Commit the colour controls independently of camera and lighting edits.
- (BOOL)publishTerrainControls {
  const std::optional<double> minimum =
      panorama::app::parse_range_value(_minimumControl.stringValue);
  const std::optional<double> maximum =
      panorama::app::parse_range_value(_maximumControl.stringValue);
  const BOOL validRange = minimum.has_value() && maximum.has_value() && *maximum > *minimum &&
                          *minimum >= -std::numeric_limits<float>::max() &&
                          *maximum <= std::numeric_limits<float>::max();
  _minimumControl.textColor = validRange ? NSColor.controlTextColor : NSColor.systemRedColor;
  _maximumControl.textColor = validRange ? NSColor.controlTextColor : NSColor.systemRedColor;
  NSString *rangeError = validRange
                             ? nil
                             : @"Enter finite metre values with the maximum greater than the "
                                "minimum; commas may only separate thousands.";
  _minimumControl.toolTip = rangeError;
  _maximumControl.toolTip = rangeError;
  const BOOL scalarColour = _colourSourceControl.indexOfSelectedItem != 0;
  if (scalarColour && !validRange) {
    return NO;
  }

  _presentation.appearance.colour_source =
      static_cast<panorama::TerrainColourSource>(_colourSourceControl.indexOfSelectedItem);
  _presentation.appearance.colourmap =
      static_cast<panorama::PresetColourmap>(_colourmapControl.indexOfSelectedItem);
  _presentation.appearance.colour_scale =
      static_cast<panorama::ScalarColourScale>(_colourScaleControl.indexOfSelectedItem);
  if (validRange) {
    _presentation.colour_range = {
        static_cast<float>(*minimum),
        static_cast<float>(*maximum),
    };
  }
  _renderer->request_presentation(_presentation);
  return YES;
}

/// Resize only after both text fields form a valid Metal image size. Invalid
/// edits remain visible in red so the user can correct them without dismissing
/// an alert or losing the partially entered value.
- (BOOL)commitResolutionControls {
  const std::optional<uint32_t> width =
      panorama::app::parse_image_dimension(_imageWidthControl.stringValue);
  const std::optional<uint32_t> height =
      panorama::app::parse_image_dimension(_imageHeightControl.stringValue);
  const uint64_t pixelCount =
      width.has_value() && height.has_value() ? static_cast<uint64_t>(*width) * *height : 0U;
  const BOOL valid =
      width.has_value() && height.has_value() && pixelCount <= std::numeric_limits<uint32_t>::max();
  _imageWidthControl.textColor = valid ? NSColor.controlTextColor : NSColor.systemRedColor;
  _imageHeightControl.textColor = valid ? NSColor.controlTextColor : NSColor.systemRedColor;
  NSString *resolutionError =
      valid ? nil
            : @"Width and height must be positive whole numbers whose product fits in the "
               "32-bit Metal ray-index range.";
  _imageWidthControl.toolTip = resolutionError;
  _imageHeightControl.toolTip = resolutionError;
  if (!valid) {
    return NO;
  }
  const panorama::ImageSize next_image = {*width, *height};
  if (next_image != _image) {
    _image = next_image;
    _inspectionRequestToken = _renderer->request_inspection(std::nullopt);
    _renderer->request_view(_orientation, _verticalFieldOfView, _image);
  }
  return YES;
}

- (void)drawInMTKView:(MTKView *)view {
  panorama::app::diagnostics::Scope diagnostic_scope(panorama::app::diagnostics::display);
  @autoreleasepool {
    [self drawPanoramaInView:view];
    panorama::app::diagnostics::display.mark("pool-drain");
  }
}

- (void)drawPanoramaInView:(MTKView *)view {
  namespace diagnostics = panorama::app::diagnostics;
  diagnostics::display.mark("snapshot");
  panorama::app::PresentedFrame frame = _renderer->presented_frame();
  diagnostics::display.mark("layout");
  if (frame.texture != nil && frame.output_image.width != 0U && frame.output_image.height != 0U) {
    [_aspectFitView setAspectRatio:static_cast<CGFloat>(frame.output_image.width) /
                                   static_cast<CGFloat>(frame.output_image.height)];
  }
  diagnostics::display.mark("drawable");
  id<CAMetalDrawable> drawable = view.currentDrawable;
  if (drawable == nil) {
    if (diagnostics::enabled)
      ++diagnostics::display.unavailable;
    return;
  }
  diagnostics::display.mark("command-buffer");
  id<MTLCommandBuffer> command = [_renderer->command_queue() commandBuffer];
  if (command == nil) {
    return;
  }

  command.label = @"Present panorama";
  diagnostics::display.mark("publication-lock");
  if (!_renderer->submit_presentation(command, frame, drawable))
    return;
  if (diagnostics::enabled)
    diagnostics::display.revision = frame.revision;
  diagnostics::display.mark("ui-update");

  if (frame.target_visibility_sequence != _displayedTargetVisibilitySequence) {
    _displayedTargetVisibilitySequence = frame.target_visibility_sequence;
    if (frame.target_visibility.has_value() &&
        frame.target_visibility->request_token == _targetVisibilityRequestToken) {
      _targetVisibilityRevision = frame.target_visibility->revision;
      _lockedPointOccluded = frame.target_visibility->occluded;
      [self updateLockedPointIndicatorWithOrientation:frame.orientation
                                  verticalFieldOfView:frame.vertical_field_of_view
                                                image:frame.output_image];
    }
  }

  bool acceptedRoamMove = false;
  if (frame.roam_result_sequence != _displayedRoamResultSequence) {
    _displayedRoamResultSequence = frame.roam_result_sequence;
    if (frame.roam_result.has_value()) {
      if (frame.roam_result->accepted) {
        acceptedRoamMove = true;
        _groundClearance = std::max(
            0.0,
            frame.observer.elevation - static_cast<double>(frame.roam_result->ground_elevation)
        );
        if (frame.roam_result->request_token == _roamRequestToken) {
          if (_roamAltitudeModeControl.selectedSegment == 0 &&
              _groundClearanceControl.currentEditor == nil) {
            _groundClearanceControl.stringValue =
                [NSString stringWithFormat:@"%.1f", _groundClearance];
          } else if (_roamAltitudeModeControl.selectedSegment == 1) {
            _roamAltitude = frame.observer.elevation;
            if (_groundClearanceControl.currentEditor == nil) {
              _groundClearanceControl.stringValue =
                  [NSString stringWithFormat:@"%.1f", _roamAltitude];
            }
          }
        }
      } else if (frame.roam_result->request_token == _roamRequestToken) {
        [self clearRoamKeys];
        _roamDesiredPosition = {frame.observer.easting, frame.observer.northing};
        const bool terrainCollision = std::isfinite(frame.roam_result->ground_elevation);
        if ([self isCruisingEnabled] && !_viewerPaused) {
          [self pauseCruiseForTerrainCollision:terrainCollision];
        }
        if (_cruiseRecovery) {
          _roamStatusLabel.stringValue =
              terrainCollision ? @"Blocked by terrain • drag to steer/climb • Space resumes"
                               : @"No terrain coverage • drag to turn back • Space resumes";
          _roamStatusLabel.textColor = NSColor.systemRedColor;
        } else if (![self isCruisingEnabled]) {
          _roamStatusLabel.stringValue =
              terrainCollision ? @"Movement stopped: terrain reaches the held altitude"
                               : @"Movement stopped: no terrain coverage";
          _roamStatusLabel.textColor = NSColor.systemRedColor;
        }
        NSBeep();
      }
    }
  }

  if (frame.map_point_sequence != _displayedMapPointSequence) {
    _displayedMapPointSequence = frame.map_point_sequence;
    if (frame.map_point_request_token == _mapPointRequestToken &&
        _mapPointAction != panorama::app::MapPointAction::None) {
      const panorama::app::MapPointAction action = _mapPointAction;
      const bool coordinateMove =
          action == panorama::app::MapPointAction::MoveObserver && _coordinateMovePending;
      _mapPointAction = panorama::app::MapPointAction::None;
      _coordinateMovePending = false;
      if (!frame.map_point.has_value()) {
        _mapHoverPoint.reset();
        if (action == panorama::app::MapPointAction::Hover) {
          [self updatePointInfo:std::nullopt];
          [self updateLockedPointIndicatorWithOrientation:frame.orientation
                                      verticalFieldOfView:frame.vertical_field_of_view
                                                    image:frame.output_image];
        } else {
          NSBeep();
          [self setPointInfoStatus:@"No terrain coverage"];
          if (coordinateMove) {
            _coordinateStatusLabel.stringValue = @"No terrain coverage at that coordinate";
            _coordinateStatusLabel.textColor = NSColor.systemRedColor;
          }
        }
      } else {
        const double distance = std::hypot(
            frame.map_point->easting - _observer.easting,
            frame.map_point->northing - _observer.northing
        );
        panorama::app::PointInspection point = {
            .pixel = {},
            .revision = frame.revision,
            .hit = true,
            .distance = static_cast<float>(distance),
            .elevation = frame.map_point->elevation,
            .easting = frame.map_point->easting,
            .northing = frame.map_point->northing,
            .slope_degrees = 0.0F,
            .aspect_degrees = 0.0F,
            .map_selected = true,
        };
        if (action == panorama::app::MapPointAction::Hover) {
          _mapHoverPoint = point;
          [self updatePointInfo:point];
          [self updateLockedPointIndicatorWithOrientation:frame.orientation
                                      verticalFieldOfView:frame.vertical_field_of_view
                                                    image:frame.output_image];
        } else if (action == panorama::app::MapPointAction::MoveObserver) {
          if (coordinateMove) {
            [self updateCoordinateInputValidation];
          }
          [self moveObserverToTerrainPoint:*frame.map_point];
        } else {
          _mapHoverPoint.reset();
          _pointInspectionLocked = true;
          _pointLockPending = false;
          _lockedPoint = point;
          [self requestTargetVisibilityForPoint:*frame.map_point];
          [self updatePointInfo:point];
          [self lookAtTerrainPoint:*frame.map_point];
        }
      }
    }
  }

  if (frame.inspection_sequence != _displayedInspectionSequence) {
    _displayedInspectionSequence = frame.inspection_sequence;
    const bool matches_visible_frame =
        !frame.inspection.has_value() || frame.inspection->revision == frame.revision;
    if (_pointLockPending && frame.inspection_request_token == _pointLockRequestToken &&
        matches_visible_frame) {
      _pointLockPending = false;
      if (frame.inspection.has_value() && frame.inspection->hit) {
        _pointInspectionLocked = true;
        _lockedPoint = frame.inspection;
        [self updatePointInfo:frame.inspection];
        const panorama::app::TerrainPoint target = {
            frame.inspection->easting,
            frame.inspection->northing,
            frame.inspection->elevation,
        };
        [self requestTargetVisibilityForPoint:target];
        [self updateLockedPointIndicatorWithOrientation:frame.orientation
                                    verticalFieldOfView:frame.vertical_field_of_view
                                                  image:frame.output_image];
        // The label now owns the immutable sampled values. Stop the renderer
        // resampling this screen pixel as subsequent camera views complete.
        _renderer->request_inspection(std::nullopt);
      } else {
        [self updatePointInfo:frame.inspection];
      }
    } else if (_pointInspectionEnabled && !_pointInspectionLocked && !_pointLockPending &&
               _pointerOwner == panorama::app::PointerOwner::Panorama &&
               frame.inspection_request_token == _inspectionRequestToken) {
      [self updatePointInfo:matches_visible_frame ? frame.inspection : std::nullopt];
    }
  }

  if (!frame.error.empty()) {
    _window.titleVisibility = NSWindowTitleVisible;
    _window.title = [NSString stringWithFormat:@"panorama-app — error: %s", frame.error.c_str()];
  } else if (frame.revision != 0U && frame.revision != _displayedRevision) {
    _window.titleVisibility = NSWindowTitleHidden;
    _displayedRevision = frame.revision;
    const bool observerPositionMoved = frame.observer.easting != _observer.easting ||
                                       frame.observer.northing != _observer.northing;
    const bool observerMoved =
        observerPositionMoved || frame.observer.elevation != _observer.elevation;
    _observer = frame.observer;
    if (observerMoved) {
      if (_viewerPaused || (![self isCruisingEnabled] && ![self hasPressedRoamKey])) {
        _roamDesiredPosition = {_observer.easting, _observer.northing};
      }
      [_miniMapPanel setObserverEasting:_observer.easting northing:_observer.northing];
      if (acceptedRoamMove) {
        [_miniMapPanel centerOnObserver];
      }
      if (_pointInspectionLocked && _lockedPoint.has_value()) {
        _lockedPoint->distance = static_cast<float>(std::hypot(
            _lockedPoint->easting - _observer.easting,
            _lockedPoint->northing - _observer.northing
        ));
        [self updatePointInfo:_lockedPoint];
      } else {
        [self updatePointInfo:std::nullopt];
      }
      const bool movingContinuously = ([self isRoamingEnabled] && [self hasPressedRoamKey]) ||
                                      ([self isCruisingEnabled] && !_viewerPaused);
      if (observerPositionMoved && !movingContinuously) {
        [self resolveObserverTimeZone];
      }
    }
    const double fps = frame.milliseconds > 0.0 ? 1'000.0 / frame.milliseconds : 0.0;
    [self updateDebugInfoWithOrientation:frame.orientation
                     verticalFieldOfView:frame.vertical_field_of_view
                                   image:frame.image
                            milliseconds:frame.milliseconds
                         gpuMilliseconds:frame.gpu_milliseconds
                                streamed:frame.streamed
                                revision:frame.revision];
    [self updateLockedPointIndicatorWithOrientation:frame.orientation
                                verticalFieldOfView:frame.vertical_field_of_view
                                              image:frame.output_image];
    [_miniMapPanel setCameraOrientation:frame.orientation
                    verticalFieldOfView:frame.vertical_field_of_view
                                  image:frame.output_image];
    [_miniMapPanel setVisibilityPoints:frame.visibility_points image:frame.image];
    [self updateMiniMapTelemetry];
    if (_metalfxStatusLabel != nil) {
      _metalfxStatusLabel.stringValue =
          frame.metalfx_enabled
              ? [NSString stringWithFormat:@"%u×%u → %u×%u · %s",
                                           frame.image.width,
                                           frame.image.height,
                                           frame.output_image.width,
                                           frame.output_image.height,
                                           panorama::app::metalfx_preset_name(frame.metalfx_preset)]
              : (_renderer->metalfx_supported() ? @"Native resolution"
                                                : @"MetalFX unavailable on this GPU · Native");
    }
    _window.title = [NSString
        stringWithFormat:@"panorama-app — heading %.1f°, pitch %.1f° — %.1f ms (%.1f fps)",
                         frame.orientation.heading * panorama::app::kRadiansToDegrees,
                         frame.orientation.pitch * panorama::app::kRadiansToDegrees,
                         frame.milliseconds,
                         fps];
  }
}

- (void)mtkView:(MTKView *)view drawableSizeWillChange:(CGSize)size {
  (void)view;
  (void)size;
}

@end

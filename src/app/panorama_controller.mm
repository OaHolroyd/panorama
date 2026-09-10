#include "panorama_controller.h"

#include "coordinate_input.h"
#include "metalfx_upscaler.h"
#include "solar_position.h"
#include "timer.h"
#include "trace_diagnostics.h"

#import <MapKit/MapKit.h>

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

/// Auto-detection may yield several syntactically valid grids. Prefer an
/// interpretation which lands on a prepared tile, while leaving genuine ties
/// for the user to resolve explicitly in the coordinate-system selector.
[[nodiscard]] static bool coordinate_has_terrain_coverage(
    const panorama::TerrainCoverage &coverage,
    panorama::Coord coordinate
) {
  try {
    const panorama::TileKey key = panorama::tile_key_at(coverage.grid, coordinate.x, coordinate.y);
    return std::binary_search(coverage.tiles.begin(), coverage.tiles.end(), key);
  } catch (const std::out_of_range &) {
    return false;
  }
}

/// Keep the compact point footer readable without sacrificing useful precision
/// for nearby terrain samples.
[[nodiscard]] static NSString *format_point_distance(double metres) {
  const double magnitude = std::abs(metres);
  if (magnitude < 100.0) {
    return [NSString stringWithFormat:@"%.1f m", metres];
  }
  if (magnitude < 1'000.0) {
    return [NSString stringWithFormat:@"%.0f m", metres];
  }
  const double kilometres = metres / 1'000.0;
  if (magnitude < 10'000.0) {
    return [NSString stringWithFormat:@"%.2f km", kilometres];
  }
  if (magnitude < 100'000.0) {
    return [NSString stringWithFormat:@"%.1f km", kilometres];
  }
  return [NSString stringWithFormat:@"%.0f km", kilometres];
}

[[nodiscard]] static NSString *format_movement_speed(double metres_per_second) {
  const double kilometres_per_hour = metres_per_second * 3.6;
  return std::abs(kilometres_per_hour) < 100.0
             ? [NSString stringWithFormat:@"%.1f km/h", kilometres_per_hour]
             : [NSString stringWithFormat:@"%.0f km/h", kilometres_per_hour];
}

[[nodiscard]] static double cruise_steering_response(double displacement) {
  const double magnitude = std::abs(displacement);
  if (magnitude <= panorama::app::kCruiseSteeringDeadZone) {
    return 0.0;
  }
  const double normalised = std::clamp(
      (magnitude - panorama::app::kCruiseSteeringDeadZone) /
          (1.0 - panorama::app::kCruiseSteeringDeadZone),
      0.0,
      1.0
  );
  return std::copysign(std::pow(normalised, panorama::app::kCruiseSteeringExponent), displacement);
}

@interface PanoramaController () {

@private
  panorama::app::ViewerRenderer *_renderer;
  __weak NSWindow *_window;
  __weak PanoramaView *_panoramaView;
  __weak ViewerOverlayView *_overlayView;
  __weak AspectFitContainerView *_aspectFitView;
  __weak MiniMapPanelView *_miniMapPanel;
  panorama::CameraOrientation _orientation;
  double _verticalFieldOfView;
  panorama::ImageSize _image;
  panorama::TerrainPresentationSettings _presentation;
  panorama::ObserverLocation _observer;
  std::optional<panorama::app::PointInspection> _lockedPoint;
  std::optional<panorama::app::PointInspection> _mapHoverPoint;
  NSPopUpButton *_colourSourceControl;
  NSPopUpButton *_raytracerControl;
  NSPopUpButton *_colourmapControl;
  NSPopUpButton *_colourScaleControl;
  NSPopUpButton *_metalfxActivationControl;
  NSPopUpButton *_metalfxPresetControl;
  NSTextField *_metalfxStatusLabel;
  NSTextField *_minimumControl;
  NSTextField *_maximumControl;
  NSSlider *_zoomControl;
  NSTextField *_zoomValueLabel;
  NSSlider *_panningSensitivityControl;
  NSTextField *_panningSensitivityLabel;
  NSTextField *_imageWidthControl;
  NSTextField *_imageHeightControl;
  NSButton *_aspectLockControl;
  NSButton *_matchWindowControl;
  NSButton *_invertMousePanningControl;
  NSButton *_bilinearCollisionControl;
  NSButton *_normalLightingControl;
  NSButton *_c1NormalsControl;
  NSButton *_raytracedShadowsControl;
  NSButton *_featureOutlinesControl;
  NSSlider *_featureOutlineDetailControl;
  NSTextField *_featureOutlineDetailLabel;
  NSSlider *_lodScaleControl;
  NSTextField *_lodScaleLabel;
  NSSlider *_sunAzimuthControl;
  NSTextField *_sunAzimuthLabel;
  NSSlider *_sunAltitudeControl;
  NSTextField *_sunAltitudeLabel;
  NSSegmentedControl *_sunModeControl;
  NSTextField *_astronomicalDateControl;
  NSSlider *_astronomicalTimeControl;
  NSTextField *_astronomicalTimeLabel;
  NSButton *_astronomicalTimeDecreaseControl;
  NSButton *_astronomicalTimeIncreaseControl;
  NSTextField *_daylightTimesLabel;
  NSStackView *_daylightSymbolsRow;
  NSTextField *_sunriseTimeLabel;
  NSTextField *_sunsetTimeLabel;
  MKReverseGeocodingRequest *_timeZoneRequest;
  NSTimeZone *_observerTimeZone;
  uint64_t _timeZoneRequestToken;
  bool _astronomicalControlsUseObserverTime;
  bool _timeZoneLookupInProgress;
  double _manualSunAzimuthDegrees;
  double _manualSunAltitudeDegrees;
  NSArray<NSView *> *_scalarColourRows;
  NSView *_featureOutlineDetailRow;
  NSArray<NSView *> *_normalLightingRows;
  NSArray<NSView *> *_manualSunRows;
  NSArray<NSView *> *_astronomicalSunRows;
  NSSlider *_diffusivityControl;
  NSTextField *_diffusivityLabel;
  NSSlider *_skyStrengthControl;
  NSTextField *_skyStrengthLabel;
  NSSlider *_skyDetailControl;
  NSTextField *_skyDetailLabel;
  NSTextField *_debugInfoLabel;
  NSTextField *_debugPointInfoLabel;
  NSTextField *_observerInfoLabel;
  NSTextField *_movementInfoLabel;
  NSStackView *_pointInfoRow;
  NSTextField *_pointInfoHeading;
  NSTextField *_pointInfoLabel;
  NSImageView *_pointVisibilityIcon;
  NSImageView *_pointLockIcon;
  NSButton *_moveToLockedPointControl;
  NSTextField *_groundClearanceControl;
  NSButton *_groundClearanceDecreaseControl;
  NSButton *_groundClearanceIncreaseControl;
  NSPopUpButton *_coordinateSystemControl;
  NSTextField *_coordinateInputControl;
  NSTextField *_coordinateStatusLabel;
  NSButton *_coordinateMoveControl;
  std::optional<panorama::app::ParsedCoordinateInput> _coordinateDestination;
  NSSegmentedControl *_movementModeControl;
  NSSegmentedControl *_roamTurningModeControl;
  NSView *_roamTurningModeRow;
  NSSlider *_roamMouseSensitivityControl;
  NSTextField *_roamMouseSensitivityLabel;
  NSView *_roamMouseSensitivityRow;
  NSSegmentedControl *_roamAltitudeModeControl;
  NSButton *_aircraftDynamicsControl;
  NSView *_aircraftDynamicsRow;
  NSSlider *_roamSpeedControl;
  NSTextField *_roamSpeedRowLabel;
  NSTextField *_roamSpeedLabel;
  NSSlider *_roamUpdateRateControl;
  NSTextField *_roamUpdateRateLabel;
  NSTextField *_roamStatusLabel;
  NSArray<NSView *> *_roamRows;
  NSTextField *_observerHeightLabel;
  NSTextField *_observerHeightUnit;
  id _pauseKeyMonitor;
  NSTimer *_roamTimer;
  panorama::app::MapCoordinate _roamDesiredPosition;
  std::chrono::steady_clock::time_point _lastRoamTick;
  uint64_t _roamRequestToken;
  uint64_t _displayedRoamResultSequence;
  uint64_t _displayedRevision;
  uint64_t _displayedInspectionSequence;
  uint64_t _pointLockRequestToken;
  uint64_t _displayedMapPointSequence;
  uint64_t _mapPointRequestToken;
  uint64_t _displayedTargetVisibilitySequence;
  uint64_t _targetVisibilityRequestToken;
  uint64_t _targetVisibilityRevision;
  uint64_t _inspectionRequestToken;
  panorama::app::MapPointAction _mapPointAction;
  panorama::app::PointerOwner _pointerOwner;
  double _lockedAspectRatio;
  double _panningSensitivity;
  double _groundClearance;
  double _roamAltitude;
  double _roamSpeed;
  double _cruiseSpeed;
  double _aircraftAirspeed;
  double _aircraftBank;
  double _cruiseSteeringX;
  double _cruiseSteeringY;
  bool _roamForwardPressed;
  bool _roamBackwardPressed;
  bool _roamLeftPressed;
  bool _roamRightPressed;
  bool _pointInspectionEnabled;
  bool _pointInspectionLocked;
  bool _pointLockPending;
  bool _lockedPointOccluded;
  bool _coordinateMovePending;
  bool _invertMousePanning;
  bool _viewerPaused;
  bool _cruiseRecovery;
  bool _cruiseSteeringActive;
  bool _bilinearCollisions;
  bool _c1Normals;
  bool _updatingResolutionControls;
  panorama::app::MetalFxActivation _metalfxActivation;
  panorama::app::MetalFxPreset _metalfxPreset;
  NSTimer *_metalfxSettleTimer;
}
- (instancetype)initWithRenderer:(panorama::app::ViewerRenderer *)renderer
                          window:(NSWindow *)window;
- (void)rotateHeading:(double)headingDelta pitch:(double)pitchDelta;
- (void)selectRaytracer:(NSMenuItem *)sender;
- (void)rotateForCurrentZoomHeading:(double)headingDelta pitch:(double)pitchDelta;
- (void)panForCurrentZoomHeading:(double)headingDelta pitch:(double)pitchDelta;
- (void)mouseTurnForCurrentZoomHeading:(double)headingDelta pitch:(double)pitchDelta;
- (void)zoomWithScrollDelta:(double)delta precise:(bool)precise;
- (void)attachPanoramaView:(PanoramaView *)panoramaView
               overlayView:(ViewerOverlayView *)overlayView
             aspectFitView:(AspectFitContainerView *)aspectFitView
              miniMapPanel:(MiniMapPanelView *)miniMapPanel;
- (void)inspectLocationX:(double)x y:(double)y;
- (void)invalidatePanoramaHover;
- (void)pointerMovedOverPanorama;
- (void)pointerMovedOverOccludingView:(NSView *)view;
- (void)panoramaPointerExited;
- (void)togglePointLockAtLocationX:(double)x y:(double)y;
- (void)toggleMapAndPointInspection:(id)sender;
- (BOOL)isMapAndPointInspectionEnabled;
- (BOOL)isRoamingEnabled;
- (BOOL)isCruisingEnabled;
- (BOOL)isAircraftDynamicsEnabled;
- (BOOL)isMouseTurningEnabled;
- (BOOL)isViewerPaused;
- (void)toggleViewerPause;
- (void)pauseCruiseForTerrainCollision:(bool)terrainCollision;
- (void)setRoamKey:(panorama::app::RoamKey)key pressed:(BOOL)pressed;
- (void)adjustCruiseSpeedBy:(double)delta;
- (void)setCruiseSteeringX:(double)x y:(double)y;
- (void)clearCruiseSteering;
- (void)setPointInfoStatus:(NSString *)status;
- (void)updateMiniMapTelemetry;
- (void)setPointInfoSymbolsVisible:(bool)visible locked:(bool)locked occluded:(bool)occluded;
- (void)moveObserverToTerrainPoint:(panorama::app::TerrainPoint)point;
- (void)moveToLockedPoint:(id)sender;
- (void)adjustGroundClearance:(NSButton *)sender;
- (BOOL)commitGroundClearanceControl;
- (void)movementModeChanged:(id)sender;
- (void)roamTurningModeChanged:(id)sender;
- (void)roamMouseSensitivityChanged:(id)sender;
- (void)roamAltitudeModeChanged:(id)sender;
- (void)aircraftDynamicsChanged:(id)sender;
- (void)levelAircraft;
- (void)roamSpeedChanged:(id)sender;
- (void)updateMovementSpeedControl;
- (void)roamUpdateRateChanged:(id)sender;
- (void)updateRoamControls;
- (void)updateCruiseHUD;
- (void)updateMovementStatus;
- (void)scheduleRoamTimer;
- (void)roamTimerFired:(NSTimer *)timer;
- (void)clearRoamKeys;
- (BOOL)hasPressedRoamKey;
- (void)coordinateSystemChanged:(id)sender;
- (void)moveToCoordinate:(id)sender;
- (BOOL)updateCoordinateInputValidation;
- (void)bilinearCollisionChanged:(NSButton *)sender;
- (void)c1NormalsChanged:(NSButton *)sender;
- (void)resolveObserverTimeZone;
- (void)setDaylightStatus:(NSString *)status;
- (BOOL)publishAstronomicalLighting;
- (BOOL)publishTerrainControls;
- (BOOL)commitResolutionControls;
- (void)metalfxChanged:(id)sender;
- (void)requestMetalFxInteraction;
- (void)settleMetalFx:(NSTimer *)timer;
- (NSViewController *)makeSettingsViewController;
- (NSViewController *)makePositioningViewController;
- (NSViewController *)makeDebugViewController;
- (NSViewController *)makePointInfoViewController;
@end

@implementation PanoramaController

- (instancetype)initWithRenderer:(panorama::app::ViewerRenderer *)renderer
                          window:(NSWindow *)window {
  self = [super init];
  if (self != nil) {
    _renderer = renderer;
    _window = window;
    _observer = renderer->observer();
    _orientation = renderer->initial_orientation();
    _verticalFieldOfView = renderer->initial_vertical_field_of_view();
    _image = renderer->initial_image();
    _lockedAspectRatio = static_cast<double>(_image.width) / _image.height;
    _panningSensitivity = 8.0;
    _groundClearance = renderer->ground_clearance();
    _roamAltitude = _observer.elevation;
    _roamSpeed = panorama::app::kDefaultRoamSpeed;
    _cruiseSpeed = panorama::app::kDefaultCruiseSpeed;
    _aircraftAirspeed = _cruiseSpeed;
    _roamDesiredPosition = {_observer.easting, _observer.northing};
    _presentation = renderer->initial_presentation();
    _bilinearCollisions = renderer->initial_bilinear_collisions();
    _c1Normals = renderer->initial_c1_normals();
    NSUserDefaults *defaults = NSUserDefaults.standardUserDefaults;
    const NSInteger savedActivation = [defaults integerForKey:@"panorama.metalfx.activation"];
    const NSInteger savedPreset = [defaults integerForKey:@"panorama.metalfx.preset"];
    _metalfxActivation = [defaults objectForKey:@"panorama.metalfx.activation"] != nil &&
                                 savedActivation >= 0 && savedActivation <= 2
                             ? static_cast<panorama::app::MetalFxActivation>(savedActivation)
                             : renderer->initial_metalfx_activation();
    _metalfxPreset = [defaults objectForKey:@"panorama.metalfx.preset"] != nil &&
                             savedPreset >= 0 && savedPreset <= 3
                         ? static_cast<panorama::app::MetalFxPreset>(savedPreset)
                         : renderer->initial_metalfx_preset();
    _renderer->request_metalfx(_metalfxActivation, _metalfxPreset, false);
    _mapPointAction = panorama::app::MapPointAction::None;
    _pointerOwner = panorama::app::PointerOwner::None;

    __weak PanoramaController *weakSelf = self;
    _pauseKeyMonitor = [NSEvent
        addLocalMonitorForEventsMatchingMask:NSEventMaskKeyDown
                                     handler:^NSEvent *(NSEvent *event) {
                                       PanoramaController *controller = weakSelf;
                                       if (controller == nil ||
                                           event.window != controller->_window ||
                                           event.keyCode != 49) {
                                         return event;
                                       }
                                       const NSEventModifierFlags modifiers =
                                           event.modifierFlags &
                                           NSEventModifierFlagDeviceIndependentFlagsMask;
                                       if ((modifiers & (NSEventModifierFlagCommand |
                                                         NSEventModifierFlagControl |
                                                         NSEventModifierFlagOption)) != 0U ||
                                           [controller->_window.firstResponder
                                               isKindOfClass:NSTextView.class]) {
                                         return event;
                                       }
                                       if (![controller isViewerPaused] &&
                                           ![controller isMouseTurningEnabled] &&
                                           ![controller isCruisingEnabled] &&
                                           controller->_window.firstResponder !=
                                               controller->_panoramaView) {
                                         return event;
                                       }
                                       if (!event.isARepeat) {
                                         [controller toggleViewerPause];
                                       }
                                       return nil;
                                     }];
  }
  return self;
}

- (void)dealloc {
  [_metalfxSettleTimer invalidate];
  if (_pauseKeyMonitor != nil) {
    [NSEvent removeMonitor:_pauseKeyMonitor];
  }
}

- (BOOL)isRoamingEnabled {
  return _movementModeControl != nil && _movementModeControl.selectedSegment == 1;
}

- (BOOL)isCruisingEnabled {
  return _movementModeControl != nil && _movementModeControl.selectedSegment == 2;
}

- (BOOL)isAircraftDynamicsEnabled {
  return [self isCruisingEnabled] && _aircraftDynamicsControl != nil &&
         _aircraftDynamicsControl.state == NSControlStateValueOn;
}

- (BOOL)isMouseTurningEnabled {
  return [self isRoamingEnabled] && _roamTurningModeControl != nil &&
         _roamTurningModeControl.selectedSegment == 1;
}

- (BOOL)isViewerPaused {
  return _viewerPaused;
}

- (void)toggleViewerPause {
  _viewerPaused = !_viewerPaused;
  _cruiseRecovery = false;
  [self clearRoamKeys];
  if (_viewerPaused) {
    [self clearCruiseSteering];
    [self levelAircraft];
  }
  _lastRoamTick = std::chrono::steady_clock::now();
  [_panoramaView setViewerPaused:_viewerPaused recoveryMessage:nil];
  if (_viewerPaused) {
    if ([self isCruisingEnabled]) {
      [self resolveObserverTimeZone];
    }
  }
  [self updateRoamControls];
}

- (void)pauseCruiseForTerrainCollision:(bool)terrainCollision {
  _viewerPaused = true;
  _cruiseRecovery = true;
  [self clearRoamKeys];
  [self clearCruiseSteering];
  [self levelAircraft];
  _lastRoamTick = std::chrono::steady_clock::now();
  NSString *message = terrainCollision
                          ? @"Terrain ahead — drag to steer or climb, then press Space"
                          : @"No terrain coverage — drag to turn back, then press Space";
  [_panoramaView setViewerPaused:true recoveryMessage:message];
  [self resolveObserverTimeZone];
  [self updateRoamControls];
}

- (BOOL)hasPressedRoamKey {
  return _roamForwardPressed || _roamBackwardPressed || _roamLeftPressed || _roamRightPressed;
}

- (void)clearRoamKeys {
  _roamForwardPressed = false;
  _roamBackwardPressed = false;
  _roamLeftPressed = false;
  _roamRightPressed = false;
}

- (void)setRoamKey:(panorama::app::RoamKey)key pressed:(BOOL)pressed {
  if (![self isRoamingEnabled] || _viewerPaused) {
    return;
  }
  const bool wasMoving = [self hasPressedRoamKey];
  bool *state = nullptr;
  switch (key) {
  case panorama::app::RoamKey::Forward:
    state = &_roamForwardPressed;
    break;
  case panorama::app::RoamKey::Backward:
    state = &_roamBackwardPressed;
    break;
  case panorama::app::RoamKey::Left:
    state = &_roamLeftPressed;
    break;
  case panorama::app::RoamKey::Right:
    state = &_roamRightPressed;
    break;
  }
  *state = pressed;
  if (pressed && !wasMoving) {
    _roamDesiredPosition = {_observer.easting, _observer.northing};
    _pointInspectionLocked = false;
    _pointLockPending = false;
    _lockedPoint.reset();
    _mapHoverPoint.reset();
    [self clearTargetVisibility];
    [_miniMapPanel clearInspectedPoint];
    [_panoramaView setTerrainPointIndicator:std::nullopt image:{} locked:true occluded:false];
  }
  if (![self hasPressedRoamKey]) {
    _lastRoamTick = std::chrono::steady_clock::now();
    [self updateMovementStatus];
  }
}

- (void)adjustCruiseSpeedBy:(double)delta {
  if (![self isCruisingEnabled] || (_viewerPaused && !_cruiseRecovery)) {
    return;
  }
  // Quarter-octave steps make key presses useful at walking and kilometre-per-
  // second speeds alike. Shift supplies four steps, i.e. one exact doubling.
  _cruiseSpeed = std::clamp(
      _cruiseSpeed * std::exp2(delta / 4.0),
      panorama::app::kMinimumMovementSpeed,
      panorama::app::kMaximumCruiseSpeed
  );
  [self updateMovementSpeedControl];
  [self updateMovementStatus];
  [self updateMiniMapTelemetry];
}

- (void)setCruiseSteeringX:(double)x y:(double)y {
  if (![self isCruisingEnabled]) {
    return;
  }
  _cruiseSteeringX = std::clamp(x, -1.0, 1.0);
  _cruiseSteeringY = std::clamp(y, -1.0, 1.0);
  const bool becameActive = !_cruiseSteeringActive;
  _cruiseSteeringActive = true;
  if (becameActive) {
    [self updateMovementStatus];
  }
}

- (void)clearCruiseSteering {
  const bool changed = _cruiseSteeringActive;
  _cruiseSteeringX = 0.0;
  _cruiseSteeringY = 0.0;
  _cruiseSteeringActive = false;
  if (changed) {
    [self updateMovementStatus];
  }
}

/// Pausing stops the flight simulation immediately and returns the camera to
/// wings-level attitude. Pitch and heading remain available to paused drag
/// interaction and become the aircraft's new attitude when Cruise resumes.
- (void)levelAircraft {
  _aircraftBank = 0.0;
  if (std::abs(_orientation.roll) <= 1e-12) {
    [self updateCruiseHUD];
    return;
  }
  _orientation.roll = 0.0;
  _renderer->request_view(_orientation, _verticalFieldOfView, _image);
  [self updateCruiseHUD];
  [self updateMiniMapTelemetry];
}

- (void)scheduleRoamTimer {
  [_roamTimer invalidate];
  _roamTimer = nil;
  if (![self isRoamingEnabled] && ![self isCruisingEnabled]) {
    return;
  }
  const double rate = std::max(1.0, std::round(_roamUpdateRateControl.doubleValue));
  _lastRoamTick = std::chrono::steady_clock::now();
  _roamTimer = [NSTimer timerWithTimeInterval:1.0 / rate
                                       target:self
                                     selector:@selector(roamTimerFired:)
                                     userInfo:nil
                                      repeats:YES];
  [NSRunLoop.mainRunLoop addTimer:_roamTimer forMode:NSRunLoopCommonModes];
}

- (void)roamTimerFired:(NSTimer *)timer {
  (void)timer;
  const auto now = std::chrono::steady_clock::now();
  const double requestedRate = std::max(1.0, std::round(_roamUpdateRateControl.doubleValue));
  const double elapsed = std::chrono::duration<double>(now - _lastRoamTick).count();
  _lastRoamTick = now;
  const bool roaming = [self isRoamingEnabled];
  const bool cruising = [self isCruisingEnabled];
  if ((_viewerPaused && !_cruiseRecovery) || (!roaming && !cruising) || !_window.isKeyWindow ||
      (roaming && _window.firstResponder != _panoramaView)) {
    [self clearRoamKeys];
    return;
  }

  // Do not turn a temporarily stalled main run loop into a large jump.
  const double dt = std::min(elapsed, 2.0 / requestedRate);
  if (_cruiseRecovery) {
    return;
  }

  const bool aircraft = cruising && [self isAircraftDynamicsEnabled];
  if (aircraft) {
    const double sensitivity = _roamMouseSensitivityControl.doubleValue;
    const double direction = _invertMousePanning ? -1.0 : 1.0;
    const double bankCommand =
        _cruiseSteeringActive
            ? std::clamp(
                  cruise_steering_response(_cruiseSteeringX) * sensitivity * direction,
                  -1.0,
                  1.0
              )
            : 0.0;
    const double targetBank = bankCommand * panorama::app::kAircraftMaximumBank;
    const double bankResponse = 1.0 - std::exp(-dt / panorama::app::kAircraftBankResponseSeconds);
    _aircraftBank += (targetBank - _aircraftBank) * bankResponse;

    const double pitchDelta = _cruiseSteeringActive ? cruise_steering_response(_cruiseSteeringY) *
                                                          panorama::app::kCruiseMaximumPitchRate *
                                                          sensitivity * direction * dt
                                                    : 0.0;
    constexpr double kPitchLimit = 60.0 * panorama::app::kDegreesToRadians;
    const double nextPitch = std::clamp(_orientation.pitch + pitchDelta, -kPitchLimit, kPitchLimit);
    const double horizontalSpeed = std::max(
        panorama::app::kMinimumMovementSpeed,
        _aircraftAirspeed * std::max(0.2, std::cos(nextPitch))
    );
    const double turnRate = std::clamp(
        panorama::app::kGravity * std::tan(_aircraftBank) / horizontalSpeed,
        -panorama::app::kCruiseMaximumYawRate,
        panorama::app::kCruiseMaximumYawRate
    );
    const double headingDelta = turnRate * dt;
    const bool attitudeChanged = std::abs(headingDelta) > 1e-12 || std::abs(pitchDelta) > 1e-12 ||
                                 std::abs(_orientation.roll + _aircraftBank) > 1e-12;
    _orientation.heading =
        std::remainder(_orientation.heading + headingDelta, 2.0 * std::numbers::pi);
    _orientation.pitch = nextPitch;
    // CameraOrientation's positive roll tilts the rendered horizon in the
    // opposite direction to the physical positive bank used above.
    _orientation.roll = -_aircraftBank;
    [self updateCruiseHUD];
    if (attitudeChanged) {
      [self requestMetalFxInteraction];
      _renderer->request_view(_orientation, _verticalFieldOfView, _image);
    }
  } else if (cruising && _cruiseSteeringActive) {
    const double sensitivity = _roamMouseSensitivityControl.doubleValue;
    const double direction = _invertMousePanning ? -1.0 : 1.0;
    [self rotateHeading:cruise_steering_response(_cruiseSteeringX) *
                        panorama::app::kCruiseMaximumYawRate * sensitivity * direction * dt
                  pitch:cruise_steering_response(_cruiseSteeringY) *
                        panorama::app::kCruiseMaximumPitchRate * sensitivity * direction * dt];
  }

  double forward = cruising ? 1.0
                            : static_cast<double>(_roamForwardPressed) -
                                  static_cast<double>(_roamBackwardPressed);
  double right =
      cruising ? 0.0
               : static_cast<double>(_roamRightPressed) - static_cast<double>(_roamLeftPressed);
  const double magnitude = std::hypot(forward, right);
  if (!(magnitude > 0.0)) {
    return;
  }
  forward /= magnitude;
  right /= magnitude;
  double movement_speed = cruising ? _cruiseSpeed : _roamSpeed;
  if (aircraft) {
    const double previousAirspeed = _aircraftAirspeed;
    const double trimAcceleration =
        (_cruiseSpeed - _aircraftAirspeed) / panorama::app::kAircraftTrimResponseSeconds;
    const double gravityAcceleration = -panorama::app::kGravity * std::sin(_orientation.pitch);
    _aircraftAirspeed = std::clamp(
        _aircraftAirspeed + (trimAcceleration + gravityAcceleration) * dt,
        panorama::app::kMinimumMovementSpeed,
        panorama::app::kMaximumCruiseSpeed
    );
    movement_speed = 0.5 * (previousAirspeed + _aircraftAirspeed);
  }
  const double distance = movement_speed * dt;
  const double heading = _orientation.heading;
  const bool flight = cruising && _roamAltitudeModeControl.selectedSegment == 1;
  const double horizontal_distance = flight ? distance * std::cos(_orientation.pitch) : distance;
  _roamDesiredPosition.easting +=
      horizontal_distance * (forward * std::sin(heading) + right * std::cos(heading));
  _roamDesiredPosition.northing +=
      horizontal_distance * (forward * std::cos(heading) - right * std::sin(heading));
  if (flight) {
    _roamAltitude += distance * std::sin(_orientation.pitch);
    if (_groundClearanceControl.currentEditor == nil) {
      _groundClearanceControl.stringValue = [NSString stringWithFormat:@"%.1f", _roamAltitude];
    }
  }
  const panorama::app::RoamAltitudeMode altitudeMode =
      _roamAltitudeModeControl.selectedSegment == 0 ? panorama::app::RoamAltitudeMode::FollowTerrain
                                                    : panorama::app::RoamAltitudeMode::HoldAltitude;
  const double height = altitudeMode == panorama::app::RoamAltitudeMode::FollowTerrain
                            ? _groundClearance
                            : _roamAltitude;
  // TODO: validate the complete movement segment rather than only its endpoint.
  // At very high Cruise speeds a single step could otherwise pass through a
  // narrow terrain ridge before the worker samples the destination elevation.
  [self requestMetalFxInteraction];
  _roamRequestToken = _renderer->request_roam(_roamDesiredPosition, altitudeMode, height);
  if (cruising) {
    [self updateMovementStatus];
    [self updateMiniMapTelemetry];
  } else {
    _roamStatusLabel.stringValue = @"Moving…";
    _roamStatusLabel.textColor = NSColor.secondaryLabelColor;
  }
}

- (void)updateMovementStatus {
  if (_roamStatusLabel == nil) {
    return;
  }
  if (_cruiseRecovery) {
    _roamStatusLabel.stringValue = @"Blocked • drag to steer/climb • Space resumes";
  } else if (_viewerPaused) {
    _roamStatusLabel.stringValue = @"Paused • Space resumes";
  } else if ([self isCruisingEnabled]) {
    _roamStatusLabel.stringValue = [self isAircraftDynamicsEnabled]
                                       ? @"Aircraft • mouse banks/pitches • W/S trim speed"
                                       : @"Cruising • W/S speed • Space pauses";
  } else {
    _roamStatusLabel.stringValue =
        [self isMouseTurningEnabled] ? @"WASD move • mouse looks" : @"WASD move • arrow keys look";
  }
  _roamStatusLabel.textColor = NSColor.secondaryLabelColor;
}

- (void)updateRoamControls {
  const BOOL roaming = [self isRoamingEnabled];
  const BOOL cruising = [self isCruisingEnabled];
  const BOOL moving = roaming || cruising;
  for (NSView *row in _roamRows) {
    row.hidden = !moving;
  }
  _roamTurningModeRow.hidden = !roaming;
  _aircraftDynamicsRow.hidden = !cruising;
  _roamMouseSensitivityRow.hidden = !cruising && ![self isMouseTurningEnabled];
  _roamMouseSensitivityControl.toolTip =
      [self isAircraftDynamicsEnabled]
          ? @"Aircraft bank and pitch sensitivity"
          : (cruising ? @"Maximum Cruise yaw and pitch rate" : @"Mouse turning sensitivity");
  [_roamAltitudeModeControl setLabel:cruising ? @"Flight" : @"Altitude" forSegment:1];
  _roamAltitudeModeControl.toolTip =
      cruising ? @"Maintain height above terrain, or use pitch to climb and descend in Flight mode"
               : @"Maintain height above terrain or hold absolute elevation while moving";
  const BOOL holdAltitude = moving && _roamAltitudeModeControl.selectedSegment == 1;
  _observerHeightLabel.stringValue = holdAltitude ? @"Altitude" : @"Eye height";
  _observerHeightUnit.stringValue = holdAltitude ? @"m AMSL" : @"m AGL";
  _observerHeightLabel.toolTip = holdAltitude
                                     ? @"Observer elevation above mean sea level"
                                     : @"Observer height above the terrain directly beneath it";
  _observerHeightUnit.toolTip =
      holdAltitude ? @"Metres above mean sea level" : @"Metres above ground level";
  _groundClearanceControl.toolTip = holdAltitude
                                        ? @"Observer elevation above mean sea level"
                                        : @"Observer height above the terrain directly beneath it";
  _groundClearanceDecreaseControl.toolTip =
      holdAltitude ? @"Lower altitude by 1 m (Option: 0.1 m; Shift: 10 m)"
                   : @"Lower eye height by 1 m (Option: 0.1 m; Shift: 10 m)";
  _groundClearanceIncreaseControl.toolTip =
      holdAltitude ? @"Raise altitude by 1 m (Option: 0.1 m; Shift: 10 m)"
                   : @"Raise eye height by 1 m (Option: 0.1 m; Shift: 10 m)";
  _groundClearanceControl.doubleValue = holdAltitude ? _roamAltitude : _groundClearance;
  _groundClearanceControl.stringValue =
      [NSString stringWithFormat:@"%.1f", _groundClearanceControl.doubleValue];
  [self updateMovementSpeedControl];
  [_panoramaView setMouseTurningEnabled:[self isMouseTurningEnabled] && !_viewerPaused];
  [_panoramaView setCruiseSteeringEnabled:cruising && !_viewerPaused];
  [self updateCruiseHUD];
  if (![self hasPressedRoamKey]) {
    [self updateMovementStatus];
  }
  [self updateMiniMapTelemetry];
}

- (void)updateCruiseHUD {
  [_panoramaView setCruiseHUDHeading:_orientation.heading
                               pitch:_orientation.pitch
                                bank:_aircraftBank
                 verticalFieldOfView:_verticalFieldOfView
                        aircraftMode:[self isAircraftDynamicsEnabled]];
}

- (void)movementModeChanged:(id)sender {
  (void)sender;
  [self clearRoamKeys];
  [self clearCruiseSteering];
  [self levelAircraft];
  if (_cruiseRecovery && ![self isCruisingEnabled]) {
    _cruiseRecovery = false;
    [_panoramaView setViewerPaused:_viewerPaused recoveryMessage:nil];
  }
  _roamDesiredPosition = {_observer.easting, _observer.northing};
  _roamAltitude = _observer.elevation;
  if ([self isCruisingEnabled]) {
    // Cruise is potentially fast and begins under continuous input. Enter it
    // in the safer absolute-altitude mode and require an explicit resume.
    _roamAltitudeModeControl.selectedSegment = 1;
    if ([self isAircraftDynamicsEnabled]) {
      _aircraftAirspeed = _cruiseSpeed;
    }
    _viewerPaused = true;
    _cruiseRecovery = false;
    [_panoramaView setViewerPaused:true recoveryMessage:nil];
  }
  [self updateRoamControls];
  [self scheduleRoamTimer];
  if ([self isCruisingEnabled]) {
    [_window makeFirstResponder:_panoramaView];
  }
}

- (void)roamTurningModeChanged:(id)sender {
  (void)sender;
  [self updateRoamControls];
}

- (void)roamMouseSensitivityChanged:(id)sender {
  (void)sender;
  _roamMouseSensitivityLabel.stringValue =
      [NSString stringWithFormat:@"%.2f×", _roamMouseSensitivityControl.doubleValue];
}

- (void)roamAltitudeModeChanged:(id)sender {
  (void)sender;
  [self clearRoamKeys];
  if (_roamAltitudeModeControl.selectedSegment == 0 && [self isAircraftDynamicsEnabled]) {
    _cruiseSpeed = _aircraftAirspeed;
    _aircraftDynamicsControl.state = NSControlStateValueOff;
    [self levelAircraft];
  }
  _roamDesiredPosition = {_observer.easting, _observer.northing};
  if (_roamAltitudeModeControl.selectedSegment == 1) {
    _roamAltitude = _observer.elevation;
  }
  [self updateRoamControls];
}

- (void)aircraftDynamicsChanged:(id)sender {
  (void)sender;
  if (_aircraftDynamicsControl.state == NSControlStateValueOn) {
    // Coordinated flight uses pitch to change absolute altitude and is not
    // compatible with the terrain-hugging Cruise mode.
    _roamAltitudeModeControl.selectedSegment = 1;
    _roamAltitude = _observer.elevation;
    _aircraftAirspeed = _cruiseSpeed;
    _aircraftBank = 0.0;
  } else {
    _cruiseSpeed = _aircraftAirspeed;
    [self levelAircraft];
  }
  _roamDesiredPosition = {_observer.easting, _observer.northing};
  [self updateRoamControls];
}

- (void)updateMovementSpeedControl {
  if (_roamSpeedControl == nil) {
    return;
  }
  if ([self isCruisingEnabled]) {
    const BOOL aircraft = [self isAircraftDynamicsEnabled];
    _roamSpeedControl.minValue = std::log10(panorama::app::kMinimumMovementSpeed);
    _roamSpeedControl.maxValue = std::log10(panorama::app::kMaximumCruiseSpeed);
    _roamSpeedControl.doubleValue = std::log10(_cruiseSpeed);
    _roamSpeedRowLabel.stringValue = aircraft ? @"Trim speed" : @"Speed";
    _roamSpeedControl.toolTip =
        aircraft ? @"Logarithmic target airspeed; climbs and dives change the actual airspeed"
                 : @"Logarithmic cruise speed from 3.6 km/h to 36,000 km/h";
    _roamSpeedLabel.stringValue = format_movement_speed(_cruiseSpeed);
  } else {
    _roamSpeedRowLabel.stringValue = @"Speed";
    _roamSpeedControl.minValue = panorama::app::kMinimumMovementSpeed;
    _roamSpeedControl.maxValue = panorama::app::kMaximumRoamSpeed;
    _roamSpeedControl.doubleValue = _roamSpeed;
    _roamSpeedControl.toolTip = @"Horizontal roaming speed";
    _roamSpeedLabel.stringValue = format_movement_speed(_roamSpeed);
  }
}

- (void)roamSpeedChanged:(id)sender {
  (void)sender;
  if ([self isCruisingEnabled]) {
    _cruiseSpeed = std::pow(10.0, _roamSpeedControl.doubleValue);
    [self updateMovementStatus];
  } else {
    _roamSpeed = _roamSpeedControl.doubleValue;
  }
  _roamSpeedLabel.stringValue =
      format_movement_speed([self isCruisingEnabled] ? _cruiseSpeed : _roamSpeed);
  [self updateMiniMapTelemetry];
}

- (void)roamUpdateRateChanged:(id)sender {
  (void)sender;
  _roamUpdateRateControl.doubleValue = std::round(_roamUpdateRateControl.doubleValue);
  _roamUpdateRateLabel.stringValue =
      [NSString stringWithFormat:@"%.0f Hz", _roamUpdateRateControl.doubleValue];
  [self scheduleRoamTimer];
}

- (void)rotateHeading:(double)headingDelta pitch:(double)pitchDelta {
  _orientation.heading =
      std::remainder(_orientation.heading + headingDelta, 2.0 * std::numbers::pi);
  constexpr double kPitchLimit = 85.0 * std::numbers::pi / 180.0;
  _orientation.pitch = std::clamp(_orientation.pitch + pitchDelta, -kPitchLimit, kPitchLimit);
  [self requestMetalFxInteraction];
  _renderer->request_view(_orientation, _verticalFieldOfView, _image);
  [self updateCruiseHUD];
  [self updateMiniMapTelemetry];
}

- (void)requestMetalFxInteraction {
  if (_metalfxActivation != panorama::app::MetalFxActivation::PanMoveOnly ||
      _metalfxPreset == panorama::app::MetalFxPreset::Off || !_renderer->metalfx_supported())
    return;
  [_metalfxSettleTimer invalidate];
  _renderer->request_metalfx(_metalfxActivation, _metalfxPreset, true);
  __weak PanoramaController *weakSelf = self;
  _metalfxSettleTimer = [NSTimer timerWithTimeInterval:0.2
                                               repeats:NO
                                                 block:^(NSTimer *timer) {
                                                   PanoramaController *controller = weakSelf;
                                                   if (controller != nil)
                                                     [controller settleMetalFx:timer];
                                                 }];
  [NSRunLoop.mainRunLoop addTimer:_metalfxSettleTimer forMode:NSRunLoopCommonModes];
}

- (void)settleMetalFx:(NSTimer *)timer {
  if (timer != _metalfxSettleTimer)
    return;
  _metalfxSettleTimer = nil;
  if (!_viewerPaused && !_cruiseRecovery && _window.isKeyWindow &&
      ([self isCruisingEnabled] || ([self isRoamingEnabled] && [self hasPressedRoamKey] &&
                                    _window.firstResponder == _panoramaView))) {
    [self requestMetalFxInteraction];
    return;
  }
  _renderer->request_metalfx(_metalfxActivation, _metalfxPreset, false);
}

- (void)persistMetalFxSettings {
  [NSUserDefaults.standardUserDefaults setInteger:static_cast<NSInteger>(_metalfxActivation)
                                           forKey:@"panorama.metalfx.activation"];
  [NSUserDefaults.standardUserDefaults setInteger:static_cast<NSInteger>(_metalfxPreset)
                                           forKey:@"panorama.metalfx.preset"];
}

- (void)metalfxChanged:(id)sender {
  (void)sender;
  _metalfxActivation =
      static_cast<panorama::app::MetalFxActivation>(_metalfxActivationControl.indexOfSelectedItem);
  _metalfxPreset =
      static_cast<panorama::app::MetalFxPreset>(_metalfxPresetControl.indexOfSelectedItem);
  [_metalfxSettleTimer invalidate];
  _metalfxSettleTimer = nil;
  _renderer->request_metalfx(_metalfxActivation, _metalfxPreset, false);
  [self persistMetalFxSettings];
  _metalfxStatusLabel.stringValue =
      _metalfxPreset == panorama::app::MetalFxPreset::Off ||
              _metalfxActivation == panorama::app::MetalFxActivation::Disabled
          ? @"Native resolution"
          : @"Applies on next frame";
}

- (void)selectRaytracer:(NSMenuItem *)sender {
  [_raytracerControl selectItemWithTag:sender.tag];
  _renderer->request_raytracer(static_cast<panorama::Raytracer>(sender.tag));
}

- (void)raytracerChanged:(NSPopUpButton *)sender {
  [self selectRaytracer:sender.selectedItem];
}

- (BOOL)validateMenuItem:(NSMenuItem *)item {
  if (item.action == @selector(selectRaytracer:)) {
    item.state = static_cast<panorama::Raytracer>(item.tag) == _renderer->requested_raytracer()
                     ? NSControlStateValueOn
                     : NSControlStateValueOff;
  }
  return YES;
}

- (void)rotateForCurrentZoomHeading:(double)headingDelta pitch:(double)pitchDelta {
  // Mouse and keyboard deltas define their desired feel at the default FOV.
  // Scaling by the current angular extent preserves that behaviour while
  // providing proportionally finer control over a magnified view.
  const double zoom_scale = _verticalFieldOfView / panorama::app::kDefaultVerticalFieldOfView;
  constexpr double kExistingSensitivity = 8.0;
  const double sensitivity_scale = _panningSensitivity / kExistingSensitivity;
  const double scale = zoom_scale * sensitivity_scale;
  [self rotateHeading:headingDelta * scale pitch:pitchDelta * scale];
}

- (void)panForCurrentZoomHeading:(double)headingDelta pitch:(double)pitchDelta {
  const double direction = _invertMousePanning ? -1.0 : 1.0;
  [self rotateForCurrentZoomHeading:headingDelta * direction pitch:pitchDelta * direction];
}

- (void)mouseTurnForCurrentZoomHeading:(double)headingDelta pitch:(double)pitchDelta {
  const double direction = _invertMousePanning ? -1.0 : 1.0;
  const double zoom_scale = _verticalFieldOfView / panorama::app::kDefaultVerticalFieldOfView;
  const double sensitivity =
      _roamMouseSensitivityControl == nil ? 1.0 : _roamMouseSensitivityControl.doubleValue;
  [self rotateHeading:headingDelta * direction * zoom_scale * sensitivity
                pitch:pitchDelta * direction * zoom_scale * sensitivity];
}

- (void)updateZoomControls {
  const double degrees = _verticalFieldOfView * panorama::app::kRadiansToDegrees;
  if (_zoomControl != nil) {
    _zoomControl.doubleValue = degrees;
  }
  if (_zoomValueLabel != nil) {
    _zoomValueLabel.stringValue = [NSString stringWithFormat:@"%.1f°", degrees];
  }
}

- (void)setVerticalFieldOfViewDegrees:(double)degrees {
  constexpr double kMinimumDegrees = 5.0;
  constexpr double kMaximumDegrees = 140.0;
  const double next =
      std::clamp(degrees, kMinimumDegrees, kMaximumDegrees) * panorama::app::kDegreesToRadians;
  if (std::abs(next - _verticalFieldOfView) <= 1e-12) {
    [self updateZoomControls];
    return;
  }
  _verticalFieldOfView = next;
  [self requestMetalFxInteraction];
  [self updateZoomControls];
  _renderer->request_view(_orientation, _verticalFieldOfView, _image);
  [self updateCruiseHUD];
}

- (void)zoomControlChanged:(NSSlider *)sender {
  double degrees = sender.doubleValue;
  // A small detent makes the original 70-degree view easy to recover while
  // leaving the remainder of the slider continuously adjustable.
  constexpr double kDefaultDetentDegrees = 70.0;
  constexpr double kDetentRadiusDegrees = 2.0;
  if (std::abs(degrees - kDefaultDetentDegrees) <= kDetentRadiusDegrees) {
    degrees = kDefaultDetentDegrees;
  }
  [self setVerticalFieldOfViewDegrees:degrees];
}

- (void)invertMousePanningChanged:(NSButton *)sender {
  _invertMousePanning = sender.state == NSControlStateValueOn;
}

- (void)panningSensitivityChanged:(NSSlider *)sender {
  _panningSensitivity = std::round(sender.doubleValue);
  sender.doubleValue = _panningSensitivity;
  _panningSensitivityLabel.stringValue = [NSString stringWithFormat:@"%.0f", _panningSensitivity];
}

- (void)setDaylightStatus:(NSString *)status {
  _daylightTimesLabel.stringValue = status;
  _daylightSymbolsRow.hidden = YES;
}

- (void)resolveObserverTimeZone {
  const uint64_t requestToken = ++_timeZoneRequestToken;
  _observerTimeZone = nil;
  _timeZoneLookupInProgress = true;
  [self setDaylightStatus:@"Finding observer time zone…"];
  [self updateSettingsControlAvailability];

  [_timeZoneRequest cancel];
  const panorama::LatLon geographic =
      _renderer->terrain_crs().to_lat_lon({_observer.easting, _observer.northing});
  CLLocation *location = [[CLLocation alloc] initWithLatitude:geographic.lat
                                                    longitude:geographic.lon];
  _timeZoneRequest = [[MKReverseGeocodingRequest alloc] initWithLocation:location];
  __weak PanoramaController *weakSelf = self;
  [_timeZoneRequest
      getMapItemsWithCompletionHandler:^(NSArray<MKMapItem *> *mapItems, NSError *error) {
        // Geocoding may finish after another observer move. Marshal UI
        // work to the main queue and discard superseded responses.
        dispatch_async(dispatch_get_main_queue(), ^{
          PanoramaController *strongSelf = weakSelf;
          if (strongSelf == nil || requestToken != strongSelf->_timeZoneRequestToken) {
            return;
          }
          strongSelf->_timeZoneLookupInProgress = false;
          NSTimeZone *timeZone = mapItems.firstObject.timeZone;
          if (error != nil || timeZone == nil) {
            [strongSelf setDaylightStatus:@"Observer time zone unavailable"];
            [strongSelf updateSettingsControlAvailability];
            return;
          }

          strongSelf->_observerTimeZone = timeZone;
          strongSelf->_astronomicalTimeControl.toolTip = [NSString
              stringWithFormat:@"Local time in %@ at one-minute resolution", timeZone.name];
          strongSelf->_daylightTimesLabel.toolTip =
              [NSString stringWithFormat:@"Local geometric-horizon crossings in %@", timeZone.name];

          // Populate the initial controls with the current civil time at
          // the observer. Later observer moves preserve the user's chosen
          // wall-clock date and time, but reinterpret them at the new site.
          if (!strongSelf->_astronomicalControlsUseObserverTime) {
            NSDate *now = [NSDate date];
            NSDateFormatter *dateFormatter = [[NSDateFormatter alloc] init];
            dateFormatter.locale = [[NSLocale alloc] initWithLocaleIdentifier:@"en_US_POSIX"];
            dateFormatter.timeZone = timeZone;
            dateFormatter.dateFormat = @"dd-MM-yyyy";
            strongSelf->_astronomicalDateControl.stringValue = [dateFormatter stringFromDate:now];

            NSCalendar *calendar =
                [[NSCalendar alloc] initWithCalendarIdentifier:NSCalendarIdentifierGregorian];
            calendar.timeZone = timeZone;
            NSDateComponents *components =
                [calendar components:NSCalendarUnitHour | NSCalendarUnitMinute fromDate:now];
            const double minutes = static_cast<double>(components.hour * 60 + components.minute);
            strongSelf->_astronomicalTimeControl.doubleValue = minutes;
            strongSelf->_astronomicalTimeLabel.stringValue =
                panorama::app::format_clock_minutes(minutes);
            strongSelf->_astronomicalControlsUseObserverTime = true;
          }

          [strongSelf updateSettingsControlAvailability];
          if (strongSelf->_sunModeControl.selectedSegment == 1) {
            [strongSelf publishAstronomicalLighting];
          } else {
            [strongSelf setDaylightStatus:[NSString stringWithFormat:@"Observer time · %@",
                                                                     timeZone.abbreviation]];
          }
        });
      }];
}

/// Publish the astronomical direction as grid azimuth and altitude without
/// retracing the terrain.
- (BOOL)publishAstronomicalLighting {
  if (_observerTimeZone == nil) {
    [self setDaylightStatus:_timeZoneLookupInProgress ? @"Finding observer time zone…"
                                                      : @"Observer time zone unavailable"];
    return NO;
  }

  const char *date = _astronomicalDateControl.stringValue.UTF8String;
  NSString *timeValue = panorama::app::format_clock_minutes(_astronomicalTimeControl.doubleValue);
  const char *time = timeValue.UTF8String;
  const std::optional<panorama::app::CalendarDateTime> local = panorama::app::parse_date_time(
      date == nullptr ? std::string_view{} : std::string_view(date),
      time == nullptr ? std::string_view{} : std::string_view(time)
  );
  _astronomicalDateControl.textColor =
      local.has_value() ? NSColor.labelColor : NSColor.systemRedColor;
  if (!local.has_value()) {
    [self setDaylightStatus:@"Enter a valid DD-MM-YYYY date"];
    return NO;
  }
  const std::optional<panorama::app::CalendarDateTime> utc =
      panorama::app::local_date_time_to_utc(*local, _observerTimeZone);
  if (!utc.has_value()) {
    [self setDaylightStatus:@"This local time does not exist"];
    return NO;
  }

  const panorama::app::DaylightTimes daylightInfo = panorama::app::daylight_times(
      _renderer->terrain_crs(),
      {_observer.easting, _observer.northing},
      *local
  );
  NSString *timeZoneSummary =
      panorama::app::format_time_zone_summary(_observerTimeZone, utc.value());
  switch (daylightInfo.state) {
  case panorama::app::DaylightState::Normal:
    _daylightTimesLabel.stringValue = timeZoneSummary;
    _sunriseTimeLabel.stringValue = panorama::app::format_local_daylight_time(
        *local,
        daylightInfo.sunrise_minutes,
        _observerTimeZone
    );
    _sunsetTimeLabel.stringValue = panorama::app::format_local_daylight_time(
        *local,
        daylightInfo.sunset_minutes,
        _observerTimeZone
    );
    _daylightSymbolsRow.hidden = NO;
    break;
  case panorama::app::DaylightState::PolarDay:
    [self setDaylightStatus:[NSString stringWithFormat:@"%@\nSun above horizon all day",
                                                       timeZoneSummary]];
    break;
  case panorama::app::DaylightState::PolarNight:
    [self setDaylightStatus:[NSString stringWithFormat:@"%@\nSun below horizon all day",
                                                       timeZoneSummary]];
    break;
  }

  const panorama::app::SolarPosition sun = panorama::app::solar_position(
      _renderer->terrain_crs(),
      {_observer.easting, _observer.northing},
      utc.value()
  );
  const double azimuthDegrees = sun.azimuth * panorama::app::kRadiansToDegrees;
  const double altitudeDegrees = sun.elevation * panorama::app::kRadiansToDegrees;
  _sunAzimuthControl.doubleValue = azimuthDegrees;
  _sunAltitudeControl.doubleValue = altitudeDegrees;
  _sunAzimuthLabel.stringValue = [NSString stringWithFormat:@"%.1f°", azimuthDegrees];
  _sunAltitudeLabel.stringValue = [NSString stringWithFormat:@"%.1f°", altitudeDegrees];
  _presentation.appearance.sun_azimuth = sun.azimuth;
  _presentation.appearance.sun_elevation = sun.elevation;
  _renderer->request_presentation(_presentation);
  return YES;
}

- (void)publishLightingControls {
  _presentation.appearance.sun_azimuth =
      _sunAzimuthControl.doubleValue * panorama::app::kDegreesToRadians;
  _presentation.appearance.sun_elevation =
      _sunAltitudeControl.doubleValue * panorama::app::kDegreesToRadians;
  _presentation.appearance.diffusivity = static_cast<float>(_diffusivityControl.doubleValue);
  _presentation.appearance.ambient_light = static_cast<float>(_skyStrengthControl.doubleValue);
  _presentation.appearance.ambient_detail = static_cast<float>(_skyDetailControl.doubleValue);
  _renderer->request_presentation(_presentation);
}

- (void)sunModeChanged:(NSSegmentedControl *)sender {
  [self updateSettingsControlAvailability];
  if (sender.selectedSegment == 1) {
    _manualSunAzimuthDegrees = _sunAzimuthControl.doubleValue;
    _manualSunAltitudeDegrees = _sunAltitudeControl.doubleValue;
    if (_observerTimeZone == nil) {
      if (!_timeZoneLookupInProgress) {
        [self resolveObserverTimeZone];
      }
    } else if (![self publishAstronomicalLighting]) {
      NSBeep();
    }
  } else {
    _sunAzimuthControl.doubleValue = _manualSunAzimuthDegrees;
    _sunAltitudeControl.doubleValue = _manualSunAltitudeDegrees;
    _sunAzimuthLabel.stringValue = [NSString stringWithFormat:@"%.0f°", _manualSunAzimuthDegrees];
    _sunAltitudeLabel.stringValue = [NSString stringWithFormat:@"%.0f°", _manualSunAltitudeDegrees];
    [self publishLightingControls];
  }
}

- (void)astronomicalInputChanged:(NSTextField *)sender {
  (void)sender;
  if (_sunModeControl.selectedSegment == 1 && ![self publishAstronomicalLighting]) {
    NSBeep();
  }
}

- (void)astronomicalTimeChanged:(NSSlider *)sender {
  sender.doubleValue = std::round(sender.doubleValue);
  _astronomicalTimeLabel.stringValue = panorama::app::format_clock_minutes(sender.doubleValue);
  [self updateSettingsControlAvailability];
  if (_sunModeControl.selectedSegment == 1) {
    [self publishAstronomicalLighting];
  }
}

- (void)adjustAstronomicalTime:(NSButton *)sender {
  const double minutes = std::clamp(
      std::round(_astronomicalTimeControl.doubleValue) + static_cast<double>(sender.tag),
      _astronomicalTimeControl.minValue,
      _astronomicalTimeControl.maxValue
  );
  _astronomicalTimeControl.doubleValue = minutes;
  _astronomicalTimeLabel.stringValue = panorama::app::format_clock_minutes(minutes);
  [self updateSettingsControlAvailability];
  if (_sunModeControl.selectedSegment == 1) {
    [self publishAstronomicalLighting];
  }
}

- (void)sunAzimuthChanged:(NSSlider *)sender {
  constexpr double kDetentRadiusDegrees = 3.0;
  if (std::abs(sender.doubleValue - panorama::app::kDefaultSunAzimuthDegrees) <=
      kDetentRadiusDegrees) {
    sender.doubleValue = panorama::app::kDefaultSunAzimuthDegrees;
  }
  _manualSunAzimuthDegrees = sender.doubleValue;
  _sunAzimuthLabel.stringValue = [NSString stringWithFormat:@"%.0f°", sender.doubleValue];
  [self publishLightingControls];
}

- (void)sunAltitudeChanged:(NSSlider *)sender {
  constexpr double kDetentRadiusDegrees = 2.0;
  if (std::abs(sender.doubleValue - panorama::app::kDefaultSunAltitudeDegrees) <=
      kDetentRadiusDegrees) {
    sender.doubleValue = panorama::app::kDefaultSunAltitudeDegrees;
  }
  _manualSunAltitudeDegrees = sender.doubleValue;
  _sunAltitudeLabel.stringValue = [NSString stringWithFormat:@"%.0f°", sender.doubleValue];
  [self publishLightingControls];
}

- (void)diffusivityChanged:(NSSlider *)sender {
  constexpr double kDetentRadius = 0.02;
  if (std::abs(sender.doubleValue - panorama::app::kDefaultDiffusivity) <= kDetentRadius) {
    sender.doubleValue = panorama::app::kDefaultDiffusivity;
  }
  _diffusivityLabel.stringValue = [NSString stringWithFormat:@"%.2f", sender.doubleValue];
  [self publishLightingControls];
}

- (void)skyStrengthChanged:(NSSlider *)sender {
  constexpr double kDetentRadius = 0.02;
  if (std::abs(sender.doubleValue - panorama::app::kDefaultSkyStrength) <= kDetentRadius) {
    sender.doubleValue = panorama::app::kDefaultSkyStrength;
  }
  _skyStrengthLabel.stringValue = [NSString stringWithFormat:@"%.2f", sender.doubleValue];
  [self publishLightingControls];
}

- (void)skyDetailChanged:(NSSlider *)sender {
  constexpr double kDetentRadius = 0.02;
  if (std::abs(sender.doubleValue - panorama::app::kDefaultSkyDetail) <= kDetentRadius) {
    sender.doubleValue = panorama::app::kDefaultSkyDetail;
  }
  _skyDetailLabel.stringValue = [NSString stringWithFormat:@"%.2f", sender.doubleValue];
  [self publishLightingControls];
}

- (void)normalLightingChanged:(NSButton *)sender {
  _presentation.use_surface_normals = sender.state == NSControlStateValueOn;
  [self updateSettingsControlAvailability];
  _renderer->request_presentation(_presentation);
}

- (void)bilinearCollisionChanged:(NSButton *)sender {
  _bilinearCollisions = sender.state == NSControlStateValueOn;
  _renderer->request_collision_settings(_bilinearCollisions, _c1Normals);
}

- (void)c1NormalsChanged:(NSButton *)sender {
  _c1Normals = sender.state == NSControlStateValueOn;
  _renderer->request_collision_settings(_bilinearCollisions, _c1Normals);
}

- (void)raytracedShadowsChanged:(NSButton *)sender {
  _presentation.appearance.raytraced_shadows = sender.state == NSControlStateValueOn;
  _renderer->request_presentation(_presentation);
}

/// Feature outlines are presentation-only, like lighting, so both controls
/// update the current trace immediately.
- (void)publishFeatureOutlineControls {
  _presentation.appearance.feature_outlines =
      _featureOutlinesControl.state == NSControlStateValueOn;
  _presentation.appearance.feature_outline_detail =
      static_cast<float>(_featureOutlineDetailControl.doubleValue / 10.0);
  _renderer->request_presentation(_presentation);
}

- (void)featureOutlinesChanged:(NSButton *)sender {
  (void)sender;
  [self updateSettingsControlAvailability];
  [self publishFeatureOutlineControls];
}

- (void)featureOutlineDetailChanged:(NSSlider *)sender {
  sender.doubleValue = std::round(sender.doubleValue);
  _featureOutlineDetailLabel.stringValue = [NSString stringWithFormat:@"%.0f", sender.doubleValue];
  [self publishFeatureOutlineControls];
}

/// LOD is a trace setting: zero retains LOD 1 everywhere, while positive
/// values permit a tile representation no wider than this pixel-footprint
/// multiplier.
- (void)lodScaleChanged:(NSSlider *)sender {
  const double scale = sender.doubleValue;
  _lodScaleLabel.stringValue = scale == 0.0 ? @"Off" : [NSString stringWithFormat:@"%.1f×", scale];
  _renderer->request_lod_scale(static_cast<float>(scale));
}

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

- (void)attachPanoramaView:(PanoramaView *)panoramaView
               overlayView:(ViewerOverlayView *)overlayView
             aspectFitView:(AspectFitContainerView *)aspectFitView
              miniMapPanel:(MiniMapPanelView *)miniMapPanel {
  _panoramaView = panoramaView;
  _overlayView = overlayView;
  _aspectFitView = aspectFitView;
  _miniMapPanel = miniMapPanel;
  _miniMapPanel.interactionDelegate = self;

  _pointInspectionEnabled = true;
  _renderer->request_minimap_enabled(true);
  [_panoramaView setPointInspectionEnabled:true];
  [_panoramaView setMouseTurningEnabled:[self isMouseTurningEnabled] && !_viewerPaused];
  [_panoramaView setCruiseSteeringEnabled:[self isCruisingEnabled] && !_viewerPaused];
  [self updateCruiseHUD];
  [_panoramaView setViewerPaused:_viewerPaused recoveryMessage:nil];
  [_overlayView setMapAndPointInfoVisible:true];
  [self updateMiniMapTelemetry];
  [_miniMapPanel informationFooterContentDidChange];
}

- (void)inspectLocationX:(double)x y:(double)y {
  if (_pointerOwner == panorama::app::PointerOwner::Panorama && _pointInspectionEnabled &&
      !_pointInspectionLocked && !_pointLockPending) {
    _inspectionRequestToken =
        _renderer->request_inspection(panorama::app::InspectionLocation{x, y});
  }
}

- (void)invalidatePanoramaHover {
  _inspectionRequestToken = _renderer->request_inspection(std::nullopt);
  if (!_pointInspectionLocked && !_pointLockPending) {
    [self updatePointInfo:std::nullopt];
  }
}

- (void)clearTargetVisibility {
  _targetVisibilityRequestToken = _renderer->request_target_visibility(std::nullopt);
  _targetVisibilityRevision = 0U;
  _lockedPointOccluded = false;
}

- (void)requestTargetVisibilityForPoint:(panorama::app::TerrainPoint)point {
  _targetVisibilityRevision = 0U;
  _lockedPointOccluded = false;
  _targetVisibilityRequestToken = _renderer->request_target_visibility(point);
}

- (void)clearMapHover {
  if (_mapPointAction == panorama::app::MapPointAction::Hover) {
    _mapPointAction = panorama::app::MapPointAction::None;
  }
  _mapHoverPoint.reset();
  if (!_pointInspectionLocked && !_pointLockPending) {
    [self updatePointInfo:std::nullopt];
  }
  [self updateLockedPointIndicatorWithOrientation:_orientation
                              verticalFieldOfView:_verticalFieldOfView
                                            image:_image];
}

- (void)pointerMovedOverPanorama {
  if (_pointerOwner == panorama::app::PointerOwner::Panorama) {
    return;
  }
  if (_pointerOwner == panorama::app::PointerOwner::Minimap) {
    [self clearMapHover];
  }
  _pointerOwner = panorama::app::PointerOwner::Panorama;
}

- (void)beginMinimapPointerOwnership {
  [self clearCruiseSteering];
  if (_pointerOwner == panorama::app::PointerOwner::Minimap) {
    return;
  }
  [self invalidatePanoramaHover];
  _pointerOwner = panorama::app::PointerOwner::Minimap;
}

- (void)pointerMovedOverOccludingView:(NSView *)view {
  [self clearCruiseSteering];
  const bool minimap = view == _miniMapPanel || [view isDescendantOf:_miniMapPanel];
  if (minimap) {
    [self beginMinimapPointerOwnership];
    return;
  }
  if (_pointerOwner == panorama::app::PointerOwner::Overlay) {
    return;
  }
  if (_pointerOwner == panorama::app::PointerOwner::Minimap) {
    [self clearMapHover];
  }
  [self invalidatePanoramaHover];
  _pointerOwner = panorama::app::PointerOwner::Overlay;
}

- (void)panoramaPointerExited {
  [self clearCruiseSteering];
  if (_pointerOwner != panorama::app::PointerOwner::Panorama) {
    return;
  }
  _pointerOwner = panorama::app::PointerOwner::None;
  [self invalidatePanoramaHover];
}

- (void)togglePointLockAtLocationX:(double)x y:(double)y {
  if (!_pointInspectionEnabled) {
    return;
  }
  if (_pointInspectionLocked || _pointLockPending) {
    _pointInspectionLocked = false;
    _pointLockPending = false;
    _lockedPoint.reset();
    [self clearTargetVisibility];
    [_miniMapPanel clearInspectedPoint];
    [_panoramaView setTerrainPointIndicator:std::nullopt image:{} locked:true occluded:false];
    [self setPointInfoStatus:@""];
    _inspectionRequestToken =
        _renderer->request_inspection(panorama::app::InspectionLocation{x, y});
    return;
  }

  _pointLockPending = true;
  [self setPointInfoStatus:@"Locking point…"];
  _pointLockRequestToken = _renderer->request_inspection(panorama::app::InspectionLocation{x, y});
}

- (void)miniMapPanel:(MiniMapPanelView *)panel
     didHoverEasting:(double)easting
            northing:(double)northing {
  (void)panel;
  [self beginMinimapPointerOwnership];
  if (!_pointInspectionLocked && !_pointLockPending &&
      (_mapPointAction == panorama::app::MapPointAction::None ||
       _mapPointAction == panorama::app::MapPointAction::Hover)) {
    [self requestMapPointEasting:easting
                        northing:northing
                          action:panorama::app::MapPointAction::Hover];
  }
}

- (void)miniMapPanelDidEndHover:(MiniMapPanelView *)panel {
  (void)panel;
  if (_pointerOwner == panorama::app::PointerOwner::Minimap) {
    _pointerOwner = panorama::app::PointerOwner::None;
  }
  [self clearMapHover];
}

- (void)requestMapPointEasting:(double)easting
                      northing:(double)northing
                        action:(panorama::app::MapPointAction)action {
  if (!_pointInspectionEnabled && action != panorama::app::MapPointAction::MoveObserver) {
    return;
  }
  _mapPointAction = action;
  _mapPointRequestToken = _renderer->request_map_point({easting, northing});
  if (action != panorama::app::MapPointAction::Hover) {
    [self setPointInfoStatus:action == panorama::app::MapPointAction::MoveObserver
                                 ? @"Moving observer…"
                                 : @"Locating point…"];
  }
}

- (void)miniMapPanel:(MiniMapPanelView *)panel
    didSelectEasting:(double)easting
            northing:(double)northing {
  (void)panel;
  _coordinateMovePending = false;
  [self beginMinimapPointerOwnership];
  [self requestMapPointEasting:easting
                      northing:northing
                        action:panorama::app::MapPointAction::Look];
}

- (void)miniMapPanel:(MiniMapPanelView *)panel
    didRequestObserverMoveToEasting:(double)easting
                           northing:(double)northing {
  (void)panel;
  _coordinateMovePending = false;
  [self beginMinimapPointerOwnership];
  [self requestMapPointEasting:easting
                      northing:northing
                        action:panorama::app::MapPointAction::MoveObserver];
}

- (void)moveObserverToTerrainPoint:(panorama::app::TerrainPoint)point {
  [self clearRoamKeys];
  _mapHoverPoint.reset();
  _pointInspectionLocked = false;
  _pointLockPending = false;
  _lockedPoint.reset();
  [self clearTargetVisibility];
  [_miniMapPanel clearInspectedPoint];
  [_panoramaView setTerrainPointIndicator:std::nullopt image:{} locked:true occluded:false];
  [self setPointInfoStatus:@"Moving observer…"];
  [self requestMetalFxInteraction];
  _renderer->request_observer_at(point, _groundClearance);
}

- (void)moveToLockedPoint:(id)sender {
  (void)sender;
  if (!_pointInspectionLocked || !_lockedPoint.has_value()) {
    NSBeep();
    return;
  }
  const panorama::app::PointInspection point = *_lockedPoint;
  const panorama::app::TerrainPoint target = {
      point.easting,
      point.northing,
      point.elevation,
  };
  [self moveObserverToTerrainPoint:target];
}

- (void)lookAtTerrainPoint:(panorama::app::TerrainPoint)point {
  const double east = point.easting - _observer.easting;
  const double north = point.northing - _observer.northing;
  const double horizontal = std::hypot(east, north);
  if (!(horizontal > 0.0)) {
    return;
  }
  const double apparentUp = point.elevation - _observer.elevation -
                            panorama::kCurvatureCoefficient * horizontal * horizontal;
  _orientation.heading = std::atan2(east, north);
  _orientation.pitch = std::atan2(apparentUp, horizontal);
  _renderer->request_view(_orientation, _verticalFieldOfView, _image);
  [self updateCruiseHUD];
}

- (void)toggleMapAndPointInspection:(id)sender {
  _pointInspectionEnabled = !_pointInspectionEnabled;
  _renderer->request_minimap_enabled(_pointInspectionEnabled);
  _pointInspectionLocked = false;
  _pointLockPending = false;
  _pointerOwner = panorama::app::PointerOwner::None;
  _mapPointAction = panorama::app::MapPointAction::None;
  _lockedPoint.reset();
  [self clearTargetVisibility];
  [_miniMapPanel clearInspectedPoint];
  _mapHoverPoint.reset();
  [_panoramaView setTerrainPointIndicator:std::nullopt image:{} locked:true occluded:false];
  [_panoramaView setPointInspectionEnabled:_pointInspectionEnabled];
  [_overlayView setMapAndPointInfoVisible:_pointInspectionEnabled];
  if (!_pointInspectionEnabled) {
    [self invalidatePanoramaHover];
  }
  [self updatePointInfo:std::nullopt];
  [self updateMiniMapTelemetry];

  if ([sender isKindOfClass:NSToolbarItem.class]) {
    NSToolbarItem *item = sender;
    if (@available(macOS 26.0, *)) {
      item.style = _pointInspectionEnabled ? NSToolbarItemStyleProminent : NSToolbarItemStylePlain;
    }
  }
}

- (BOOL)isMapAndPointInspectionEnabled {
  return _pointInspectionEnabled;
}

/// Build the controls shown in the trailing render-settings inspector.
- (NSViewController *)makeSettingsViewController {
  NSViewController *viewController = [[NSViewController alloc] init];
  NSScrollView *scrollView =
      [[NSScrollView alloc] initWithFrame:NSMakeRect(0.0, 0.0, 270.0, 400.0)];
  scrollView.borderType = NSNoBorder;
  scrollView.drawsBackground = NO;
  scrollView.hasHorizontalScroller = NO;
  scrollView.hasVerticalScroller = YES;
  scrollView.autohidesScrollers = YES;
  scrollView.scrollerStyle = NSScrollerStyleOverlay;
  viewController.view = scrollView;

  NSTextField *heading = [NSTextField labelWithString:@"Viewer Settings"];
  heading.font = [NSFont boldSystemFontOfSize:NSFont.systemFontSize];

  _zoomControl = [NSSlider sliderWithValue:_verticalFieldOfView * panorama::app::kRadiansToDegrees
                                  minValue:0.0
                                  maxValue:140.0
                                    target:self
                                    action:@selector(zoomControlChanged:)];
  _zoomControl.continuous = YES;
  _zoomControl.numberOfTickMarks = 3;
  _zoomControl.allowsTickMarkValuesOnly = NO;
  _zoomValueLabel = [NSTextField labelWithString:@""];
  _zoomValueLabel.alignment = NSTextAlignmentRight;
  [_zoomValueLabel.widthAnchor constraintEqualToConstant:39.0].active = YES;
  NSStackView *zoomSetting = [NSStackView stackViewWithViews:@[ _zoomControl, _zoomValueLabel ]];
  zoomSetting.orientation = NSUserInterfaceLayoutOrientationHorizontal;
  zoomSetting.alignment = NSLayoutAttributeCenterY;
  zoomSetting.spacing = 6.0;
  [self updateZoomControls];

  _panningSensitivityControl = [NSSlider sliderWithValue:_panningSensitivity
                                                minValue:1.0
                                                maxValue:10.0
                                                  target:self
                                                  action:@selector(panningSensitivityChanged:)];
  _panningSensitivityControl.continuous = YES;
  _panningSensitivityControl.numberOfTickMarks = 10;
  _panningSensitivityControl.allowsTickMarkValuesOnly = YES;
  _panningSensitivityControl.toolTip = @"Browse-mode drag and keyboard turning sensitivity";
  _panningSensitivityLabel = [NSTextField labelWithString:@"8"];
  _panningSensitivityLabel.alignment = NSTextAlignmentRight;
  [_panningSensitivityLabel.widthAnchor constraintEqualToConstant:18.0].active = YES;
  NSStackView *panningSensitivitySetting =
      [NSStackView stackViewWithViews:@[ _panningSensitivityControl, _panningSensitivityLabel ]];
  panningSensitivitySetting.orientation = NSUserInterfaceLayoutOrientationHorizontal;
  panningSensitivitySetting.alignment = NSLayoutAttributeCenterY;
  panningSensitivitySetting.spacing = 6.0;

  _imageWidthControl = [[NSTextField alloc] initWithFrame:NSZeroRect];
  _imageWidthControl.stringValue = [NSString stringWithFormat:@"%u", _image.width];
  _imageWidthControl.delegate = self;
  _imageHeightControl = [[NSTextField alloc] initWithFrame:NSZeroRect];
  _imageHeightControl.stringValue = [NSString stringWithFormat:@"%u", _image.height];
  _imageHeightControl.delegate = self;
  // Leave comfortable edit padding around common four-digit dimensions such
  // as 1920 and 1024 instead of sizing the fields to their initial values.
  [_imageWidthControl.widthAnchor constraintEqualToConstant:56.0].active = YES;
  [_imageHeightControl.widthAnchor constraintEqualToConstant:56.0].active = YES;
  NSTextField *resolutionSeparator = [NSTextField labelWithString:@"×"];

  _aspectLockControl = [[NSButton alloc] initWithFrame:NSZeroRect];
  _aspectLockControl.buttonType = NSButtonTypeToggle;
  _aspectLockControl.state = NSControlStateValueOn;
  _aspectLockControl.title = @"";
  _aspectLockControl.bordered = NO;
  _aspectLockControl.imagePosition = NSImageOnly;
  _aspectLockControl.target = self;
  _aspectLockControl.action = @selector(aspectLockChanged:);
  _aspectLockControl.toolTip = @"Keep width and height at the current aspect ratio";
  [_aspectLockControl setAccessibilityLabel:@"Lock aspect ratio"];
  [_aspectLockControl.widthAnchor constraintEqualToConstant:20.0].active = YES;
  [self updateAspectLockAppearance];

  NSStackView *resolutionSetting = [NSStackView stackViewWithViews:@[
    _imageWidthControl,
    resolutionSeparator,
    _imageHeightControl,
    _aspectLockControl,
  ]];
  resolutionSetting.orientation = NSUserInterfaceLayoutOrientationHorizontal;
  resolutionSetting.alignment = NSLayoutAttributeCenterY;
  resolutionSetting.spacing = 4.0;

  _matchWindowControl = [NSButton buttonWithTitle:@"Match Window"
                                           target:self
                                           action:@selector(matchWindowResolution:)];
  _matchWindowControl.image =
      [NSImage imageWithSystemSymbolName:@"arrow.left.and.right"
                accessibilityDescription:@"Match horizontal resolution to window"];
  _matchWindowControl.imagePosition = NSImageLeading;
  _matchWindowControl.toolTip =
      @"Change horizontal resolution to match the window; keep vertical resolution fixed";

  _metalfxActivationControl = [[NSPopUpButton alloc] initWithFrame:NSZeroRect pullsDown:NO];
  [_metalfxActivationControl addItemsWithTitles:@[ @"Disabled", @"Pan/move only", @"Always" ]];
  [_metalfxActivationControl selectItemAtIndex:static_cast<NSInteger>(_metalfxActivation)];
  _metalfxActivationControl.target = self;
  _metalfxActivationControl.action = @selector(metalfxChanged:);
  _metalfxActivationControl.toolTip = @"When MetalFX uses the selected reduced render resolution";

  _metalfxPresetControl = [[NSPopUpButton alloc] initWithFrame:NSZeroRect pullsDown:NO];
  [_metalfxPresetControl
      addItemsWithTitles:@[ @"Off", @"Quality (75%)", @"Balanced (50%)", @"Performance (33%)" ]];
  [_metalfxPresetControl selectItemAtIndex:static_cast<NSInteger>(_metalfxPreset)];
  _metalfxPresetControl.target = self;
  _metalfxPresetControl.action = @selector(metalfxChanged:);
  _metalfxPresetControl.toolTip = @"Terrain resolution before MetalFX spatial upscaling";
  _metalfxStatusLabel = [NSTextField labelWithString:@"Native resolution"];
  _metalfxStatusLabel.font = [NSFont systemFontOfSize:NSFont.smallSystemFontSize];
  _metalfxStatusLabel.textColor = NSColor.secondaryLabelColor;
  if (!_renderer->metalfx_supported()) {
    _metalfxStatusLabel.stringValue = @"MetalFX unavailable on this GPU";
    _metalfxActivationControl.enabled = NO;
    _metalfxPresetControl.enabled = NO;
  }

  _invertMousePanningControl = [[NSButton alloc] initWithFrame:NSZeroRect];
  _invertMousePanningControl.buttonType = NSButtonTypeSwitch;
  _invertMousePanningControl.title = @"Invert drag direction";
  _invertMousePanningControl.state = NSControlStateValueOff;
  _invertMousePanningControl.target = self;
  _invertMousePanningControl.action = @selector(invertMousePanningChanged:);

  _raytracerControl = [[NSPopUpButton alloc] initWithFrame:NSZeroRect pullsDown:NO];
  for (const auto backend : {panorama::Raytracer::Software, panorama::Raytracer::MetalBvh}) {
    [_raytracerControl
        addItemWithTitle:backend == panorama::Raytracer::MetalBvh ? @"BVH" : @"Mipmap"];
    _raytracerControl.lastItem.tag = static_cast<NSInteger>(backend);
  }
  [_raytracerControl selectItemWithTag:static_cast<NSInteger>(_renderer->requested_raytracer())];
  _raytracerControl.target = self;
  _raytracerControl.action = @selector(raytracerChanged:);
  _raytracerControl.toolTip = @"Switch the terrain raytracing method and redraw the current view";

  _colourSourceControl = [[NSPopUpButton alloc] initWithFrame:NSZeroRect pullsDown:NO];
  [_colourSourceControl addItemsWithTitles:@[
    @"None (white)",
    @"Distance",
    @"Elevation",
    @"Traversal steps",
    @"Collision evaluations",
  ]];
  [_colourSourceControl
      selectItemAtIndex:static_cast<NSInteger>(_presentation.appearance.colour_source)];

  _colourmapControl = [[NSPopUpButton alloc] initWithFrame:NSZeroRect pullsDown:NO];
  [_colourmapControl addItemsWithTitles:@[
    @"Viridis",
    @"Plasma",
    @"Inferno",
    @"Magma",
    @"Cividis",
    @"Turbo",
    @"Viewfinder"
  ]];
  [_colourmapControl selectItemAtIndex:static_cast<NSInteger>(_presentation.appearance.colourmap)];

  _colourScaleControl = [[NSPopUpButton alloc] initWithFrame:NSZeroRect pullsDown:NO];
  [_colourScaleControl
      addItemsWithTitles:@[ @"Linear", @"Logarithmic", @"Square root", @"Quadratic" ]];
  [_colourScaleControl
      selectItemAtIndex:static_cast<NSInteger>(_presentation.appearance.colour_scale)];

  _minimumControl = [[NSTextField alloc] initWithFrame:NSZeroRect];
  _minimumControl.stringValue =
      panorama::app::format_range_value(_presentation.colour_range.minimum);
  _minimumControl.delegate = self;

  _maximumControl = [[NSTextField alloc] initWithFrame:NSZeroRect];
  _maximumControl.stringValue =
      panorama::app::format_range_value(_presentation.colour_range.maximum);
  _maximumControl.delegate = self;

  _bilinearCollisionControl = [[NSButton alloc] initWithFrame:NSZeroRect];
  _bilinearCollisionControl.buttonType = NSButtonTypeSwitch;
  _bilinearCollisionControl.title = @"Bilinear patches";
  _bilinearCollisionControl.state =
      _bilinearCollisions ? NSControlStateValueOn : NSControlStateValueOff;
  _bilinearCollisionControl.target = self;
  _bilinearCollisionControl.action = @selector(bilinearCollisionChanged:);
  _bilinearCollisionControl.toolTip =
      @"Compute terrain collisions with bilinear patches rather than split triangles";

  _featureOutlinesControl = [[NSButton alloc] initWithFrame:NSZeroRect];
  _featureOutlinesControl.buttonType = NSButtonTypeSwitch;
  _featureOutlinesControl.title = @"Feature outlines";
  _featureOutlinesControl.state =
      _presentation.appearance.feature_outlines ? NSControlStateValueOn : NSControlStateValueOff;
  _featureOutlinesControl.target = self;
  _featureOutlinesControl.action = @selector(featureOutlinesChanged:);
  _featureOutlinesControl.toolTip = @"Draw black lines at multiscale geometric surface separations";

  const double initialOutlineDetail = 10.0 * _presentation.appearance.feature_outline_detail;
  _featureOutlineDetailControl = [NSSlider sliderWithValue:initialOutlineDetail
                                                  minValue:0.0
                                                  maxValue:10.0
                                                    target:self
                                                    action:@selector(featureOutlineDetailChanged:)];
  _featureOutlineDetailControl.continuous = YES;
  _featureOutlineDetailControl.numberOfTickMarks = 11;
  _featureOutlineDetailControl.allowsTickMarkValuesOnly = YES;
  _featureOutlineDetailControl.toolTip = @"Higher values outline smaller surface separations";
  _featureOutlineDetailLabel =
      [NSTextField labelWithString:[NSString stringWithFormat:@"%.0f", initialOutlineDetail]];
  _featureOutlineDetailLabel.alignment = NSTextAlignmentRight;
  [_featureOutlineDetailLabel.widthAnchor constraintEqualToConstant:39.0].active = YES;
  NSStackView *featureOutlineDetailSetting = [NSStackView
      stackViewWithViews:@[ _featureOutlineDetailControl, _featureOutlineDetailLabel ]];
  featureOutlineDetailSetting.orientation = NSUserInterfaceLayoutOrientationHorizontal;
  featureOutlineDetailSetting.alignment = NSLayoutAttributeCenterY;
  featureOutlineDetailSetting.spacing = 6.0;

  const double initialLodScale = _renderer->initial_lod_scale();
  _lodScaleControl = [NSSlider sliderWithValue:initialLodScale
                                      minValue:0.0
                                      maxValue:8.0
                                        target:self
                                        action:@selector(lodScaleChanged:)];
  _lodScaleControl.numberOfTickMarks = 17; // 0.0, 0.2, 0.4, ... 8.0
  _lodScaleControl.allowsTickMarkValuesOnly = YES;
  // _lodScaleControl.continuous = YES;
  _lodScaleControl.toolTip =
      @"Use coarser independently stored terrain where a cell is smaller than a pixel";
  _lodScaleLabel =
      [NSTextField labelWithString:initialLodScale == 0.0
                                       ? @"Off"
                                       : [NSString stringWithFormat:@"%.1f×", initialLodScale]];
  _lodScaleLabel.alignment = NSTextAlignmentRight;
  [_lodScaleLabel.widthAnchor constraintEqualToConstant:39.0].active = YES;
  NSStackView *lodScaleSetting =
      [NSStackView stackViewWithViews:@[ _lodScaleControl, _lodScaleLabel ]];
  lodScaleSetting.orientation = NSUserInterfaceLayoutOrientationHorizontal;
  lodScaleSetting.alignment = NSLayoutAttributeCenterY;
  lodScaleSetting.spacing = 6.0;

  _normalLightingControl = [[NSButton alloc] initWithFrame:NSZeroRect];
  _normalLightingControl.buttonType = NSButtonTypeSwitch;
  _normalLightingControl.title = @"Surface shading";
  _normalLightingControl.state =
      _presentation.use_surface_normals ? NSControlStateValueOn : NSControlStateValueOff;
  _normalLightingControl.target = self;
  _normalLightingControl.action = @selector(normalLightingChanged:);

  _c1NormalsControl = [[NSButton alloc] initWithFrame:NSZeroRect];
  _c1NormalsControl.buttonType = NSButtonTypeSwitch;
  _c1NormalsControl.title = @"Smooth normals";
  _c1NormalsControl.state = _c1Normals ? NSControlStateValueOn : NSControlStateValueOff;
  _c1NormalsControl.target = self;
  _c1NormalsControl.action = @selector(c1NormalsChanged:);
  _c1NormalsControl.toolTip =
      @"Smooth normals across cell boundaries instead of using each patch independently";

  _raytracedShadowsControl = [[NSButton alloc] initWithFrame:NSZeroRect];
  _raytracedShadowsControl.buttonType = NSButtonTypeSwitch;
  _raytracedShadowsControl.title = @"Hard shadows";
  _raytracedShadowsControl.state =
      _presentation.appearance.raytraced_shadows ? NSControlStateValueOn : NSControlStateValueOff;
  _raytracedShadowsControl.target = self;
  _raytracedShadowsControl.action = @selector(raytracedShadowsChanged:);
  _raytracedShadowsControl.toolTip = @"Cast one terrain visibility ray towards the sun";

  _sunModeControl = [[NSSegmentedControl alloc] initWithFrame:NSZeroRect];
  _sunModeControl.segmentCount = 2;
  [_sunModeControl setLabel:@"Manual" forSegment:0];
  [_sunModeControl setLabel:@"Astronomical" forSegment:1];
  _sunModeControl.selectedSegment = 0;
  _sunModeControl.segmentStyle = NSSegmentStyleRounded;
  _sunModeControl.trackingMode = NSSegmentSwitchTrackingSelectOne;
  _sunModeControl.target = self;
  _sunModeControl.action = @selector(sunModeChanged:);

  NSDateFormatter *dateFormatter = [[NSDateFormatter alloc] init];
  dateFormatter.locale = [[NSLocale alloc] initWithLocaleIdentifier:@"en_US_POSIX"];
  dateFormatter.timeZone = [NSTimeZone timeZoneForSecondsFromGMT:0];
  NSDate *now = [NSDate date];
  dateFormatter.dateFormat = @"dd-MM-yyyy";
  _astronomicalDateControl = [[NSTextField alloc] initWithFrame:NSZeroRect];
  _astronomicalDateControl.stringValue = [dateFormatter stringFromDate:now];
  _astronomicalDateControl.placeholderString = @"DD-MM-YYYY";
  _astronomicalDateControl.delegate = self;
  _astronomicalDateControl.toolTip = @"Gregorian date in DD-MM-YYYY format";

  NSCalendar *utcCalendar =
      [[NSCalendar alloc] initWithCalendarIdentifier:NSCalendarIdentifierGregorian];
  utcCalendar.timeZone = [NSTimeZone timeZoneForSecondsFromGMT:0];
  const NSDateComponents *utcComponents =
      [utcCalendar components:NSCalendarUnitHour | NSCalendarUnitMinute fromDate:now];
  const double initialUtcMinutes =
      60.0 * static_cast<double>(utcComponents.hour) + static_cast<double>(utcComponents.minute);
  _astronomicalTimeControl = [NSSlider sliderWithValue:initialUtcMinutes
                                              minValue:0.0
                                              maxValue:1439.0
                                                target:self
                                                action:@selector(astronomicalTimeChanged:)];
  _astronomicalTimeControl.continuous = YES;
  _astronomicalTimeControl.numberOfTickMarks = 7;
  _astronomicalTimeControl.allowsTickMarkValuesOnly = NO;
  _astronomicalTimeControl.toolTip = @"Observer-local time at one-minute resolution";
  _astronomicalTimeLabel =
      [NSTextField labelWithString:panorama::app::format_clock_minutes(initialUtcMinutes)];
  _astronomicalTimeLabel.alignment = NSTextAlignmentRight;
  [_astronomicalTimeLabel.widthAnchor constraintEqualToConstant:39.0].active = YES;
  _astronomicalTimeDecreaseControl = [NSButton buttonWithTitle:@"−"
                                                        target:self
                                                        action:@selector(adjustAstronomicalTime:)];
  _astronomicalTimeDecreaseControl.tag = -1;
  _astronomicalTimeDecreaseControl.controlSize = NSControlSizeSmall;
  _astronomicalTimeDecreaseControl.continuous = YES;
  [_astronomicalTimeDecreaseControl setPeriodicDelay:0.4F interval:0.08F];
  _astronomicalTimeDecreaseControl.toolTip = @"Move back one minute";
  [_astronomicalTimeDecreaseControl setAccessibilityLabel:@"Decrease time by one minute"];
  [_astronomicalTimeDecreaseControl.widthAnchor constraintEqualToConstant:22.0].active = YES;
  _astronomicalTimeIncreaseControl = [NSButton buttonWithTitle:@"+"
                                                        target:self
                                                        action:@selector(adjustAstronomicalTime:)];
  _astronomicalTimeIncreaseControl.tag = 1;
  _astronomicalTimeIncreaseControl.controlSize = NSControlSizeSmall;
  _astronomicalTimeIncreaseControl.continuous = YES;
  [_astronomicalTimeIncreaseControl setPeriodicDelay:0.4F interval:0.08F];
  _astronomicalTimeIncreaseControl.toolTip = @"Move forward one minute";
  [_astronomicalTimeIncreaseControl setAccessibilityLabel:@"Increase time by one minute"];
  [_astronomicalTimeIncreaseControl.widthAnchor constraintEqualToConstant:22.0].active = YES;
  NSStackView *astronomicalTimeSlider = [NSStackView stackViewWithViews:@[
    _astronomicalTimeDecreaseControl,
    _astronomicalTimeControl,
    _astronomicalTimeIncreaseControl,
    _astronomicalTimeLabel,
  ]];
  astronomicalTimeSlider.orientation = NSUserInterfaceLayoutOrientationHorizontal;
  astronomicalTimeSlider.alignment = NSLayoutAttributeCenterY;
  astronomicalTimeSlider.spacing = 4.0;
  _daylightTimesLabel = [NSTextField labelWithString:@"Time zone —"];
  _daylightTimesLabel.font = [NSFont systemFontOfSize:NSFont.smallSystemFontSize];
  _daylightTimesLabel.textColor = NSColor.secondaryLabelColor;
  _daylightTimesLabel.maximumNumberOfLines = 2;
  _daylightTimesLabel.lineBreakMode = NSLineBreakByClipping;
  _daylightTimesLabel.toolTip = @"Local crossings of the geometric horizon";

  NSImageSymbolConfiguration *daylightSymbolConfiguration =
      [NSImageSymbolConfiguration configurationWithPointSize:NSFont.smallSystemFontSize
                                                      weight:NSFontWeightRegular];
  NSImageView *sunriseIcon = [NSImageView
      imageViewWithImage:[[NSImage imageWithSystemSymbolName:@"sunrise"
                                    accessibilityDescription:@"Sunrise"]
                             imageWithSymbolConfiguration:daylightSymbolConfiguration]];
  sunriseIcon.contentTintColor = NSColor.secondaryLabelColor;
  sunriseIcon.toolTip = @"Sunrise";
  [sunriseIcon setAccessibilityLabel:@"Sunrise"];
  [sunriseIcon.widthAnchor constraintEqualToConstant:15.0].active = YES;
  NSImageView *sunsetIcon = [NSImageView
      imageViewWithImage:[[NSImage imageWithSystemSymbolName:@"sunset"
                                    accessibilityDescription:@"Sunset"]
                             imageWithSymbolConfiguration:daylightSymbolConfiguration]];
  sunsetIcon.contentTintColor = NSColor.secondaryLabelColor;
  sunsetIcon.toolTip = @"Sunset";
  [sunsetIcon setAccessibilityLabel:@"Sunset"];
  [sunsetIcon.widthAnchor constraintEqualToConstant:15.0].active = YES;

  _sunriseTimeLabel = [NSTextField labelWithString:@"—"];
  _sunsetTimeLabel = [NSTextField labelWithString:@"—"];
  NSTextField *daylightSeparator = [NSTextField labelWithString:@"•"];
  for (NSTextField *label in @[ _sunriseTimeLabel, daylightSeparator, _sunsetTimeLabel ]) {
    label.font = [NSFont systemFontOfSize:NSFont.smallSystemFontSize];
    label.textColor = NSColor.secondaryLabelColor;
  }
  _daylightSymbolsRow = [NSStackView stackViewWithViews:@[
    sunriseIcon,
    _sunriseTimeLabel,
    daylightSeparator,
    sunsetIcon,
    _sunsetTimeLabel,
  ]];
  _daylightSymbolsRow.orientation = NSUserInterfaceLayoutOrientationHorizontal;
  _daylightSymbolsRow.alignment = NSLayoutAttributeCenterY;
  _daylightSymbolsRow.spacing = 4.0;

  NSStackView *astronomicalTimeSetting = [NSStackView stackViewWithViews:@[
    astronomicalTimeSlider,
    _daylightTimesLabel,
    _daylightSymbolsRow,
  ]];
  astronomicalTimeSetting.orientation = NSUserInterfaceLayoutOrientationVertical;
  astronomicalTimeSetting.alignment = NSLayoutAttributeLeading;
  astronomicalTimeSetting.spacing = 3.0;

  _sunAzimuthControl = [NSSlider
      sliderWithValue:_presentation.appearance.sun_azimuth * panorama::app::kRadiansToDegrees
             minValue:0.0
             maxValue:360.0
               target:self
               action:@selector(sunAzimuthChanged:)];
  _sunAzimuthControl.continuous = YES;
  _sunAzimuthControl.numberOfTickMarks = 9;
  _sunAzimuthControl.allowsTickMarkValuesOnly = NO;
  _sunAzimuthControl.toolTip = @"Clockwise from grid north";
  _sunAzimuthLabel = [NSTextField
      labelWithString:[NSString stringWithFormat:@"%.0f°", _sunAzimuthControl.doubleValue]];
  _sunAzimuthLabel.alignment = NSTextAlignmentRight;
  [_sunAzimuthLabel.widthAnchor constraintEqualToConstant:39.0].active = YES;
  NSStackView *sunAzimuthSetting =
      [NSStackView stackViewWithViews:@[ _sunAzimuthControl, _sunAzimuthLabel ]];
  sunAzimuthSetting.orientation = NSUserInterfaceLayoutOrientationHorizontal;
  sunAzimuthSetting.alignment = NSLayoutAttributeCenterY;
  sunAzimuthSetting.spacing = 6.0;
  _manualSunAzimuthDegrees = _sunAzimuthControl.doubleValue;

  const double initialAltitude =
      _presentation.appearance.sun_elevation * panorama::app::kRadiansToDegrees;
  _sunAltitudeControl = [NSSlider sliderWithValue:initialAltitude
                                         minValue:-90.0
                                         maxValue:90.0
                                           target:self
                                           action:@selector(sunAltitudeChanged:)];
  _sunAltitudeControl.continuous = YES;
  _sunAltitudeControl.numberOfTickMarks = 7;
  _sunAltitudeControl.allowsTickMarkValuesOnly = NO;
  _sunAltitudeControl.toolTip = @"Degrees above or below the horizon";
  _sunAltitudeLabel =
      [NSTextField labelWithString:[NSString stringWithFormat:@"%.0f°", initialAltitude]];
  _sunAltitudeLabel.alignment = NSTextAlignmentRight;
  [_sunAltitudeLabel.widthAnchor constraintEqualToConstant:39.0].active = YES;
  NSStackView *sunAltitudeSetting =
      [NSStackView stackViewWithViews:@[ _sunAltitudeControl, _sunAltitudeLabel ]];
  sunAltitudeSetting.orientation = NSUserInterfaceLayoutOrientationHorizontal;
  sunAltitudeSetting.alignment = NSLayoutAttributeCenterY;
  sunAltitudeSetting.spacing = 6.0;
  _manualSunAltitudeDegrees = _sunAltitudeControl.doubleValue;

  _diffusivityControl = [NSSlider sliderWithValue:_presentation.appearance.diffusivity
                                         minValue:0.0
                                         maxValue:1.0
                                           target:self
                                           action:@selector(diffusivityChanged:)];
  _diffusivityControl.continuous = YES;
  _diffusivityControl.numberOfTickMarks = 11;
  _diffusivityControl.allowsTickMarkValuesOnly = NO;
  _diffusivityControl.toolTip = @"Strength of directional diffuse lighting";
  _diffusivityLabel = [NSTextField
      labelWithString:[NSString stringWithFormat:@"%.2f", _diffusivityControl.doubleValue]];
  _diffusivityLabel.alignment = NSTextAlignmentRight;
  [_diffusivityLabel.widthAnchor constraintEqualToConstant:39.0].active = YES;
  NSStackView *diffusivitySetting =
      [NSStackView stackViewWithViews:@[ _diffusivityControl, _diffusivityLabel ]];
  diffusivitySetting.orientation = NSUserInterfaceLayoutOrientationHorizontal;
  diffusivitySetting.alignment = NSLayoutAttributeCenterY;
  diffusivitySetting.spacing = 6.0;

  _skyStrengthControl = [NSSlider sliderWithValue:_presentation.appearance.ambient_light
                                         minValue:0.0
                                         maxValue:1.0
                                           target:self
                                           action:@selector(skyStrengthChanged:)];
  _skyStrengthControl.continuous = YES;
  _skyStrengthControl.numberOfTickMarks = 11;
  _skyStrengthControl.allowsTickMarkValuesOnly = NO;
  _skyStrengthControl.toolTip = @"Overall strength of diffuse atmospheric light";
  _skyStrengthLabel = [NSTextField
      labelWithString:[NSString stringWithFormat:@"%.2f", _skyStrengthControl.doubleValue]];
  _skyStrengthLabel.alignment = NSTextAlignmentRight;
  [_skyStrengthLabel.widthAnchor constraintEqualToConstant:39.0].active = YES;
  NSStackView *skyStrengthSetting =
      [NSStackView stackViewWithViews:@[ _skyStrengthControl, _skyStrengthLabel ]];
  skyStrengthSetting.orientation = NSUserInterfaceLayoutOrientationHorizontal;
  skyStrengthSetting.alignment = NSLayoutAttributeCenterY;
  skyStrengthSetting.spacing = 6.0;

  _skyDetailControl = [NSSlider sliderWithValue:_presentation.appearance.ambient_detail
                                       minValue:0.0
                                       maxValue:1.0
                                         target:self
                                         action:@selector(skyDetailChanged:)];
  _skyDetailControl.continuous = YES;
  _skyDetailControl.numberOfTickMarks = 11;
  _skyDetailControl.allowsTickMarkValuesOnly = NO;
  _skyDetailControl.toolTip = @"Normal-dependent detail from five sampled sky directions";
  _skyDetailLabel = [NSTextField
      labelWithString:[NSString stringWithFormat:@"%.2f", _skyDetailControl.doubleValue]];
  _skyDetailLabel.alignment = NSTextAlignmentRight;
  [_skyDetailLabel.widthAnchor constraintEqualToConstant:39.0].active = YES;
  NSStackView *skyDetailSetting =
      [NSStackView stackViewWithViews:@[ _skyDetailControl, _skyDetailLabel ]];
  skyDetailSetting.orientation = NSUserInterfaceLayoutOrientationHorizontal;
  skyDetailSetting.alignment = NSLayoutAttributeCenterY;
  skyDetailSetting.spacing = 6.0;

  _colourSourceControl.target = self;
  _colourSourceControl.action = @selector(renderModeChanged:);
  _colourmapControl.target = self;
  _colourmapControl.action = @selector(renderModeChanged:);
  _colourScaleControl.target = self;
  _colourScaleControl.action = @selector(renderModeChanged:);

  auto make_row = [](NSString *title, NSView *control) {
    NSTextField *label = [NSTextField labelWithString:title];
    [label.widthAnchor constraintEqualToConstant:82.0].active = YES;
    // The 300-point panel has 268 points inside its horizontal margins.
    // Keep each row within that width instead of allowing controls to crowd
    // the trailing glass edge.
    [control.widthAnchor constraintGreaterThanOrEqualToConstant:178.0].active = YES;
    NSStackView *row = [NSStackView stackViewWithViews:@[ label, control ]];
    row.orientation = NSUserInterfaceLayoutOrientationHorizontal;
    row.alignment = NSLayoutAttributeCenterY;
    row.spacing = 8.0;
    return row;
  };

  NSStackView *rangeSetting = [NSStackView stackViewWithViews:@[
    _minimumControl,
    [NSTextField labelWithString:@"–"],
    _maximumControl,
  ]];
  rangeSetting.orientation = NSUserInterfaceLayoutOrientationHorizontal;
  rangeSetting.alignment = NSLayoutAttributeCenterY;
  rangeSetting.spacing = 5.0;
  [_minimumControl.widthAnchor constraintEqualToConstant:80.0].active = YES;
  [_maximumControl.widthAnchor constraintEqualToConstant:80.0].active = YES;

  NSView *colourmapRow = make_row(@"Colourmap", _colourmapControl);
  NSView *colourScaleRow = make_row(@"Scale", _colourScaleControl);
  NSView *colourRangeRow = make_row(@"Range", rangeSetting);
  _scalarColourRows = @[ colourmapRow, colourScaleRow, colourRangeRow ];
  _featureOutlineDetailRow = make_row(@"Detail", featureOutlineDetailSetting);
  NSView *lodScaleRow = make_row(@"LOD scale", lodScaleSetting);

  NSView *sunModeRow = make_row(@"Sun", _sunModeControl);
  NSView *dateRow = make_row(@"Date", _astronomicalDateControl);
  NSView *timeRow = make_row(@"Local time", astronomicalTimeSetting);
  NSView *azimuthRow = make_row(@"Azimuth", sunAzimuthSetting);
  NSView *altitudeRow = make_row(@"Altitude", sunAltitudeSetting);
  NSView *skyStrengthRow = make_row(@"Sky strength", skyStrengthSetting);
  NSView *skyDetailRow = make_row(@"Sky detail", skyDetailSetting);
  NSView *diffusivityRow = make_row(@"Sun strength", diffusivitySetting);
  _manualSunRows = @[ azimuthRow, altitudeRow ];
  _astronomicalSunRows = @[ dateRow, timeRow ];
  _normalLightingRows = @[
    _c1NormalsControl,
    _raytracedShadowsControl,
    sunModeRow,
    dateRow,
    timeRow,
    azimuthRow,
    altitudeRow,
    skyStrengthRow,
    skyDetailRow,
    diffusivityRow,
  ];

  InspectorSectionView *cameraSection =
      [[InspectorSectionView alloc] initWithTitle:@"Camera"
                                         controls:@[
                                           make_row(@"FOV", zoomSetting),
                                           make_row(@"Resolution", resolutionSetting),
                                           _matchWindowControl,
                                           make_row(@"MetalFX", _metalfxActivationControl),
                                           make_row(@"Preset", _metalfxPresetControl),
                                           _metalfxStatusLabel,
                                           make_row(@"Pan speed", panningSensitivitySetting),
                                           _invertMousePanningControl,
                                         ]
                                      defaultsKey:@"panorama.inspector.camera.expanded"];
  InspectorSectionView *terrainSection =
      [[InspectorSectionView alloc] initWithTitle:@"Terrain"
                                         controls:@[
                                           make_row(@"Raytracer", _raytracerControl),
                                           make_row(@"Colour by", _colourSourceControl),
                                           colourmapRow,
                                           colourScaleRow,
                                           colourRangeRow,
                                           lodScaleRow,
                                           _bilinearCollisionControl,
                                           _featureOutlinesControl,
                                           _featureOutlineDetailRow,
                                         ]
                                      defaultsKey:@"panorama.inspector.terrain.expanded"];
  InspectorSectionView *lightingSection =
      [[InspectorSectionView alloc] initWithTitle:@"Lighting"
                                         controls:@[
                                           _normalLightingControl,
                                           _c1NormalsControl,
                                           _raytracedShadowsControl,
                                           sunModeRow,
                                           dateRow,
                                           timeRow,
                                           azimuthRow,
                                           altitudeRow,
                                           skyStrengthRow,
                                           skyDetailRow,
                                           diffusivityRow,
                                         ]
                                      defaultsKey:@"panorama.inspector.lighting.expanded"];

  NSStackView *settings = [[NSStackView alloc] initWithFrame:NSZeroRect];
  for (NSView *view in @[ heading, cameraSection, terrainSection, lightingSection ]) {
    [settings addArrangedSubview:view];
  }
  settings.orientation = NSUserInterfaceLayoutOrientationVertical;
  settings.alignment = NSLayoutAttributeLeading;
  settings.spacing = 18.0;
  settings.edgeInsets = NSEdgeInsetsMake(20.0, 16.0, 20.0, 16.0);
  settings.translatesAutoresizingMaskIntoConstraints = NO;
  InspectorDocumentView *document = [[InspectorDocumentView alloc] initWithFrame:NSZeroRect];
  document.translatesAutoresizingMaskIntoConstraints = NO;
  [document addSubview:settings];
  scrollView.documentView = document;
  NSLayoutConstraint *viewportHeight =
      [document.heightAnchor constraintEqualToAnchor:scrollView.contentView.heightAnchor];
  // Prefer a viewport-height document when the controls are short. Its lower
  // priority lets the settings grow the document and enable scrolling in a
  // shorter window.
  viewportHeight.priority = NSLayoutPriorityDefaultLow;
  [NSLayoutConstraint activateConstraints:@[
    [document.widthAnchor constraintEqualToAnchor:scrollView.contentView.widthAnchor],
    [document.heightAnchor
        constraintGreaterThanOrEqualToAnchor:scrollView.contentView.heightAnchor],
    viewportHeight,
    [settings.topAnchor constraintEqualToAnchor:document.topAnchor],
    [settings.leadingAnchor constraintEqualToAnchor:document.leadingAnchor],
    [settings.trailingAnchor constraintEqualToAnchor:document.trailingAnchor],
    [settings.bottomAnchor constraintLessThanOrEqualToAnchor:document.bottomAnchor],
  ]];

  [self updateSettingsControlAvailability];
  [self resolveObserverTimeZone];
  return viewController;
}

/// Build observer-position controls separately from camera and presentation
/// settings. This pane is intentionally small for now; roaming controls can be
/// added here without crowding the minimap or the viewer tab.
- (NSViewController *)makePositioningViewController {
  NSViewController *viewController = [[NSViewController alloc] init];
  NSScrollView *scrollView =
      [[NSScrollView alloc] initWithFrame:NSMakeRect(0.0, 0.0, 270.0, 400.0)];
  scrollView.borderType = NSNoBorder;
  scrollView.drawsBackground = NO;
  scrollView.hasHorizontalScroller = NO;
  scrollView.hasVerticalScroller = YES;
  scrollView.autohidesScrollers = YES;
  scrollView.scrollerStyle = NSScrollerStyleOverlay;
  viewController.view = scrollView;

  NSTextField *heading = [NSTextField labelWithString:@"Position & Movement"];
  heading.font = [NSFont boldSystemFontOfSize:NSFont.systemFontSize];

  _coordinateSystemControl = [[NSPopUpButton alloc] initWithFrame:NSZeroRect pullsDown:NO];
  [_coordinateSystemControl addItemWithTitle:@"Auto"];
  _coordinateSystemControl.lastItem.tag = -1;
  [_coordinateSystemControl.menu addItem:NSMenuItem.separatorItem];
  [_coordinateSystemControl addItemWithTitle:@"Latitude / longitude"];
  _coordinateSystemControl.lastItem.tag =
      static_cast<NSInteger>(panorama::app::CoordinateInputSystem::Wgs84);
  [_coordinateSystemControl addItemWithTitle:@"Swiss LV95"];
  _coordinateSystemControl.lastItem.tag =
      static_cast<NSInteger>(panorama::app::CoordinateInputSystem::SwissLv95);
  [_coordinateSystemControl addItemWithTitle:@"OS National Grid"];
  _coordinateSystemControl.lastItem.tag =
      static_cast<NSInteger>(panorama::app::CoordinateInputSystem::BritishNationalGrid);
  [_coordinateSystemControl.menu addItem:NSMenuItem.separatorItem];
  [_coordinateSystemControl
      addItemWithTitle:[NSString
                           stringWithFormat:@"Dataset grid — %s", _renderer->terrain_crs().name()]];
  _coordinateSystemControl.lastItem.tag =
      static_cast<NSInteger>(panorama::app::CoordinateInputSystem::Terrain);
  _coordinateSystemControl.target = self;
  _coordinateSystemControl.action = @selector(coordinateSystemChanged:);
  _coordinateSystemControl.toolTip =
      @"Auto detects the coordinate system; choose one explicitly to resolve ambiguity";
  // Cap the row at the inspector's 268-point content width. Pop-up buttons use
  // their longest menu item as an intrinsic width; without this constraint the
  // dataset-grid title can force the whole inset stack beyond the panel edge.
  [_coordinateSystemControl.widthAnchor constraintEqualToConstant:178.0].active = YES;
  NSTextField *coordinateSystemLabel = [NSTextField labelWithString:@"System"];
  [coordinateSystemLabel.widthAnchor constraintEqualToConstant:82.0].active = YES;
  NSStackView *coordinateSystemRow =
      [NSStackView stackViewWithViews:@[ coordinateSystemLabel, _coordinateSystemControl ]];
  coordinateSystemRow.orientation = NSUserInterfaceLayoutOrientationHorizontal;
  coordinateSystemRow.alignment = NSLayoutAttributeCenterY;
  coordinateSystemRow.spacing = 8.0;

  _coordinateInputControl = [[NSTextField alloc] initWithFrame:NSZeroRect];
  _coordinateInputControl.delegate = self;
  _coordinateInputControl.placeholderString = @"Enter or paste a coordinate";
  _coordinateInputControl.toolTip = @"The coordinate system will be detected automatically";
  _coordinateInputControl.target = self;
  _coordinateInputControl.action = @selector(moveToCoordinate:);
  [_coordinateInputControl.widthAnchor constraintEqualToConstant:178.0].active = YES;
  NSTextField *coordinateLabel = [NSTextField labelWithString:@"Coordinate"];
  [coordinateLabel.widthAnchor constraintEqualToConstant:82.0].active = YES;
  NSStackView *coordinateRow =
      [NSStackView stackViewWithViews:@[ coordinateLabel, _coordinateInputControl ]];
  coordinateRow.orientation = NSUserInterfaceLayoutOrientationHorizontal;
  coordinateRow.alignment = NSLayoutAttributeCenterY;
  coordinateRow.spacing = 8.0;

  _coordinateStatusLabel = [NSTextField labelWithString:@"Format will be detected automatically"];
  _coordinateStatusLabel.font = [NSFont systemFontOfSize:NSFont.smallSystemFontSize];
  _coordinateStatusLabel.textColor = NSColor.secondaryLabelColor;
  _coordinateStatusLabel.maximumNumberOfLines = 2;
  _coordinateStatusLabel.lineBreakMode = NSLineBreakByWordWrapping;
  [_coordinateStatusLabel.widthAnchor constraintEqualToConstant:268.0].active = YES;

  _coordinateMoveControl = [NSButton buttonWithTitle:@"Move"
                                              target:self
                                              action:@selector(moveToCoordinate:)];
  _coordinateMoveControl.image = [NSImage imageWithSystemSymbolName:@"location.fill"
                                           accessibilityDescription:@"Move observer to coordinate"];
  _coordinateMoveControl.imagePosition = NSImageLeading;
  _coordinateMoveControl.enabled = NO;
  NSView *coordinateSpacer = [[NSView alloc] initWithFrame:NSZeroRect];
  [coordinateSpacer setContentHuggingPriority:NSLayoutPriorityDefaultLow
                               forOrientation:NSLayoutConstraintOrientationHorizontal];
  [coordinateSpacer
      setContentCompressionResistancePriority:NSLayoutPriorityDefaultLow
                               forOrientation:NSLayoutConstraintOrientationHorizontal];
  NSStackView *coordinateActionRow =
      [NSStackView stackViewWithViews:@[ coordinateSpacer, _coordinateMoveControl ]];
  coordinateActionRow.orientation = NSUserInterfaceLayoutOrientationHorizontal;
  coordinateActionRow.alignment = NSLayoutAttributeCenterY;
  [coordinateActionRow.widthAnchor constraintEqualToConstant:268.0].active = YES;

  InspectorSectionView *destinationSection =
      [[InspectorSectionView alloc] initWithTitle:@"Destination"
                                         controls:@[
                                           coordinateSystemRow,
                                           coordinateRow,
                                           _coordinateStatusLabel,
                                           coordinateActionRow,
                                         ]
                                      defaultsKey:@"panorama.inspector.destination.expanded"];

  const auto makeMovementRow = [](NSString *title, NSView *control) {
    NSTextField *label = [NSTextField labelWithString:title];
    [label.widthAnchor constraintEqualToConstant:82.0].active = YES;
    [control.widthAnchor constraintEqualToConstant:178.0].active = YES;
    NSStackView *row = [NSStackView stackViewWithViews:@[ label, control ]];
    row.orientation = NSUserInterfaceLayoutOrientationHorizontal;
    row.alignment = NSLayoutAttributeCenterY;
    row.spacing = 8.0;
    return row;
  };

  _movementModeControl = [[NSSegmentedControl alloc] initWithFrame:NSZeroRect];
  _movementModeControl.segmentCount = 3;
  [_movementModeControl setLabel:@"Browse" forSegment:0];
  [_movementModeControl setLabel:@"Roam" forSegment:1];
  [_movementModeControl setLabel:@"Cruise" forSegment:2];
  _movementModeControl.selectedSegment = 0;
  _movementModeControl.trackingMode = NSSegmentSwitchTrackingSelectOne;
  _movementModeControl.target = self;
  _movementModeControl.action = @selector(movementModeChanged:);
  _movementModeControl.toolTip = @"Browse looks around; Roam uses WASD; Cruise moves forward "
                                  "continuously under mouse control";
  NSView *movementModeRow = makeMovementRow(@"Mode", _movementModeControl);

  _roamTurningModeControl = [[NSSegmentedControl alloc] initWithFrame:NSZeroRect];
  _roamTurningModeControl.segmentCount = 2;
  [_roamTurningModeControl setLabel:@"Arrow keys" forSegment:0];
  [_roamTurningModeControl setLabel:@"Mouse" forSegment:1];
  _roamTurningModeControl.selectedSegment = 0;
  _roamTurningModeControl.trackingMode = NSSegmentSwitchTrackingSelectOne;
  _roamTurningModeControl.target = self;
  _roamTurningModeControl.action = @selector(roamTurningModeChanged:);
  _roamTurningModeControl.toolTip =
      @"Turn with the arrow keys or by moving the pointer over the panorama";
  _roamTurningModeRow = makeMovementRow(@"Turning", _roamTurningModeControl);

  _roamMouseSensitivityControl = [NSSlider sliderWithValue:1.0
                                                  minValue:0.25
                                                  maxValue:3.0
                                                    target:self
                                                    action:@selector(roamMouseSensitivityChanged:)];
  _roamMouseSensitivityControl.continuous = YES;
  _roamMouseSensitivityControl.toolTip = @"Mouse turning sensitivity";
  _roamMouseSensitivityLabel = [NSTextField labelWithString:@"1.00×"];
  _roamMouseSensitivityLabel.alignment = NSTextAlignmentRight;
  [_roamMouseSensitivityLabel.widthAnchor constraintEqualToConstant:54.0].active = YES;
  NSStackView *roamMouseSensitivitySetting = [NSStackView
      stackViewWithViews:@[ _roamMouseSensitivityControl, _roamMouseSensitivityLabel ]];
  roamMouseSensitivitySetting.orientation = NSUserInterfaceLayoutOrientationHorizontal;
  roamMouseSensitivitySetting.alignment = NSLayoutAttributeCenterY;
  roamMouseSensitivitySetting.spacing = 6.0;
  _roamMouseSensitivityRow = makeMovementRow(@"Sensitivity", roamMouseSensitivitySetting);

  _roamAltitudeModeControl = [[NSSegmentedControl alloc] initWithFrame:NSZeroRect];
  _roamAltitudeModeControl.segmentCount = 2;
  [_roamAltitudeModeControl setLabel:@"Terrain" forSegment:0];
  [_roamAltitudeModeControl setLabel:@"Altitude" forSegment:1];
  _roamAltitudeModeControl.selectedSegment = 0;
  _roamAltitudeModeControl.trackingMode = NSSegmentSwitchTrackingSelectOne;
  _roamAltitudeModeControl.target = self;
  _roamAltitudeModeControl.action = @selector(roamAltitudeModeChanged:);
  _roamAltitudeModeControl.toolTip =
      @"Maintain height above terrain or hold absolute elevation while moving";
  NSView *roamAltitudeRow = makeMovementRow(@"Height mode", _roamAltitudeModeControl);

  _aircraftDynamicsControl = [NSButton checkboxWithTitle:@"Enabled"
                                                  target:self
                                                  action:@selector(aircraftDynamicsChanged:)];
  _aircraftDynamicsControl.toolTip =
      @"Use coordinated banked turns and exchange airspeed with climbs and dives";
  _aircraftDynamicsRow = makeMovementRow(@"Aircraft", _aircraftDynamicsControl);

  _roamSpeedControl = [NSSlider sliderWithValue:panorama::app::kDefaultRoamSpeed
                                       minValue:panorama::app::kMinimumMovementSpeed
                                       maxValue:panorama::app::kMaximumRoamSpeed
                                         target:self
                                         action:@selector(roamSpeedChanged:)];
  _roamSpeedControl.continuous = YES;
  _roamSpeedControl.toolTip = @"Horizontal roaming speed";
  _roamSpeedLabel = [NSTextField labelWithString:@"72.0 km/h"];
  _roamSpeedLabel.alignment = NSTextAlignmentRight;
  [_roamSpeedLabel.widthAnchor constraintEqualToConstant:62.0].active = YES;
  NSStackView *roamSpeedSetting =
      [NSStackView stackViewWithViews:@[ _roamSpeedControl, _roamSpeedLabel ]];
  roamSpeedSetting.orientation = NSUserInterfaceLayoutOrientationHorizontal;
  roamSpeedSetting.alignment = NSLayoutAttributeCenterY;
  roamSpeedSetting.spacing = 6.0;
  _roamSpeedRowLabel = [NSTextField labelWithString:@"Speed"];
  [_roamSpeedRowLabel.widthAnchor constraintEqualToConstant:82.0].active = YES;
  [roamSpeedSetting.widthAnchor constraintEqualToConstant:178.0].active = YES;
  NSStackView *roamSpeedRow =
      [NSStackView stackViewWithViews:@[ _roamSpeedRowLabel, roamSpeedSetting ]];
  roamSpeedRow.orientation = NSUserInterfaceLayoutOrientationHorizontal;
  roamSpeedRow.alignment = NSLayoutAttributeCenterY;
  roamSpeedRow.spacing = 8.0;

  _roamUpdateRateControl = [NSSlider sliderWithValue:30.0
                                            minValue:1.0
                                            maxValue:60.0
                                              target:self
                                              action:@selector(roamUpdateRateChanged:)];
  _roamUpdateRateControl.continuous = YES;
  _roamUpdateRateControl.toolTip =
      @"Maximum observer-position requests per second; rendering may complete more slowly";
  _roamUpdateRateLabel = [NSTextField labelWithString:@"30 Hz"];
  _roamUpdateRateLabel.alignment = NSTextAlignmentRight;
  [_roamUpdateRateLabel.widthAnchor constraintEqualToConstant:54.0].active = YES;
  NSStackView *roamUpdateRateSetting =
      [NSStackView stackViewWithViews:@[ _roamUpdateRateControl, _roamUpdateRateLabel ]];
  roamUpdateRateSetting.orientation = NSUserInterfaceLayoutOrientationHorizontal;
  roamUpdateRateSetting.alignment = NSLayoutAttributeCenterY;
  roamUpdateRateSetting.spacing = 6.0;
  NSView *roamUpdateRateRow = makeMovementRow(@"Updates", roamUpdateRateSetting);

  _roamStatusLabel = [NSTextField labelWithString:@"WASD move • arrow keys look"];
  _roamStatusLabel.font = [NSFont systemFontOfSize:NSFont.smallSystemFontSize];
  _roamStatusLabel.textColor = NSColor.secondaryLabelColor;
  [_roamStatusLabel.widthAnchor constraintEqualToConstant:268.0].active = YES;
  _roamRows = @[
    _roamTurningModeRow,
    _roamMouseSensitivityRow,
    roamAltitudeRow,
    _aircraftDynamicsRow,
    roamSpeedRow,
    roamUpdateRateRow,
    _roamStatusLabel,
  ];

  InspectorSectionView *movementSection =
      [[InspectorSectionView alloc] initWithTitle:@"Movement"
                                         controls:@[
                                           movementModeRow,
                                           _roamTurningModeRow,
                                           _roamMouseSensitivityRow,
                                           roamAltitudeRow,
                                           _aircraftDynamicsRow,
                                           roamSpeedRow,
                                           roamUpdateRateRow,
                                           _roamStatusLabel,
                                         ]
                                      defaultsKey:@"panorama.inspector.movement.expanded"];

  _groundClearanceDecreaseControl =
      [NSButton buttonWithImage:[NSImage imageWithSystemSymbolName:@"minus"
                                          accessibilityDescription:@"Lower observer"]
                         target:self
                         action:@selector(adjustGroundClearance:)];
  _groundClearanceDecreaseControl.tag = -1;
  _groundClearanceDecreaseControl.controlSize = NSControlSizeSmall;
  _groundClearanceDecreaseControl.bezelStyle = NSBezelStyleTexturedRounded;
  _groundClearanceDecreaseControl.toolTip = @"Lower eye height by 1 m (Option: 0.1 m; Shift: 10 m)";
  [_groundClearanceDecreaseControl.widthAnchor constraintEqualToConstant:24.0].active = YES;

  _groundClearanceControl = [[NSTextField alloc] initWithFrame:NSZeroRect];
  _groundClearanceControl.delegate = self;
  _groundClearanceControl.alignment = NSTextAlignmentRight;
  _groundClearanceControl.font = [NSFont monospacedDigitSystemFontOfSize:12.0
                                                                  weight:NSFontWeightRegular];
  _groundClearanceControl.stringValue = [NSString stringWithFormat:@"%.1f", _groundClearance];
  _groundClearanceControl.toolTip = @"Observer height above the terrain directly beneath it";
  [_groundClearanceControl.widthAnchor constraintEqualToConstant:56.0].active = YES;

  _observerHeightUnit = [NSTextField labelWithString:@"m AGL"];
  _observerHeightUnit.textColor = NSColor.secondaryLabelColor;
  _observerHeightUnit.toolTip = @"Metres above ground level";

  _groundClearanceIncreaseControl =
      [NSButton buttonWithImage:[NSImage imageWithSystemSymbolName:@"plus"
                                          accessibilityDescription:@"Raise observer"]
                         target:self
                         action:@selector(adjustGroundClearance:)];
  _groundClearanceIncreaseControl.tag = 1;
  _groundClearanceIncreaseControl.controlSize = NSControlSizeSmall;
  _groundClearanceIncreaseControl.bezelStyle = NSBezelStyleTexturedRounded;
  _groundClearanceIncreaseControl.toolTip = @"Raise eye height by 1 m (Option: 0.1 m; Shift: 10 m)";
  [_groundClearanceIncreaseControl.widthAnchor constraintEqualToConstant:24.0].active = YES;

  NSStackView *heightSetting = [NSStackView stackViewWithViews:@[
    _groundClearanceDecreaseControl,
    _groundClearanceControl,
    _observerHeightUnit,
    _groundClearanceIncreaseControl,
  ]];
  heightSetting.orientation = NSUserInterfaceLayoutOrientationHorizontal;
  heightSetting.alignment = NSLayoutAttributeCenterY;
  heightSetting.spacing = 6.0;
  [heightSetting.widthAnchor constraintEqualToConstant:178.0].active = YES;

  _observerHeightLabel = [NSTextField labelWithString:@"Eye height"];
  _observerHeightLabel.toolTip = @"Observer height above the terrain directly beneath it";
  [_observerHeightLabel.widthAnchor constraintEqualToConstant:82.0].active = YES;
  NSStackView *heightRow =
      [NSStackView stackViewWithViews:@[ _observerHeightLabel, heightSetting ]];
  heightRow.orientation = NSUserInterfaceLayoutOrientationHorizontal;
  heightRow.alignment = NSLayoutAttributeCenterY;
  heightRow.spacing = 8.0;

  InspectorSectionView *observerSection =
      [[InspectorSectionView alloc] initWithTitle:@"Observer"
                                         controls:@[ heightRow ]
                                      defaultsKey:@"panorama.inspector.observer.expanded"];
  NSStackView *settings = [NSStackView
      stackViewWithViews:@[ heading, destinationSection, movementSection, observerSection ]];
  settings.orientation = NSUserInterfaceLayoutOrientationVertical;
  settings.alignment = NSLayoutAttributeLeading;
  settings.spacing = 18.0;
  settings.edgeInsets = NSEdgeInsetsMake(20.0, 16.0, 20.0, 16.0);
  settings.translatesAutoresizingMaskIntoConstraints = NO;

  InspectorDocumentView *document = [[InspectorDocumentView alloc] initWithFrame:NSZeroRect];
  document.translatesAutoresizingMaskIntoConstraints = NO;
  [document addSubview:settings];
  scrollView.documentView = document;
  NSLayoutConstraint *viewportHeight =
      [document.heightAnchor constraintEqualToAnchor:scrollView.contentView.heightAnchor];
  viewportHeight.priority = NSLayoutPriorityDefaultLow;
  [NSLayoutConstraint activateConstraints:@[
    [document.widthAnchor constraintEqualToAnchor:scrollView.contentView.widthAnchor],
    [document.heightAnchor
        constraintGreaterThanOrEqualToAnchor:scrollView.contentView.heightAnchor],
    viewportHeight,
    [settings.topAnchor constraintEqualToAnchor:document.topAnchor],
    [settings.leadingAnchor constraintEqualToAnchor:document.leadingAnchor],
    [settings.trailingAnchor constraintEqualToAnchor:document.trailingAnchor],
    [settings.bottomAnchor constraintLessThanOrEqualToAnchor:document.bottomAnchor],
  ]];
  [self coordinateSystemChanged:_coordinateSystemControl];
  [self updateRoamControls];
  return viewController;
}

/// Build the read-only diagnostics displayed over the leading side of the
/// rendered scene. Camera values change with completed revisions; inspected
/// point details update independently as hover samples arrive.
- (NSViewController *)makeDebugViewController {
  NSViewController *viewController = [[NSViewController alloc] init];
  NSView *content = [[NSView alloc] initWithFrame:NSMakeRect(0.0, 0.0, 240.0, 430.0)];
  viewController.view = content;

  NSTextField *heading = [NSTextField labelWithString:@"Viewer Debug Info"];
  heading.font = [NSFont boldSystemFontOfSize:NSFont.systemFontSize];

  _debugInfoLabel = [NSTextField labelWithString:@""];
  _debugInfoLabel.font = [NSFont monospacedSystemFontOfSize:12.0 weight:NSFontWeightRegular];
  _debugInfoLabel.maximumNumberOfLines = 0;
  _debugInfoLabel.lineBreakMode = NSLineBreakByClipping;

  NSTextField *pointHeading = [NSTextField labelWithString:@"Inspected Point"];
  pointHeading.font = [NSFont boldSystemFontOfSize:NSFont.systemFontSize];
  _debugPointInfoLabel = [NSTextField labelWithString:@"No point selected."];
  _debugPointInfoLabel.font = [NSFont monospacedSystemFontOfSize:12.0 weight:NSFontWeightRegular];
  _debugPointInfoLabel.maximumNumberOfLines = 0;
  _debugPointInfoLabel.lineBreakMode = NSLineBreakByClipping;

  NSStackView *debugInfo = [NSStackView
      stackViewWithViews:@[ heading, _debugInfoLabel, pointHeading, _debugPointInfoLabel ]];
  debugInfo.orientation = NSUserInterfaceLayoutOrientationVertical;
  debugInfo.alignment = NSLayoutAttributeLeading;
  debugInfo.spacing = 12.0;
  debugInfo.translatesAutoresizingMaskIntoConstraints = NO;
  [content addSubview:debugInfo];
  [NSLayoutConstraint activateConstraints:@[
    [debugInfo.topAnchor constraintEqualToAnchor:content.topAnchor constant:16.0],
    [debugInfo.leadingAnchor constraintEqualToAnchor:content.leadingAnchor constant:16.0],
    [debugInfo.trailingAnchor constraintLessThanOrEqualToAnchor:content.trailingAnchor
                                                       constant:-16.0],
  ]];

  [self updateDebugInfoWithOrientation:_orientation
                   verticalFieldOfView:_verticalFieldOfView
                                 image:_image
                          milliseconds:0.0
                       gpuMilliseconds:0.0
                              streamed:NO
                              revision:0U];
  return viewController;
}

- (void)updateDebugInfoWithOrientation:(panorama::CameraOrientation)orientation
                   verticalFieldOfView:(double)verticalFieldOfView
                                 image:(panorama::ImageSize)image
                          milliseconds:(double)milliseconds
                       gpuMilliseconds:(double)gpuMilliseconds
                              streamed:(BOOL)streamed
                              revision:(uint64_t)revision {
  if (_debugInfoLabel == nil) {
    return;
  }

  const panorama::ObserverLocation observer = _renderer->observer();
  double heading = std::fmod(orientation.heading * panorama::app::kRadiansToDegrees, 360.0);
  if (heading < 0.0) {
    heading += 360.0;
  }
  const double fps = milliseconds > 0.0 ? 1'000.0 / milliseconds : 0.0;
  NSString *performance =
      milliseconds > 0.0
          ? [NSString
                stringWithFormat:@"FPS          %8.2f\nWall latency %8.2f ms", fps, milliseconds]
          : @"FPS                 —\nWall latency        —";
  performance = [performance
      stringByAppendingString:streamed ? @"\nGPU frame      streaming"
                                       : [NSString stringWithFormat:@"\nGPU frame    %8.2f ms",
                                                                    gpuMilliseconds]];
  _debugInfoLabel.stringValue = [NSString
      stringWithFormat:@"%@\nRevision     %8llu\n\n"
                        "Easting    %11.2f m\nNorthing   %11.2f m\nElevation  %11.2f m\n\n"
                        "Heading      %8.2f°\nPitch        %8.2f°\nRoll         %8.2f°\n"
                        "V. FOV       %8.2f°\n\nResolution   %4u × %4u\nMax range  %10.0f m",
                       performance,
                       static_cast<unsigned long long>(revision),
                       observer.easting,
                       observer.northing,
                       observer.elevation,
                       heading,
                       orientation.pitch * panorama::app::kRadiansToDegrees,
                       orientation.roll * panorama::app::kRadiansToDegrees,
                       verticalFieldOfView * panorama::app::kRadiansToDegrees,
                       image.width,
                       image.height,
                       _renderer->max_distance()];
}

/// Build the compact hover readout shown while point inspection is enabled.
- (NSViewController *)makePointInfoViewController {
  NSViewController *viewController = [[NSViewController alloc] init];
  NSView *content = [[NSView alloc] initWithFrame:NSMakeRect(0.0, 0.0, 230.0, 36.0)];
  viewController.view = content;

  _observerInfoLabel = [NSTextField labelWithString:@""];
  _observerInfoLabel.font = [NSFont monospacedSystemFontOfSize:10.0 weight:NSFontWeightRegular];
  _observerInfoLabel.textColor = NSColor.secondaryLabelColor;
  _observerInfoLabel.maximumNumberOfLines = 2;
  _observerInfoLabel.lineBreakMode = NSLineBreakByClipping;
  _movementInfoLabel = [NSTextField labelWithString:@""];
  _movementInfoLabel.font = [NSFont monospacedSystemFontOfSize:10.0 weight:NSFontWeightRegular];
  _movementInfoLabel.textColor = NSColor.secondaryLabelColor;
  _movementInfoLabel.maximumNumberOfLines = 2;
  _movementInfoLabel.lineBreakMode = NSLineBreakByClipping;
  _movementInfoLabel.hidden = YES;

  _pointInfoHeading = [NSTextField labelWithString:@" "];
  _pointInfoHeading.font = [NSFont boldSystemFontOfSize:NSFont.systemFontSize];
  _pointInfoHeading.maximumNumberOfLines = 1;
  _pointInfoHeading.lineBreakMode = NSLineBreakByTruncatingTail;
  _pointInfoLabel = [NSTextField labelWithString:@""];
  _pointInfoLabel.font = [NSFont monospacedSystemFontOfSize:12.0 weight:NSFontWeightRegular];
  _pointInfoLabel.maximumNumberOfLines = 1;
  _pointInfoLabel.lineBreakMode = NSLineBreakByClipping;

  _pointVisibilityIcon = [[NSImageView alloc] initWithFrame:NSZeroRect];
  _pointVisibilityIcon.contentTintColor = NSColor.secondaryLabelColor;
  _pointVisibilityIcon.hidden = YES;
  _pointLockIcon = [[NSImageView alloc] initWithFrame:NSZeroRect];
  _pointLockIcon.contentTintColor = NSColor.secondaryLabelColor;
  _pointLockIcon.hidden = YES;

  _moveToLockedPointControl = [NSButton buttonWithTitle:@"Move here"
                                                 target:self
                                                 action:@selector(moveToLockedPoint:)];
  _moveToLockedPointControl.controlSize = NSControlSizeSmall;
  _moveToLockedPointControl.bezelStyle = NSBezelStyleRounded;
  _moveToLockedPointControl.toolTip = @"Move the observer to the locked terrain point";
  _moveToLockedPointControl.hidden = YES;

  NSView *pointSpacer = [[NSView alloc] initWithFrame:NSZeroRect];
  [pointSpacer setContentHuggingPriority:NSLayoutPriorityDefaultLow
                          forOrientation:NSLayoutConstraintOrientationHorizontal];
  [pointSpacer setContentCompressionResistancePriority:NSLayoutPriorityDefaultLow
                                        forOrientation:NSLayoutConstraintOrientationHorizontal];

  _pointInfoRow = [NSStackView stackViewWithViews:@[
    _pointInfoHeading,
    _pointInfoLabel,
    _pointVisibilityIcon,
    _pointLockIcon,
    pointSpacer,
    _moveToLockedPointControl,
  ]];
  _pointInfoRow.orientation = NSUserInterfaceLayoutOrientationHorizontal;
  _pointInfoRow.alignment = NSLayoutAttributeCenterY;
  _pointInfoRow.spacing = 7.0;

  NSStackView *footer = [NSStackView stackViewWithViews:@[
    _observerInfoLabel,
    _movementInfoLabel,
    _pointInfoRow,
  ]];
  footer.orientation = NSUserInterfaceLayoutOrientationVertical;
  footer.alignment = NSLayoutAttributeLeading;
  footer.spacing = 2.0;
  footer.translatesAutoresizingMaskIntoConstraints = NO;
  [content addSubview:footer];
  [NSLayoutConstraint activateConstraints:@[
    [footer.topAnchor constraintEqualToAnchor:content.topAnchor constant:6.0],
    [footer.bottomAnchor constraintEqualToAnchor:content.bottomAnchor constant:-6.0],
    [footer.leadingAnchor constraintEqualToAnchor:content.leadingAnchor constant:14.0],
    [footer.trailingAnchor constraintEqualToAnchor:content.trailingAnchor constant:-14.0],
    [_observerInfoLabel.widthAnchor constraintEqualToAnchor:footer.widthAnchor],
    [_movementInfoLabel.widthAnchor constraintEqualToAnchor:footer.widthAnchor],
    [_pointInfoRow.widthAnchor constraintEqualToAnchor:footer.widthAnchor],
    [_pointInfoRow.heightAnchor constraintEqualToConstant:24.0],
  ]];
  return viewController;
}

- (void)adjustGroundClearance:(NSButton *)sender {
  NSEventModifierFlags modifiers = NSApp.currentEvent.modifierFlags;
  const double step = (modifiers & NSEventModifierFlagShift) != 0U    ? 10.0
                      : (modifiers & NSEventModifierFlagOption) != 0U ? 0.1
                                                                      : 1.0;
  const bool holdAltitude = ([self isRoamingEnabled] || [self isCruisingEnabled]) &&
                            _roamAltitudeModeControl.selectedSegment == 1;
  const double current = holdAltitude ? _roamAltitude : _groundClearance;
  const double minimum = holdAltitude ? -500.0 : 0.0;
  _groundClearanceControl.doubleValue =
      std::clamp(current + static_cast<double>(sender.tag) * step, minimum, 100'000.0);
  _groundClearanceControl.stringValue =
      [NSString stringWithFormat:@"%.1f", _groundClearanceControl.doubleValue];
  [self commitGroundClearanceControl];
}

- (BOOL)commitGroundClearanceControl {
  const std::optional<double> parsed =
      panorama::app::parse_range_value(_groundClearanceControl.stringValue);
  const bool holdAltitude = ([self isRoamingEnabled] || [self isCruisingEnabled]) &&
                            _roamAltitudeModeControl.selectedSegment == 1;
  const double minimum = holdAltitude ? -500.0 : 0.0;
  const BOOL valid = parsed.has_value() && *parsed >= minimum && *parsed <= 100'000.0;
  _groundClearanceControl.textColor = valid ? NSColor.controlTextColor : NSColor.systemRedColor;
  _groundClearanceControl.toolTip =
      valid
          ? (holdAltitude ? @"Observer elevation above mean sea level"
                          : @"Observer height above the terrain directly beneath it")
          : (holdAltitude ? @"Altitude must be between -500 and 100,000 metres AMSL."
                          : @"Eye height must be between 0 and 100,000 metres above ground level.");
  if (!valid) {
    NSBeep();
    return NO;
  }
  _groundClearanceControl.stringValue = [NSString stringWithFormat:@"%.1f", *parsed];
  if (holdAltitude) {
    if (std::abs(*parsed - _roamAltitude) <= 1e-9) {
      return YES;
    }
    _roamAltitude = *parsed;
    _roamRequestToken = _renderer->request_roam(
        _roamDesiredPosition,
        panorama::app::RoamAltitudeMode::HoldAltitude,
        _roamAltitude
    );
    return YES;
  }
  if (std::abs(*parsed - _groundClearance) <= 1e-9) {
    return YES;
  }
  _groundClearance = *parsed;
  if ([self isRoamingEnabled] || [self isCruisingEnabled]) {
    _roamRequestToken = _renderer->request_roam(
        _roamDesiredPosition,
        panorama::app::RoamAltitudeMode::FollowTerrain,
        _groundClearance
    );
    return YES;
  }
  _renderer->request_ground_clearance(_groundClearance);
  return YES;
}

- (void)coordinateSystemChanged:(id)sender {
  (void)sender;
  const NSInteger tag = _coordinateSystemControl.selectedItem.tag;
  switch (tag) {
  case -1:
    _coordinateInputControl.placeholderString = @"Enter or paste a coordinate";
    _coordinateInputControl.toolTip = @"The coordinate system will be detected automatically";
    break;
  case static_cast<NSInteger>(panorama::app::CoordinateInputSystem::Wgs84):
    _coordinateInputControl.placeholderString = @"46.948, 7.447";
    _coordinateInputControl.toolTip = @"Latitude, longitude in decimal WGS 84 degrees";
    break;
  case static_cast<NSInteger>(panorama::app::CoordinateInputSystem::SwissLv95):
    _coordinateInputControl.placeholderString = @"2600000, 1200000";
    _coordinateInputControl.toolTip = @"LV95 easting, northing in metres";
    break;
  case static_cast<NSInteger>(panorama::app::CoordinateInputSystem::BritishNationalGrid):
    _coordinateInputControl.placeholderString = @"NG 90716 59877";
    _coordinateInputControl.toolTip =
        @"OS grid reference, or British National Grid easting, northing in metres";
    break;
  case static_cast<NSInteger>(panorama::app::CoordinateInputSystem::Terrain):
    _coordinateInputControl.placeholderString =
        _renderer->terrain_crs().id() == panorama::CrsId::FrenchLambert93 ? @"700000, 6600000"
        : _renderer->terrain_crs().id() == panorama::CrsId::SwissLv95     ? @"2600000, 1200000"
                                                                          : @"400000, 300000";
    _coordinateInputControl.toolTip = [NSString
        stringWithFormat:@"Easting, northing in %s metres", _renderer->terrain_crs().name()];
    break;
  default:
    throw std::logic_error("Unknown coordinate-system menu item");
  }
  [self updateCoordinateInputValidation];
}

- (BOOL)updateCoordinateInputValidation {
  if (_coordinateSystemControl == nil || _coordinateInputControl == nil ||
      _coordinateStatusLabel == nil || _coordinateMoveControl == nil) {
    return NO;
  }
  _coordinateDestination.reset();
  NSString *input = [_coordinateInputControl.stringValue
      stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceAndNewlineCharacterSet];
  if (input.length == 0U) {
    _coordinateInputControl.textColor = NSColor.controlTextColor;
    const NSInteger tag = _coordinateSystemControl.selectedItem.tag;
    _coordinateStatusLabel.stringValue =
        tag == -1
            ? @"Format will be detected automatically"
            : [NSString
                  stringWithFormat:@"Enter coordinates as %s",
                                   tag == static_cast<NSInteger>(
                                              panorama::app::CoordinateInputSystem::Terrain
                                          )
                                       ? _renderer->terrain_crs().name()
                                       : _coordinateSystemControl.titleOfSelectedItem.UTF8String];
    _coordinateStatusLabel.textColor = NSColor.secondaryLabelColor;
    _coordinateMoveControl.enabled = NO;
    return NO;
  }

  try {
    panorama::app::ParsedCoordinateInput parsed;
    const NSInteger tag = _coordinateSystemControl.selectedItem.tag;
    if (tag == -1) {
      std::vector<panorama::app::ParsedCoordinateInput> candidates =
          panorama::app::detect_coordinate_inputs(input.UTF8String, _renderer->terrain_crs());
      if (candidates.size() > 1U) {
        std::vector<panorama::app::ParsedCoordinateInput> covered;
        for (const auto &candidate : candidates) {
          if (coordinate_has_terrain_coverage(_renderer->terrain_coverage(), candidate.projected)) {
            covered.push_back(candidate);
          }
        }
        if (!covered.empty()) {
          candidates = std::move(covered);
        }
      }
      if (candidates.size() > 1U) {
        NSMutableArray<NSString *> *names = [NSMutableArray arrayWithCapacity:candidates.size()];
        for (const auto &candidate : candidates) {
          [names addObject:[NSString stringWithUTF8String:candidate.source_name.c_str()]];
        }
        NSString *possibilities = [names componentsJoinedByString:@" or "];
        _coordinateStatusLabel.stringValue =
            [NSString stringWithFormat:@"Could be %@ — choose a system above", possibilities];
        _coordinateStatusLabel.textColor = NSColor.systemOrangeColor;
        _coordinateInputControl.textColor = NSColor.controlTextColor;
        _coordinateInputControl.toolTip = _coordinateStatusLabel.stringValue;
        _coordinateMoveControl.enabled = NO;
        return NO;
      }
      parsed = candidates.front();
    } else {
      parsed = panorama::app::parse_coordinate_input(
          input.UTF8String,
          _renderer->terrain_crs(),
          static_cast<panorama::app::CoordinateInputSystem>(tag)
      );
    }
    _coordinateDestination = parsed;
    NSString *source = [NSString stringWithUTF8String:parsed.source_name.c_str()];
    _coordinateStatusLabel.stringValue = [NSString stringWithFormat:@"%@ • %.5f°, %.5f°",
                                                                    source,
                                                                    parsed.geographic.lat,
                                                                    parsed.geographic.lon];
    _coordinateStatusLabel.textColor = NSColor.secondaryLabelColor;
    _coordinateInputControl.textColor = NSColor.controlTextColor;
    _coordinateInputControl.toolTip = _coordinateStatusLabel.stringValue;
    _coordinateMoveControl.enabled = YES;
    return YES;
  } catch (const std::exception &exception) {
    NSString *error = [NSString stringWithUTF8String:exception.what()];
    _coordinateStatusLabel.stringValue = error;
    _coordinateStatusLabel.textColor = NSColor.systemRedColor;
    _coordinateInputControl.textColor = NSColor.systemRedColor;
    _coordinateInputControl.toolTip = error;
    _coordinateMoveControl.enabled = NO;
    return NO;
  }
}

- (void)moveToCoordinate:(id)sender {
  (void)sender;
  if (![self updateCoordinateInputValidation] || !_coordinateDestination.has_value()) {
    NSBeep();
    return;
  }
  const panorama::app::ParsedCoordinateInput parsed = *_coordinateDestination;
  _coordinateMovePending = true;
  _coordinateStatusLabel.stringValue =
      [NSString stringWithFormat:@"%s • locating terrain…", parsed.source_name.c_str()];
  _coordinateStatusLabel.textColor = NSColor.secondaryLabelColor;
  [self requestMapPointEasting:parsed.projected.x
                      northing:parsed.projected.y
                        action:panorama::app::MapPointAction::MoveObserver];
}

- (void)setPointInfoStatus:(NSString *)status {
  if (_pointInfoHeading == nil) {
    return;
  }
  // Keep one blank glyph in the otherwise empty point row so entering and
  // leaving hover does not resize the minimap panel beneath the pointer.
  _pointInfoHeading.stringValue = status.length == 0U ? @" " : status;
  _pointInfoLabel.stringValue = @"";
  _moveToLockedPointControl.hidden = YES;
  [self setPointInfoSymbolsVisible:false locked:false occluded:false];
}

/// Keep observer state visible independently of the transient hover/lock row.
/// Compact projected coordinates are more useful here than place names: they
/// update immediately during movement and match the terrain dataset's grid.
- (void)updateMiniMapTelemetry {
  if (!_pointInspectionEnabled || _observerInfoLabel == nil) {
    return;
  }
  _observerInfoLabel.stringValue =
      [NSString stringWithFormat:@"Observer  E %.0f • N %.0f\n%.1f m AGL • %.0f m AMSL",
                                 _observer.easting,
                                 _observer.northing,
                                 _groundClearance,
                                 _observer.elevation];
  _observerInfoLabel.toolTip =
      [NSString stringWithFormat:@"Observer: easting %.1f m, northing %.1f m, %.1f m above ground, "
                                  "%.1f m above mean sea level",
                                 _observer.easting,
                                 _observer.northing,
                                 _groundClearance,
                                 _observer.elevation];

  const BOOL roaming = [self isRoamingEnabled];
  const BOOL cruising = [self isCruisingEnabled];
  const BOOL movementHidden = !roaming && !cruising;
  const BOOL movementVisibilityChanged = _movementInfoLabel.hidden != movementHidden;
  _movementInfoLabel.hidden = movementHidden;
  if (roaming || cruising) {
    const BOOL aircraft = [self isAircraftDynamicsEnabled];
    double heading = std::fmod(_orientation.heading * panorama::app::kRadiansToDegrees, 360.0);
    if (heading < 0.0) {
      heading += 360.0;
    }
    NSString *mode = _roamAltitudeModeControl.selectedSegment == 0
                         ? @"Terrain"
                         : (cruising ? @"Flight" : @"Altitude");
    NSString *state = aircraft ? (_viewerPaused ? @"Aircraft paused" : @"Aircraft")
                               : (cruising ? (_viewerPaused ? @"Cruise paused" : @"Cruise")
                                           : (_viewerPaused ? @"Roam paused" : @"Roam"));
    const double speed = aircraft ? _aircraftAirspeed : (cruising ? _cruiseSpeed : _roamSpeed);
    _movementInfoLabel.stringValue =
        aircraft
            ? [NSString
                  stringWithFormat:@"%@ • %@ • %@\nHeading %03.0f° • Pitch %+.0f° • Bank %+.0f°",
                                   state,
                                   format_movement_speed(speed),
                                   mode,
                                   heading,
                                   _orientation.pitch * panorama::app::kRadiansToDegrees,
                                   _aircraftBank * panorama::app::kRadiansToDegrees]
        : cruising
            ? [NSString stringWithFormat:@"%@ • %@ • %@\nHeading %03.0f° • Pitch %+.0f°",
                                         state,
                                         format_movement_speed(speed),
                                         mode,
                                         heading,
                                         _orientation.pitch * panorama::app::kRadiansToDegrees]
            : [NSString stringWithFormat:@"%@ • %@ • %@\nHeading %03.0f°",
                                         state,
                                         format_movement_speed(speed),
                                         mode,
                                         heading];
    _movementInfoLabel.toolTip = _movementInfoLabel.stringValue;
  }
  if (movementVisibilityChanged) {
    [_miniMapPanel informationFooterContentDidChange];
  }
}

- (void)setPointInfoSymbolsVisible:(bool)visible locked:(bool)locked occluded:(bool)occluded {
  if (_pointVisibilityIcon == nil || _pointLockIcon == nil) {
    return;
  }
  _pointVisibilityIcon.hidden = !visible;
  _pointLockIcon.hidden = !visible;
  _moveToLockedPointControl.hidden =
      !(visible && locked && _pointInspectionLocked && _lockedPoint.has_value());
  if (!visible) {
    return;
  }

  NSString *visibilitySymbol = occluded ? @"eye.slash" : @"eye";
  NSString *visibilityDescription =
      occluded ? @"Terrain point is occluded" : @"Terrain point is visible";
  NSString *lockSymbol = locked ? @"lock.fill" : @"lock.open";
  NSString *lockDescription = locked ? @"Terrain point is locked" : @"Terrain point is unlocked";
  NSImageSymbolConfiguration *configuration =
      [NSImageSymbolConfiguration configurationWithPointSize:12.0 weight:NSFontWeightMedium];
  _pointVisibilityIcon.image = [[NSImage imageWithSystemSymbolName:visibilitySymbol
                                          accessibilityDescription:visibilityDescription]
      imageWithSymbolConfiguration:configuration];
  _pointVisibilityIcon.toolTip = visibilityDescription;
  [_pointVisibilityIcon setAccessibilityLabel:visibilityDescription];
  _pointLockIcon.image = [[NSImage imageWithSystemSymbolName:lockSymbol
                                    accessibilityDescription:lockDescription]
      imageWithSymbolConfiguration:configuration];
  _pointLockIcon.toolTip = lockDescription;
  [_pointLockIcon setAccessibilityLabel:lockDescription];
}

/// Put detailed samples in the opt-in debug overlay, leaving the persistent
/// point card small enough to sit naturally beneath the minimap.
- (void)updateDebugPointInfo:(std::optional<panorama::app::PointInspection>)inspection {
  if (_debugPointInfoLabel == nil) {
    return;
  }
  if (!inspection.has_value()) {
    _debugPointInfoLabel.stringValue = @"No point selected.";
    return;
  }
  const panorama::app::PointInspection &point = *inspection;
  if (!point.hit) {
    _debugPointInfoLabel.stringValue =
        [NSString stringWithFormat:@"Pixel      %4u, %4u\nNo terrain intersection",
                                   point.pixel.x,
                                   point.pixel.y];
    return;
  }
  if (point.map_selected) {
    _debugPointInfoLabel.stringValue =
        [NSString stringWithFormat:@"Map selection\n"
                                    "Distance   %10.1f m\nElevation  %10.1f m\n"
                                    "Easting    %10.1f m\nNorthing   %10.1f m",
                                   point.distance,
                                   point.elevation,
                                   point.easting,
                                   point.northing];
    return;
  }
  _debugPointInfoLabel.stringValue =
      [NSString stringWithFormat:@"Pixel      %4u, %4u\n"
                                  "Distance   %10.1f m\nElevation  %10.1f m\n"
                                  "Easting    %10.1f m\nNorthing   %10.1f m\n"
                                  "Slope      %10.1f°\nAspect     %10.1f°",
                                 point.pixel.x,
                                 point.pixel.y,
                                 point.distance,
                                 point.elevation,
                                 point.easting,
                                 point.northing,
                                 point.slope_degrees,
                                 point.aspect_degrees];
}

- (void)updatePointInfo:(std::optional<panorama::app::PointInspection>)inspection {
  if (_pointInfoLabel == nil) {
    return;
  }
  [self updateDebugPointInfo:inspection];
  if (!_pointInspectionEnabled)
    return;
  if (!inspection.has_value()) {
    if (!_pointInspectionLocked) {
      [_miniMapPanel clearInspectedPoint];
    }
    [self setPointInfoStatus:@""];
    return;
  }
  const panorama::app::PointInspection &point = *inspection;
  if (!point.hit) {
    [_miniMapPanel clearInspectedPoint];
    [self setPointInfoStatus:@"No terrain intersection"];
    return;
  }
  [_miniMapPanel setInspectedPointEasting:point.easting
                                 northing:point.northing
                                   locked:_pointInspectionLocked || _pointLockPending];
  _pointInfoHeading.stringValue = @"Distance";
  _pointInfoLabel.stringValue = format_point_distance(point.distance);
  [self setPointInfoSymbolsVisible:true
                            locked:_pointInspectionLocked || _pointLockPending
                          occluded:_pointInspectionLocked && _lockedPointOccluded];
}

/// Keep a locked world point aligned with the latest completed camera view.
/// The off-screen state is represented both by an edge arrow and in text, so
/// the lock remains unambiguous even if another overlay obscures the marker.
- (void)updateLockedPointIndicatorWithOrientation:(panorama::CameraOrientation)orientation
                              verticalFieldOfView:(double)verticalFieldOfView
                                            image:(panorama::ImageSize)image {
  const panorama::app::PointInspection *point = nullptr;
  bool locked = false;
  if (_pointInspectionLocked && _lockedPoint.has_value()) {
    point = &*_lockedPoint;
    locked = true;
  } else if (_mapHoverPoint.has_value()) {
    point = &*_mapHoverPoint;
  }
  if (point == nullptr) {
    [_panoramaView setTerrainPointIndicator:std::nullopt image:{} locked:true occluded:false];
    return;
  }
  const panorama::app::LockedPointProjection projection = panorama::app::project_locked_point(
      {point->easting, point->northing, point->elevation},
      _observer,
      image,
      verticalFieldOfView,
      orientation
  );
  const bool occluded =
      locked && _lockedPointOccluded && _targetVisibilityRevision == _displayedRevision;
  [_panoramaView setTerrainPointIndicator:projection image:image locked:locked occluded:occluded];
  if (locked) {
    _pointInfoHeading.stringValue = @"Distance";
    [self setPointInfoSymbolsVisible:true locked:true occluded:occluded];
  }
}

/// Palette and range controls have no effect on the uncoloured white mode.
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

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

@implementation PanoramaController (PanoramaInput)

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

@end

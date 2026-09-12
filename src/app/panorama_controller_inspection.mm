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

@implementation PanoramaController (MiniMapInteraction)

[[nodiscard]] static bool coordinate_has_terrain_coverage(
    const panorama::TerrainCoverage &coverage,
    panorama::Coord coordinate
) {
  for (const auto &dataset : coverage.datasets) {
    try {
      const auto native = panorama::transform_coordinates(
                              coverage.datasets.front().epsg_code,
                              dataset.epsg_code,
                              std::span(&coordinate, 1U)
      )
                              .front();
      const panorama::TileKey key = panorama::tile_key_at(dataset.grid, native.x, native.y);
      if (std::binary_search(dataset.tiles.begin(), dataset.tiles.end(), key))
        return true;
    } catch (const std::out_of_range &) {
      // Try the remaining datasets when this point is outside a native grid.
    }
  }
  return false;
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

@end

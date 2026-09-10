#include "panorama_view.h"
#include "peak_label_overlay.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>

/// Non-interactive symbol layered over the Metal view for a locked point.
@interface LockedPointMarkerView : NSImageView
@end

@implementation LockedPointMarkerView
- (NSView *)hitTest:(NSPoint)point {
  (void)point;
  return nil;
}
@end

/// A pause badge must never take pointer ownership away from the panorama.
@interface ViewerPauseIndicatorView : NSStackView
@end

@implementation ViewerPauseIndicatorView
- (NSView *)hitTest:(NSPoint)point {
  (void)point;
  return nil;
}
@end

static void stroke_hud_path(NSBezierPath *path, CGFloat foregroundWidth) {
  path.lineCapStyle = NSLineCapStyleRound;
  path.lineJoinStyle = NSLineJoinStyleRound;
  [NSColor.blackColor setStroke];
  path.lineWidth = foregroundWidth + 2.0;
  [path stroke];
  [NSColor.whiteColor setStroke];
  path.lineWidth = foregroundWidth;
  [path stroke];
}

[[nodiscard]] static NSString *heading_label(int unwrappedDegrees) {
  const int degrees = (unwrappedDegrees % 360 + 360) % 360;
  switch (degrees) {
  case 0:
    return @"N";
  case 90:
    return @"E";
  case 180:
    return @"S";
  case 270:
    return @"W";
  default:
    return [NSString stringWithFormat:@"%03d", degrees];
  }
}

/// Pointer-transparent attitude HUD centred on Cruise's neutral steering point.
@interface CruiseHUDView : NSView {
@private
  double _heading;
  double _pitch;
  double _bank;
  double _verticalFieldOfView;
  bool _aircraftMode;
}
- (void)setHeading:(double)heading
                  pitch:(double)pitch
                   bank:(double)bank
    verticalFieldOfView:(double)verticalFieldOfView
           aircraftMode:(bool)aircraftMode;
@end

@implementation CruiseHUDView
- (NSView *)hitTest:(NSPoint)point {
  (void)point;
  return nil;
}

- (void)setHeading:(double)heading
                  pitch:(double)pitch
                   bank:(double)bank
    verticalFieldOfView:(double)verticalFieldOfView
           aircraftMode:(bool)aircraftMode {
  _heading = heading;
  _pitch = pitch;
  _bank = bank;
  _verticalFieldOfView = verticalFieldOfView;
  _aircraftMode = aircraftMode;
  self.needsDisplay = YES;
}

- (void)drawRect:(NSRect)dirtyRect {
  (void)dirtyRect;
  if (!(self.bounds.size.width > 0.0) || !(self.bounds.size.height > 0.0) ||
      !(_verticalFieldOfView > 0.0)) {
    return;
  }

  const NSPoint centre = NSMakePoint(NSMidX(self.bounds), NSMidY(self.bounds));
  const double bank = _aircraftMode ? _bank : 0.0;
  const double sinBank = std::sin(bank);
  const double cosBank = std::cos(bank);
  const double focalLength =
      self.bounds.size.height * 0.5 / std::tan(std::clamp(_verticalFieldOfView, 0.01, 3.0) * 0.5);
  const auto attitudePoint = [&](double x, double y) {
    return NSMakePoint(centre.x + x * cosBank - y * sinBank, centre.y + x * sinBank + y * cosBank);
  };

  NSShadow *textShadow = [[NSShadow alloc] init];
  textShadow.shadowColor = NSColor.blackColor;
  textShadow.shadowBlurRadius = 2.0;
  textShadow.shadowOffset = NSMakeSize(0.0, -1.0);
  NSDictionary<NSAttributedStringKey, id> *textAttributes = @{
    NSFontAttributeName : [NSFont monospacedDigitSystemFontOfSize:10.0 weight:NSFontWeightSemibold],
    NSForegroundColorAttributeName : NSColor.whiteColor,
    NSShadowAttributeName : textShadow,
  };
  const auto drawCentredText = [&](NSString *text, NSPoint point) {
    const NSSize size = [text sizeWithAttributes:textAttributes];
    [text drawAtPoint:NSMakePoint(point.x - size.width * 0.5, point.y - size.height * 0.5)
        withAttributes:textAttributes];
  };

  // The pitch ladder is expressed in world elevation angles. It moves behind
  // the fixed boresight and rotates with the apparent horizon as the aircraft
  // banks. Clipping keeps the overlay useful without obscuring much terrain.
  [NSGraphicsContext saveGraphicsState];
  [NSBezierPath clipRect:NSMakeRect(centre.x - 135.0, centre.y - 82.0, 270.0, 164.0)];
  for (int pitchDegrees = -60; pitchDegrees <= 60; pitchDegrees += 10) {
    const double pitchRadians =
        static_cast<double>(pitchDegrees) * panorama::app::kDegreesToRadians;
    const double y = std::tan(pitchRadians - _pitch) * focalLength;
    if (std::abs(y) > 100.0) {
      continue;
    }

    const bool horizon = pitchDegrees == 0;
    const double halfWidth = horizon ? 112.0 : (pitchDegrees % 20 == 0 ? 43.0 : 34.0);
    const double centreGap = horizon ? 20.0 : 9.0;
    NSBezierPath *line = [NSBezierPath bezierPath];
    [line moveToPoint:attitudePoint(-halfWidth, y)];
    [line lineToPoint:attitudePoint(-centreGap, y)];
    [line moveToPoint:attitudePoint(centreGap, y)];
    [line lineToPoint:attitudePoint(halfWidth, y)];
    if (pitchDegrees < 0) {
      const CGFloat dash[] = {4.0, 3.0};
      [line setLineDash:dash count:2 phase:0.0];
    }
    stroke_hud_path(line, horizon ? 1.6 : 1.0);

    if (!horizon) {
      NSString *label = [NSString stringWithFormat:@"%d", std::abs(pitchDegrees)];
      drawCentredText(label, attitudePoint(-halfWidth - 13.0, y));
      drawCentredText(label, attitudePoint(halfWidth + 13.0, y));
    }
  }
  [NSGraphicsContext restoreGraphicsState];

  if (_aircraftMode) {
    constexpr double kBankRadius = 70.0;
    NSBezierPath *bankArc = [NSBezierPath bezierPath];
    [bankArc appendBezierPathWithArcWithCenter:centre
                                        radius:kBankRadius
                                    startAngle:30.0
                                      endAngle:150.0];
    for (const int degrees : {-60, -45, -30, -20, -10, 0, 10, 20, 30, 45, 60}) {
      const double radians = static_cast<double>(degrees) * panorama::app::kDegreesToRadians;
      const double innerRadius = kBankRadius - (degrees % 30 == 0 ? 7.0 : 4.0);
      [bankArc moveToPoint:NSMakePoint(
                               centre.x + innerRadius * std::sin(radians),
                               centre.y + innerRadius * std::cos(radians)
                           )];
      [bankArc lineToPoint:NSMakePoint(
                               centre.x + kBankRadius * std::sin(radians),
                               centre.y + kBankRadius * std::cos(radians)
                           )];
    }
    stroke_hud_path(bankArc, 1.0);

    const double indicatedBank = std::clamp(
        bank,
        -60.0 * panorama::app::kDegreesToRadians,
        60.0 * panorama::app::kDegreesToRadians
    );
    const double tangentX = std::cos(indicatedBank);
    const double tangentY = -std::sin(indicatedBank);
    const double radialX = std::sin(indicatedBank);
    const double radialY = std::cos(indicatedBank);
    NSBezierPath *bankPointer = [NSBezierPath bezierPath];
    [bankPointer moveToPoint:NSMakePoint(
                                 centre.x + radialX * (kBankRadius - 6.0),
                                 centre.y + radialY * (kBankRadius - 6.0)
                             )];
    [bankPointer lineToPoint:NSMakePoint(
                                 centre.x + radialX * (kBankRadius + 4.0) + tangentX * 4.0,
                                 centre.y + radialY * (kBankRadius + 4.0) + tangentY * 4.0
                             )];
    [bankPointer lineToPoint:NSMakePoint(
                                 centre.x + radialX * (kBankRadius + 4.0) - tangentX * 4.0,
                                 centre.y + radialY * (kBankRadius + 4.0) - tangentY * 4.0
                             )];
    [bankPointer closePath];
    [NSColor.blackColor setStroke];
    bankPointer.lineWidth = 3.0;
    [bankPointer stroke];
    [NSColor.whiteColor setFill];
    [bankPointer fill];
  }

  // A short compass ribbon makes heading readable without moving the user's
  // attention to a corner of the view.
  const double headingDegrees = std::remainder(_heading * panorama::app::kRadiansToDegrees, 360.0);
  constexpr double kHeadingPixelsPerDegree = 2.35;
  const double tapeY = centre.y + 108.0;
  [NSGraphicsContext saveGraphicsState];
  [NSBezierPath clipRect:NSMakeRect(centre.x - 108.0, tapeY - 4.0, 216.0, 35.0)];
  NSBezierPath *headingTape = [NSBezierPath bezierPath];
  [headingTape moveToPoint:NSMakePoint(centre.x - 108.0, tapeY)];
  [headingTape lineToPoint:NSMakePoint(centre.x + 108.0, tapeY)];
  const int firstTick = static_cast<int>(std::floor((headingDegrees - 50.0) / 10.0)) * 10;
  for (int tick = firstTick; tick <= headingDegrees + 50.0; tick += 10) {
    const double x =
        centre.x + (static_cast<double>(tick) - headingDegrees) * kHeadingPixelsPerDegree;
    const int normalisedTick = (tick % 360 + 360) % 360;
    const bool labelled = normalisedTick % 30 == 0;
    [headingTape moveToPoint:NSMakePoint(x, tapeY)];
    [headingTape lineToPoint:NSMakePoint(x, tapeY + (labelled ? 9.0 : 5.0))];
    if (labelled) {
      drawCentredText(heading_label(tick), NSMakePoint(x, tapeY + 18.0));
    }
  }
  stroke_hud_path(headingTape, 1.0);
  [NSGraphicsContext restoreGraphicsState];

  NSBezierPath *headingPointer = [NSBezierPath bezierPath];
  [headingPointer moveToPoint:NSMakePoint(centre.x, tapeY - 1.0)];
  [headingPointer lineToPoint:NSMakePoint(centre.x - 4.0, tapeY - 7.0)];
  [headingPointer lineToPoint:NSMakePoint(centre.x + 4.0, tapeY - 7.0)];
  [headingPointer closePath];
  [NSColor.blackColor setStroke];
  headingPointer.lineWidth = 3.0;
  [headingPointer stroke];
  [NSColor.whiteColor setFill];
  [headingPointer fill];
  const int displayedHeading = (static_cast<int>(std::lround(headingDegrees)) % 360 + 360) % 360;
  drawCentredText(
      [NSString stringWithFormat:@"%03d°", displayedHeading],
      NSMakePoint(centre.x, tapeY - 17.0)
  );

  // Draw the boresight last so it remains the dominant, fixed steering datum.
  NSBezierPath *boresight = [NSBezierPath bezierPath];
  [boresight appendBezierPathWithOvalInRect:NSMakeRect(centre.x - 5.0, centre.y - 5.0, 10.0, 10.0)];
  [boresight moveToPoint:NSMakePoint(centre.x - 23.0, centre.y)];
  [boresight lineToPoint:NSMakePoint(centre.x - 9.0, centre.y)];
  [boresight moveToPoint:NSMakePoint(centre.x + 9.0, centre.y)];
  [boresight lineToPoint:NSMakePoint(centre.x + 23.0, centre.y)];
  [boresight moveToPoint:NSMakePoint(centre.x, centre.y - 16.0)];
  [boresight lineToPoint:NSMakePoint(centre.x, centre.y - 9.0)];
  stroke_hud_path(boresight, 1.3);
}
@end

@interface PanoramaView () {
  NSPoint _lastMouseLocation;
  NSTrackingArea *_inspectionTrackingArea;
  LockedPointMarkerView *_lockedPointMarker;
  ViewerPauseIndicatorView *_pauseIndicator;
  NSTextField *_pauseIndicatorLabel;
  CruiseHUDView *_cruiseHUD;
  PeakLabelOverlayView *_peakLabelOverlay;
  double _lockedPointPixelX;
  double _lockedPointPixelY;
  panorama::ImageSize _lockedPointImage;
  double _lockedPointDirectionX;
  double _lockedPointDirectionY;
  bool _pointInspectionEnabled;
  bool _mouseTurningEnabled;
  bool _cruiseSteeringEnabled;
  bool _peakLabelsNearPointer;
  std::optional<uint64_t> _peakLabelFrameRevision;
  bool _viewerPaused;
  bool _lockedPointIndicatorActive;
  bool _lockedPointOnscreen;
  bool _pointIndicatorLocked;
  bool _lockedPointOccluded;
}
@end

@implementation PanoramaView

- (void)ensurePeakLabelOverlay {
  if (_peakLabelOverlay != nil)
    return;
  _peakLabelOverlay = [[PeakLabelOverlayView alloc] initWithFrame:NSZeroRect];
  _peakLabelOverlay.translatesAutoresizingMaskIntoConstraints = NO;
  [self addSubview:_peakLabelOverlay];
  [NSLayoutConstraint activateConstraints:@[
    [_peakLabelOverlay.leadingAnchor constraintEqualToAnchor:self.leadingAnchor],
    [_peakLabelOverlay.trailingAnchor constraintEqualToAnchor:self.trailingAnchor],
    [_peakLabelOverlay.topAnchor constraintEqualToAnchor:self.topAnchor],
    [_peakLabelOverlay.bottomAnchor constraintEqualToAnchor:self.bottomAnchor],
  ]];
}

- (void)setPeakLabelMode:(panorama::app::PeakLabelMode)mode {
  [self ensurePeakLabelOverlay];
  _peakLabelsNearPointer = mode == panorama::app::PeakLabelMode::NearPointer;
  if (mode == panorama::app::PeakLabelMode::Off) {
    _peakLabelFrameRevision.reset();
    const panorama::app::PeakLabelFrame empty = {};
    [_peakLabelOverlay setPeakFrame:empty];
  } else if (_peakLabelsNearPointer) {
    [_peakLabelOverlay setPointerLocation:std::nullopt];
  }
  [_peakLabelOverlay setLabelMode:mode];
  self.window.acceptsMouseMovedEvents = _peakLabelsNearPointer || _pointInspectionEnabled ||
                                        _mouseTurningEnabled || _cruiseSteeringEnabled;
  [self updateTrackingAreas];
}

- (void)setPeakLabelFrame:(const panorama::app::PeakLabelFrame &)frame {
  if (_peakLabelFrameRevision == frame.revision)
    return;
  [self ensurePeakLabelOverlay];
  _peakLabelFrameRevision = frame.revision;
  [_peakLabelOverlay setPeakFrame:frame];
}

/// Position the marker using the current displayed view bounds. Keeping the
/// projection in image coordinates lets ordinary AppKit layout handle window
/// resizing without retracing or resampling the locked point.
- (void)layoutLockedPointIndicator {
  if (!_lockedPointIndicatorActive || _lockedPointMarker == nil) {
    _lockedPointMarker.hidden = YES;
    return;
  }
  const NSRect bounds = self.bounds;
  const double image_width = _lockedPointImage.width;
  const double image_height = _lockedPointImage.height;
  constexpr CGFloat kMarkerSize = 26.0;
  constexpr CGFloat kEdgeInset = 18.0;
  if (bounds.size.width <= kMarkerSize || bounds.size.height <= kMarkerSize || image_width <= 0.0 ||
      image_height <= 0.0) {
    _lockedPointMarker.hidden = YES;
    return;
  }

  NSPoint centre = {};
  if (_lockedPointOnscreen) {
    centre.x =
        NSMinX(bounds) + static_cast<CGFloat>(_lockedPointPixelX / image_width) * bounds.size.width;
    centre.y = NSMaxY(bounds) -
               static_cast<CGFloat>(_lockedPointPixelY / image_height) * bounds.size.height;
    NSString *description = _lockedPointOccluded
                                ? @"Locked terrain point is occluded"
                                : (_pointIndicatorLocked ? @"Locked terrain point"
                                                         : @"Terrain point under map pointer");
    _lockedPointMarker.image = [NSImage imageWithSystemSymbolName:@"scope"
                                         accessibilityDescription:description];
    [_lockedPointMarker.layer setAffineTransform:CGAffineTransformIdentity];
    _lockedPointMarker.toolTip = description;
  } else {
    // Convert the image's downward-positive direction to AppKit's upward-
    // positive coordinates, then intersect it with an inset view rectangle.
    double direction_x = _lockedPointDirectionX;
    double direction_y = -_lockedPointDirectionY;
    const double length = std::hypot(direction_x, direction_y);
    if (!std::isfinite(length) || length < 1e-12) {
      direction_x = 1.0;
      direction_y = 0.0;
    }
    const CGFloat half_width = std::max(0.0, bounds.size.width * 0.5 - kEdgeInset);
    const CGFloat half_height = std::max(0.0, bounds.size.height * 0.5 - kEdgeInset);
    const double horizontal_scale = std::abs(direction_x) > 1e-12
                                        ? half_width / std::abs(direction_x)
                                        : std::numeric_limits<double>::infinity();
    const double vertical_scale = std::abs(direction_y) > 1e-12
                                      ? half_height / std::abs(direction_y)
                                      : std::numeric_limits<double>::infinity();
    const double scale = std::min(horizontal_scale, vertical_scale);
    centre = NSMakePoint(
        NSMidX(bounds) + static_cast<CGFloat>(direction_x * scale),
        NSMidY(bounds) + static_cast<CGFloat>(direction_y * scale)
    );
    NSString *description = _pointIndicatorLocked ? @"Locked terrain point is outside the view"
                                                  : @"Map pointer is outside the view";
    _lockedPointMarker.image = [NSImage imageWithSystemSymbolName:@"arrow.up.circle.fill"
                                         accessibilityDescription:description];
    const CGFloat rotation = static_cast<CGFloat>(std::atan2(-direction_x, direction_y));
    [_lockedPointMarker.layer setAffineTransform:CGAffineTransformMakeRotation(rotation)];
    _lockedPointMarker.toolTip = description;
  }

  _lockedPointMarker.frame = NSMakeRect(
      centre.x - kMarkerSize * 0.5,
      centre.y - kMarkerSize * 0.5,
      kMarkerSize,
      kMarkerSize
  );
  _lockedPointMarker.hidden = NO;
}

- (void)setTerrainPointIndicator:(std::optional<panorama::app::LockedPointProjection>)projection
                           image:(panorama::ImageSize)image
                          locked:(bool)locked
                        occluded:(bool)occluded {
  _lockedPointIndicatorActive = projection.has_value();
  _pointIndicatorLocked = locked;
  _lockedPointOccluded = locked && occluded;
  if (!_lockedPointIndicatorActive) {
    _lockedPointMarker.hidden = YES;
    return;
  }
  if (_lockedPointMarker == nil) {
    _lockedPointMarker = [[LockedPointMarkerView alloc] initWithFrame:NSZeroRect];
    _lockedPointMarker.imageScaling = NSImageScaleProportionallyUpOrDown;
    _lockedPointMarker.wantsLayer = YES;
    _lockedPointMarker.layer.shadowColor = NSColor.blackColor.CGColor;
    _lockedPointMarker.layer.shadowOpacity = 0.9F;
    _lockedPointMarker.layer.shadowRadius = 2.0;
    _lockedPointMarker.layer.shadowOffset = CGSizeZero;
    [self addSubview:_lockedPointMarker];
  }
  _lockedPointMarker.contentTintColor =
      locked ? NSColor.systemOrangeColor : NSColor.systemBlueColor;
  _lockedPointMarker.alphaValue = _lockedPointOccluded ? 0.45 : 1.0;
  _lockedPointMarker.layer.shadowOpacity = _lockedPointOccluded ? 0.35F : 0.9F;
  _lockedPointOnscreen = projection->onscreen;
  _lockedPointPixelX = projection->pixel_x;
  _lockedPointPixelY = projection->pixel_y;
  _lockedPointImage = image;
  _lockedPointDirectionX = projection->direction_x;
  _lockedPointDirectionY = projection->direction_y;
  [self layoutLockedPointIndicator];
}

- (void)layout {
  [super layout];
  [self layoutLockedPointIndicator];
}

- (void)updateTrackingAreas {
  [super updateTrackingAreas];
  if (_inspectionTrackingArea != nil) {
    [self removeTrackingArea:_inspectionTrackingArea];
    _inspectionTrackingArea = nil;
  }
  if (!_pointInspectionEnabled && !_mouseTurningEnabled && !_cruiseSteeringEnabled &&
      !_peakLabelsNearPointer) {
    return;
  }
  _inspectionTrackingArea =
      [[NSTrackingArea alloc] initWithRect:NSZeroRect
                                   options:NSTrackingMouseMoved | NSTrackingMouseEnteredAndExited |
                                           NSTrackingActiveInKeyWindow | NSTrackingInVisibleRect
                                     owner:self
                                  userInfo:nil];
  [self addTrackingArea:_inspectionTrackingArea];
}

- (void)resetCursorRects {
  [super resetCursorRects];
  if (_viewerPaused) {
    [self addCursorRect:self.bounds cursor:NSCursor.arrowCursor];
  } else if (_cruiseSteeringEnabled) {
    [self addCursorRect:self.bounds cursor:NSCursor.arrowCursor];
  } else if (_pointInspectionEnabled) {
    [self addCursorRect:self.bounds cursor:NSCursor.crosshairCursor];
  }
}

- (void)setPointInspectionEnabled:(bool)enabled {
  _pointInspectionEnabled = enabled;
  self.window.acceptsMouseMovedEvents =
      enabled || _mouseTurningEnabled || _cruiseSteeringEnabled || _peakLabelsNearPointer;
  [self updateTrackingAreas];
  [self.window invalidateCursorRectsForView:self];
}

- (void)setMouseTurningEnabled:(bool)enabled {
  _mouseTurningEnabled = enabled;
  self.window.acceptsMouseMovedEvents =
      enabled || _pointInspectionEnabled || _cruiseSteeringEnabled || _peakLabelsNearPointer;
  [self updateTrackingAreas];
}

- (void)setCruiseSteeringEnabled:(bool)enabled {
  _cruiseSteeringEnabled = enabled;
  if (_cruiseHUD == nil) {
    _cruiseHUD = [[CruiseHUDView alloc] initWithFrame:NSZeroRect];
    _cruiseHUD.translatesAutoresizingMaskIntoConstraints = NO;
    [self addSubview:_cruiseHUD];
    [NSLayoutConstraint activateConstraints:@[
      [_cruiseHUD.leadingAnchor constraintEqualToAnchor:self.leadingAnchor],
      [_cruiseHUD.trailingAnchor constraintEqualToAnchor:self.trailingAnchor],
      [_cruiseHUD.topAnchor constraintEqualToAnchor:self.topAnchor],
      [_cruiseHUD.bottomAnchor constraintEqualToAnchor:self.bottomAnchor],
    ]];
  }
  _cruiseHUD.hidden = !enabled;
  self.window.acceptsMouseMovedEvents =
      enabled || _pointInspectionEnabled || _mouseTurningEnabled || _peakLabelsNearPointer;
  [self updateTrackingAreas];
  [self.window invalidateCursorRectsForView:self];
}

- (void)setCruiseHUDHeading:(double)heading
                      pitch:(double)pitch
                       bank:(double)bank
        verticalFieldOfView:(double)verticalFieldOfView
               aircraftMode:(bool)aircraftMode {
  [_cruiseHUD setHeading:heading
                    pitch:pitch
                     bank:bank
      verticalFieldOfView:verticalFieldOfView
             aircraftMode:aircraftMode];
}

- (void)setViewerPaused:(bool)paused recoveryMessage:(NSString *)recoveryMessage {
  _viewerPaused = paused;
  if (_pauseIndicator == nil) {
    NSImageView *icon =
        [NSImageView imageViewWithImage:[NSImage imageWithSystemSymbolName:@"pause.fill"
                                                  accessibilityDescription:@"Viewer paused"]];
    icon.contentTintColor = NSColor.labelColor;
    _pauseIndicatorLabel = [NSTextField labelWithString:@"Paused — Space to resume"];
    _pauseIndicatorLabel.font = [NSFont systemFontOfSize:NSFont.systemFontSize
                                                  weight:NSFontWeightSemibold];
    _pauseIndicator = [[ViewerPauseIndicatorView alloc] initWithFrame:NSZeroRect];
    [_pauseIndicator addArrangedSubview:icon];
    [_pauseIndicator addArrangedSubview:_pauseIndicatorLabel];
    _pauseIndicator.orientation = NSUserInterfaceLayoutOrientationHorizontal;
    _pauseIndicator.alignment = NSLayoutAttributeCenterY;
    _pauseIndicator.spacing = 8.0;
    _pauseIndicator.edgeInsets = NSEdgeInsetsMake(8.0, 12.0, 8.0, 12.0);
    _pauseIndicator.wantsLayer = YES;
    _pauseIndicator.layer.backgroundColor =
        [NSColor.windowBackgroundColor colorWithAlphaComponent:0.88].CGColor;
    _pauseIndicator.layer.cornerRadius = 10.0;
    _pauseIndicator.layer.borderColor =
        [NSColor.separatorColor colorWithAlphaComponent:0.6].CGColor;
    _pauseIndicator.layer.borderWidth = 1.0;
    _pauseIndicator.layer.shadowColor = NSColor.blackColor.CGColor;
    _pauseIndicator.layer.shadowOpacity = 0.35F;
    _pauseIndicator.layer.shadowRadius = 5.0;
    _pauseIndicator.layer.shadowOffset = CGSizeMake(0.0, -1.0);
    _pauseIndicator.translatesAutoresizingMaskIntoConstraints = NO;
    [self addSubview:_pauseIndicator];
    [NSLayoutConstraint activateConstraints:@[
      [_pauseIndicator.centerXAnchor constraintEqualToAnchor:self.centerXAnchor],
      [_pauseIndicator.topAnchor constraintEqualToAnchor:self.topAnchor constant:16.0],
    ]];
  }
  _pauseIndicatorLabel.stringValue =
      recoveryMessage != nil ? recoveryMessage : @"Paused — Space to resume";
  _pauseIndicator.hidden = !paused;
  [self.window invalidateCursorRectsForView:self];
}

/// Keep cursor requests relative to the view: the worker may change ray
/// resolution between this event and sampling the completed frame.
- (BOOL)inspectionLocationForEvent:(NSEvent *)event x:(double *)x y:(double *)y {
  const NSRect bounds = self.bounds;
  const NSPoint location = [self convertPoint:event.locationInWindow fromView:nil];
  if (bounds.size.width <= 0.0 || bounds.size.height <= 0.0 || !NSPointInRect(location, bounds)) {
    return NO;
  }

  // AppKit view coordinates rise from the bottom-left, whereas ray-image rows
  // use image coordinates from the top-left.
  const double normalised_x =
      std::clamp((location.x - NSMinX(bounds)) / bounds.size.width, 0.0, 1.0);
  const double normalised_y =
      std::clamp((NSMaxY(bounds) - location.y) / bounds.size.height, 0.0, 1.0);
  *x = normalised_x;
  *y = normalised_y;
  return YES;
}

- (void)mouseMoved:(NSEvent *)event {
  if (!_pointInspectionEnabled && !_mouseTurningEnabled && !_cruiseSteeringEnabled &&
      !_peakLabelsNearPointer) {
    [super mouseMoved:event];
    return;
  }
  NSView *content = self.window.contentView;
  const NSPoint contentPoint = [content convertPoint:event.locationInWindow fromView:nil];
  NSView *hit = [content hitTest:contentPoint];
  if (hit != self && ![hit isDescendantOf:self]) {
    if (_peakLabelsNearPointer)
      [_peakLabelOverlay setPointerLocation:std::nullopt];
    [self.panoramaController pointerMovedOverOccludingView:hit];
    return;
  }
  const NSPoint panoramaPoint = [self convertPoint:event.locationInWindow fromView:nil];
  if (_peakLabelsNearPointer)
    [_peakLabelOverlay setPointerLocation:panoramaPoint];
  [self.panoramaController pointerMovedOverPanorama];
  if (_cruiseSteeringEnabled && !_viewerPaused) {
    const NSRect bounds = self.bounds;
    if (bounds.size.width > 0.0 && bounds.size.height > 0.0) {
      const double x =
          std::clamp((panoramaPoint.x - NSMidX(bounds)) / (bounds.size.width * 0.5), -1.0, 1.0);
      const double y =
          std::clamp((panoramaPoint.y - NSMidY(bounds)) / (bounds.size.height * 0.5), -1.0, 1.0);
      [self.panoramaController setCruiseSteeringX:x y:y];
    }
  } else if (_mouseTurningEnabled && !_viewerPaused) {
    // NSEvent's vertical mouse delta is positive downwards, unlike camera
    // pitch, so invert it to preserve the existing drag direction.
    [self.panoramaController mouseTurnForCurrentZoomHeading:event.deltaX * 0.003
                                                      pitch:-event.deltaY * 0.003];
  }
  if (!_pointInspectionEnabled) {
    return;
  }
  double x = 0.0;
  double y = 0.0;
  if (![self inspectionLocationForEvent:event x:&x y:&y]) {
    [self.panoramaController invalidatePanoramaHover];
    return;
  }
  [self.panoramaController inspectLocationX:x y:y];
}

/// Right-click locks the current terrain sample without consuming ordinary
/// left-button interaction.
- (void)rightMouseDown:(NSEvent *)event {
  if (!_pointInspectionEnabled) {
    [super rightMouseDown:event];
    return;
  }
  double x = 0.0;
  double y = 0.0;
  if ([self inspectionLocationForEvent:event x:&x y:&y]) {
    [self.panoramaController togglePointLockAtLocationX:x y:y];
  }
}

- (void)mouseExited:(NSEvent *)event {
  (void)event;
  if (_pointInspectionEnabled || _cruiseSteeringEnabled) {
    [self.panoramaController panoramaPointerExited];
  }
  if (_peakLabelsNearPointer)
    [_peakLabelOverlay setPointerLocation:std::nullopt];
}

- (BOOL)acceptsFirstResponder {
  return YES;
}

- (void)mouseDown:(NSEvent *)event {
  [self.window makeFirstResponder:self];
  _lastMouseLocation = [self convertPoint:event.locationInWindow fromView:nil];
}

- (void)mouseDragged:(NSEvent *)event {
  if (!_viewerPaused && (_mouseTurningEnabled || _cruiseSteeringEnabled)) {
    // AppKit changes mouse-move events into drag events while a button is
    // held. Mouse turning should remain continuous without requiring a drag.
    [self mouseMoved:event];
    return;
  }
  const NSPoint location = [self convertPoint:event.locationInWindow fromView:nil];
  const double heading = (location.x - _lastMouseLocation.x) * 0.003;
  const double pitch = (location.y - _lastMouseLocation.y) * 0.003;
  _lastMouseLocation = location;
  [self.panoramaController panForCurrentZoomHeading:heading pitch:pitch];
}

- (void)scrollWheel:(NSEvent *)event {
  [self.panoramaController zoomWithScrollDelta:event.scrollingDeltaY
                                       precise:event.hasPreciseScrollingDeltas];
}

- (void)keyDown:(NSEvent *)event {
  constexpr double kStep = 2.0 * std::numbers::pi / 180.0;
  switch (event.keyCode) {
  case 123: // Left arrow.
    if (![self.panoramaController isMouseTurningEnabled] &&
        ![self.panoramaController isCruisingEnabled]) {
      [self.panoramaController rotateForCurrentZoomHeading:-kStep pitch:0.0];
    }
    break;
  case 124: // Right arrow.
    if (![self.panoramaController isMouseTurningEnabled] &&
        ![self.panoramaController isCruisingEnabled]) {
      [self.panoramaController rotateForCurrentZoomHeading:kStep pitch:0.0];
    }
    break;
  case 125: // Down arrow.
    if (![self.panoramaController isMouseTurningEnabled] &&
        ![self.panoramaController isCruisingEnabled]) {
      [self.panoramaController rotateForCurrentZoomHeading:0.0 pitch:-kStep];
    }
    break;
  case 126: // Up arrow.
    if (![self.panoramaController isMouseTurningEnabled] &&
        ![self.panoramaController isCruisingEnabled]) {
      [self.panoramaController rotateForCurrentZoomHeading:0.0 pitch:kStep];
    }
    break;
  default: {
    const NSString *characters = event.charactersIgnoringModifiers.lowercaseString;
    if ([self.panoramaController isCruisingEnabled] &&
        ([characters isEqualToString:@"w"] || [characters isEqualToString:@"s"])) {
      const double step = (event.modifierFlags & NSEventModifierFlagShift) != 0U ? 4.0 : 1.0;
      [self.panoramaController
          adjustCruiseSpeedBy:[characters isEqualToString:@"w"] ? step : -step];
    } else if ([self.panoramaController isCruisingEnabled] &&
               ([characters isEqualToString:@"a"] || [characters isEqualToString:@"d"])) {
      // Cruise steering is deliberately mouse-only.
    } else if ([self.panoramaController isRoamingEnabled] && [characters isEqualToString:@"w"]) {
      [self.panoramaController setRoamKey:panorama::app::RoamKey::Forward pressed:YES];
    } else if ([self.panoramaController isRoamingEnabled] && [characters isEqualToString:@"s"]) {
      [self.panoramaController setRoamKey:panorama::app::RoamKey::Backward pressed:YES];
    } else if ([self.panoramaController isRoamingEnabled] && [characters isEqualToString:@"a"]) {
      [self.panoramaController setRoamKey:panorama::app::RoamKey::Left pressed:YES];
    } else if ([self.panoramaController isRoamingEnabled] && [characters isEqualToString:@"d"]) {
      [self.panoramaController setRoamKey:panorama::app::RoamKey::Right pressed:YES];
    } else if ([characters isEqualToString:@"a"]) {
      [self.panoramaController rotateForCurrentZoomHeading:-kStep pitch:0.0];
    } else if ([characters isEqualToString:@"d"]) {
      [self.panoramaController rotateForCurrentZoomHeading:kStep pitch:0.0];
    } else if ([characters isEqualToString:@"s"]) {
      [self.panoramaController rotateForCurrentZoomHeading:0.0 pitch:-kStep];
    } else if ([characters isEqualToString:@"w"]) {
      [self.panoramaController rotateForCurrentZoomHeading:0.0 pitch:kStep];
    } else {
      [super keyDown:event];
    }
    break;
  }
  }
}

- (void)keyUp:(NSEvent *)event {
  const NSString *characters = event.charactersIgnoringModifiers.lowercaseString;
  if ([self.panoramaController isCruisingEnabled] &&
      ([characters isEqualToString:@"w"] || [characters isEqualToString:@"s"] ||
       [characters isEqualToString:@"a"] || [characters isEqualToString:@"d"])) {
    return;
  }
  if (![self.panoramaController isRoamingEnabled]) {
    [super keyUp:event];
    return;
  }
  if ([characters isEqualToString:@"w"]) {
    [self.panoramaController setRoamKey:panorama::app::RoamKey::Forward pressed:NO];
  } else if ([characters isEqualToString:@"s"]) {
    [self.panoramaController setRoamKey:panorama::app::RoamKey::Backward pressed:NO];
  } else if ([characters isEqualToString:@"a"]) {
    [self.panoramaController setRoamKey:panorama::app::RoamKey::Left pressed:NO];
  } else if ([characters isEqualToString:@"d"]) {
    [self.panoramaController setRoamKey:panorama::app::RoamKey::Right pressed:NO];
  } else {
    [super keyUp:event];
  }
}

@end

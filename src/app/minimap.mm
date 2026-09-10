#include "minimap.h"
#include "trace_diagnostics.h"
#include "visibility_mask.h"

#include "crs.h"

#include <Foundation/Foundation.h>
#import <MapKit/MapKit.h>

#import <QuartzCore/QuartzCore.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>

constexpr CGFloat kCompactMapPanelWidth = 300.0;
constexpr CGFloat kCompactMapSectionHeight = 286.0;
constexpr CGFloat kLargeMapPanelWidth = 520.0;
constexpr CGFloat kLargeMapSectionHeight = 456.0;
constexpr CGFloat kMinimumPointSectionHeight = 36.0;
constexpr double kInitialMapDistance = 50'000.0;

enum class AnnotationKind : NSInteger {
  Observer,
  Hover,
  Locked,
};

/// Pairs of title/template URL for XYZ TileOverlays
static const std::array<std::pair<NSString *const, NSString *const>, 2> kTileOverlays = {
    {
        {@"SwissTopo",
         @"https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.pixelkarte-farbe/default/current/3857/"
         @"{z}/{x}/{y}.jpeg"},
        {@"OpenTopoMap", @"https://a.tile.opentopomap.org/{z}/{x}/{y}.png"},
    },
};

/// Return a small symbol-only marker rather than MapKit's full pin balloon.
[[nodiscard]] static NSImage *annotation_image(AnnotationKind kind) {
  NSString *symbolName = nil;
  NSString *description = nil;
  CGFloat pointSize = 0.0;
  NSColor *color = nil;
  switch (kind) {
  case AnnotationKind::Observer:
    symbolName = @"circle.fill";
    description = @"Observer";
    pointSize = 9.0;
    color = NSColor.systemPurpleColor;
    break;
  case AnnotationKind::Hover:
    symbolName = @"circle.fill";
    description = @"Inspected Terrain Point";
    pointSize = 7.0;
    color = NSColor.systemBlueColor;
    break;
  case AnnotationKind::Locked:
    symbolName = @"circle.fill";
    description = @"Locked Terrain Point";
    pointSize = 7.0;
    color = NSColor.systemOrangeColor;
    break;
  }
  NSImageSymbolConfiguration *size =
      [NSImageSymbolConfiguration configurationWithPointSize:pointSize weight:NSFontWeightSemibold];
  NSImageSymbolConfiguration *tint =
      [NSImageSymbolConfiguration configurationWithHierarchicalColor:color];
  return [[NSImage imageWithSystemSymbolName:symbolName accessibilityDescription:description]
      imageWithSymbolConfiguration:[size configurationByApplyingConfiguration:tint]];
}

@interface MiniMapAnnotation : NSObject <MKAnnotation>
@property(nonatomic) CLLocationCoordinate2D coordinate;
@property(nonatomic) AnnotationKind kind;
@end

@implementation MiniMapAnnotation
@end

/// Compact replacement for MapKit's comparatively large built-in scale. Its
/// length is derived from the current Web Mercator map rectangle rather than
/// from the terrain CRS, since it describes the displayed MapKit view itself.
@interface CompactMapScaleView : NSView
- (void)updateForMapView:(MKMapView *)mapView;
@end

@implementation CompactMapScaleView {
  double _barWidth;
  double _distanceMetres;
}

- (NSView *)hitTest:(NSPoint)point {
  (void)point;
  return nil;
}

- (void)updateForMapView:(MKMapView *)mapView {
  if (mapView.bounds.size.width <= 0.0) {
    return;
  }
  const double mapPointsPerDisplayPoint =
      mapView.visibleMapRect.size.width / mapView.bounds.size.width;
  const double mapPointsPerMetre = MKMapPointsPerMeterAtLatitude(mapView.centerCoordinate.latitude);
  if (!(mapPointsPerDisplayPoint > 0.0) || !(mapPointsPerMetre > 0.0)) {
    return;
  }
  const double metresPerDisplayPoint = mapPointsPerDisplayPoint / mapPointsPerMetre;
  constexpr double kMaximumBarWidth = 52.0;
  const double maximumDistance = metresPerDisplayPoint * kMaximumBarWidth;
  const double magnitude = std::pow(10.0, std::floor(std::log10(maximumDistance)));
  const double normalized = maximumDistance / magnitude;
  const double multiplier = normalized >= 5.0 ? 5.0 : (normalized >= 2.0 ? 2.0 : 1.0);
  _distanceMetres = multiplier * magnitude;
  _barWidth = _distanceMetres / metresPerDisplayPoint;
  self.needsDisplay = YES;
}

- (void)drawRect:(NSRect)dirtyRect {
  (void)dirtyRect;
  [[NSColor colorWithWhite:0.0 alpha:0.58] setFill];
  [[NSBezierPath bezierPathWithRoundedRect:self.bounds xRadius:5.0 yRadius:5.0] fill];

  const NSString *label = _distanceMetres >= 1'000.0
                              ? [NSString stringWithFormat:@"%.0f km", _distanceMetres / 1'000.0]
                              : [NSString stringWithFormat:@"%.0f m", _distanceMetres];
  NSDictionary<NSAttributedStringKey, id> *attributes = @{
    NSFontAttributeName : [NSFont systemFontOfSize:9.0 weight:NSFontWeightMedium],
    NSForegroundColorAttributeName : NSColor.whiteColor,
  };
  [label drawAtPoint:NSMakePoint(7.0, 11.0) withAttributes:attributes];

  const CGFloat start = 7.0;
  const CGFloat end = start + _barWidth;
  NSBezierPath *bar = [NSBezierPath bezierPath];
  bar.lineWidth = 1.0;
  [bar moveToPoint:NSMakePoint(start, 7.0)];
  [bar lineToPoint:NSMakePoint(end, 7.0)];
  [bar moveToPoint:NSMakePoint(start, 4.5)];
  [bar lineToPoint:NSMakePoint(start, 9.5)];
  [bar moveToPoint:NSMakePoint(end, 4.5)];
  [bar lineToPoint:NSMakePoint(end, 9.5)];
  [NSColor.whiteColor setStroke];
  [bar stroke];
}

@end

/// A stable MapKit overlay; the renderer invalidates only the old/new image
/// extents. MapKit supplies the normal map composition, with no second drawable.
@interface VisibilityImageOverlay : NSObject <MKOverlay>
@end
@implementation VisibilityImageOverlay
- (CLLocationCoordinate2D)coordinate {
  return CLLocationCoordinate2DMake(0, 0);
}
- (MKMapRect)boundingMapRect {
  return MKMapRectWorld;
}
@end

@interface VisibilityImageRenderer : MKOverlayRenderer {
  std::mutex _imageMutex;
  CGImageRef _image;
  MKMapRect _imageRect;
}
- (void)setImage:(CGImageRef)image mapRect:(MKMapRect)rect;
@end
@implementation VisibilityImageRenderer
- (void)dealloc {
  CGImageRelease(_image);
}
- (void)setImage:(CGImageRef)image mapRect:(MKMapRect)rect {
  MKMapRect dirty = rect;
  {
    std::lock_guard<std::mutex> lock(_imageMutex);
    if (_image != nullptr)
      dirty = MKMapRectUnion(dirty, _imageRect);
    CGImageRelease(_image);
    _image = CGImageRetain(image);
    _imageRect = rect;
  }
  if (!MKMapRectIsNull(dirty))
    [self setNeedsDisplayInMapRect:dirty];
}
- (void)drawMapRect:(MKMapRect)mapRect
          zoomScale:(MKZoomScale)zoomScale
          inContext:(CGContextRef)context {
  (void)zoomScale;
  CGImageRef image = nullptr;
  MKMapRect imageRect;
  {
    // MapKit may draw tiles concurrently. Retain an immutable image while
    // allowing the main thread to publish its replacement without waiting.
    std::lock_guard<std::mutex> lock(_imageMutex);
    image = CGImageRetain(_image);
    imageRect = _imageRect;
  }
  if (image == nullptr)
    return;
  if (MKMapRectIntersectsRect(mapRect, imageRect)) {
    const CGRect rect = [self rectForMapRect:imageRect];
    CGContextSaveGState(context);
    CGContextClipToRect(context, [self rectForMapRect:mapRect]);
    CGContextSetInterpolationQuality(context, kCGInterpolationNone);
    // Image rows and MapKit's map coordinates both start at the north edge;
    // CGContextDrawImage otherwise treats the first image row as the top of
    // a Cartesian (bottom-up) rectangle.
    CGContextTranslateCTM(context, CGRectGetMinX(rect), CGRectGetMaxY(rect));
    CGContextScaleCTM(context, 1.0, -1.0);
    CGContextDrawImage(context, CGRectMake(0, 0, rect.size.width, rect.size.height), image);
    CGContextRestoreGState(context);
  }
  CGImageRelease(image);
}
@end

struct VisibilityMaskRequest {
  id<MTLBuffer> points;
  panorama::app::VisibilityMaskParameters parameters;
  MKMapRect rect;
  uint64_t generation;
  panorama::app::VisibilityMapRegion region;
  double requested_at;
};

@interface MiniMapPanelView (MapInteraction)
- (void)mapPointerDidEnter;
- (void)mapPointerDidExit;
- (void)mapDidHoverCoordinate:(CLLocationCoordinate2D)coordinate;
- (void)mapDidEndHover;
- (void)mapDidSelectCoordinate:(CLLocationCoordinate2D)coordinate;
- (void)mapDidRequestObserverMoveCoordinate:(CLLocationCoordinate2D)coordinate;
- (void)showContextMenuForCoordinate:(CLLocationCoordinate2D)coordinate event:(NSEvent *)event;
@end

/// A freely navigable north-up minimap. MapKit owns panning and zooming; a
/// click recognizer preserves the separate terrain-point selection gesture.
@interface InteractiveMiniMapView : MKMapView {
@private
  NSTrackingArea *_interactionTrackingArea;
  NSPoint _lastHoverPoint;
  NSEventModifierFlags _mouseDownModifiers;
  bool _hasLastHoverPoint;
}
@property(nonatomic, weak) MiniMapPanelView *interactionOwner;
@end

@implementation InteractiveMiniMapView

- (instancetype)initWithFrame:(NSRect)frameRect {
  self = [super initWithFrame:frameRect];
  if (self == nil) {
    return nil;
  }
  NSClickGestureRecognizer *click =
      [[NSClickGestureRecognizer alloc] initWithTarget:self action:@selector(primaryClick:)];
  click.buttonMask = 0x1U;
  click.numberOfClicksRequired = 1;
  [self addGestureRecognizer:click];
  return self;
}

- (void)updateTrackingAreas {
  [super updateTrackingAreas];
  if (_interactionTrackingArea != nil) {
    [self removeTrackingArea:_interactionTrackingArea];
  }
  _interactionTrackingArea =
      [[NSTrackingArea alloc] initWithRect:NSZeroRect
                                   options:NSTrackingMouseMoved | NSTrackingMouseEnteredAndExited |
                                           NSTrackingActiveInKeyWindow | NSTrackingInVisibleRect
                                     owner:self
                                  userInfo:nil];
  [self addTrackingArea:_interactionTrackingArea];
}

- (CLLocationCoordinate2D)coordinateForEvent:(NSEvent *)event {
  const NSPoint point = [self convertPoint:event.locationInWindow fromView:nil];
  return [self convertPoint:point toCoordinateFromView:self];
}

- (void)mouseMoved:(NSEvent *)event {
  const NSPoint point = [self convertPoint:event.locationInWindow fromView:nil];
  // Terrain sampling may perform tile I/O. Ignore sub-three-point jitter while
  // still updating rapidly enough for the marker to follow deliberate motion.
  if (_hasLastHoverPoint &&
      std::hypot(point.x - _lastHoverPoint.x, point.y - _lastHoverPoint.y) < 3.0) {
    return;
  }
  _lastHoverPoint = point;
  _hasLastHoverPoint = true;
  [self.interactionOwner mapDidHoverCoordinate:[self coordinateForEvent:event]];
}

- (void)mouseEntered:(NSEvent *)event {
  (void)event;
  [self.interactionOwner mapPointerDidEnter];
}

- (void)mouseExited:(NSEvent *)event {
  (void)event;
  _hasLastHoverPoint = false;
  [self.interactionOwner mapPointerDidExit];
  [self.interactionOwner mapDidEndHover];
}

- (void)mouseDown:(NSEvent *)event {
  _hasLastHoverPoint = false;
  [self.interactionOwner mapDidEndHover];
  _mouseDownModifiers = event.modifierFlags;
  [super mouseDown:event];
}

- (void)primaryClick:(NSClickGestureRecognizer *)recognizer {
  if (recognizer.state != NSGestureRecognizerStateEnded) {
    return;
  }
  const NSPoint point = [recognizer locationInView:self];
  const CLLocationCoordinate2D coordinate = [self convertPoint:point toCoordinateFromView:self];
  if ((_mouseDownModifiers & NSEventModifierFlagOption) != 0U) {
    [self.interactionOwner mapDidRequestObserverMoveCoordinate:coordinate];
  } else {
    [self.interactionOwner mapDidSelectCoordinate:coordinate];
  }
}

- (void)rightMouseDown:(NSEvent *)event {
  [self.interactionOwner showContextMenuForCoordinate:[self coordinateForEvent:event] event:event];
}

@end

@interface MiniMapPanelView () <MKMapViewDelegate> {
@private
  NSView *_mapSection;
  NSView *_pointInfoView;
  InteractiveMiniMapView *_mapView;
  VisibilityImageOverlay *_visibilityOverlay;
  VisibilityImageRenderer *_visibilityRenderer;
  std::shared_ptr<panorama::app::VisibilityMask> _visibilityMask;
  dispatch_queue_t _maskQueue;
  id<MTLBuffer> _visibilityPoints;
  panorama::ImageSize _visibilityImage;
  std::optional<VisibilityMaskRequest> _pendingMask;
  std::optional<VisibilityMaskRequest> _lastMask;
  bool _maskActive;
  uint64_t _visibilityGeneration;
  panorama::CameraOrientation _cameraOrientation;
  double _cameraFieldOfView;
  panorama::ImageSize _cameraImage;
  bool _cameraDirty;
  bool _observerDirty;
  bool _centrePending;
  CompactMapScaleView *_scaleView;
  NSPopUpButton *_mapStyleControl;
  NSButton *_coverageControl;
  NSButton *_returnToObserverControl;
  NSButton *_mapFocusControl;
  NSButton *_mapSizeControl;
  MiniMapAnnotation *_observerAnnotation;
  MiniMapAnnotation *_inspectionAnnotation;
  id<MKOverlay> _fieldOfViewOverlay;
  id<MKOverlay> _headingOverlay;
  MKTileOverlay *_tileOverlay;
  MKMultiPolygon *_coverageOverlay;
  double _observerEasting;
  double _observerNorthing;
  double _maxDistance;
  CGFloat _pointInfoHeight;
  uint32_t _terrainEpsgCode;
  __weak id<MiniMapPanelViewSizeDelegate> _sizeDelegate;
  __weak id<MiniMapPanelViewInteractionDelegate> _interactionDelegate;
  CLLocationCoordinate2D _contextCoordinate;
  bool _contentVisible;
  bool _coverageVisible;
  bool _followInspection;
  bool _mapPointerInside;
  bool _largeMap;
}
- (void)updateVisibilityTransform;
- (void)startPendingMask;
- (void)updateObserverGraphics;
- (void)updateCameraGraphics;
- (void)updateCoverageControl;
- (void)updateMapFocusControl;
- (void)updateMapSizeControl;
@end

@implementation MiniMapPanelView

@synthesize sizeDelegate = _sizeDelegate;
@synthesize interactionDelegate = _interactionDelegate;

- (instancetype)initWithObserverEasting:(double)easting
                               northing:(double)northing
                        terrainEpsgCode:(uint32_t)epsgCode
                        terrainCoverage:(const panorama::TerrainCoverage &)coverage
               coverageInitiallyVisible:(bool)coverageVisible
                            maxDistance:(double)maxDistance
                          pointInfoView:(NSView *)pointInfoView
                            metalDevice:(id<MTLDevice>)metalDevice
                           commandQueue:(id<MTLCommandQueue>)commandQueue
                                library:(id<MTLLibrary>)library {
  self =
      [super initWithFrame:NSMakeRect(0.0, 0.0, kCompactMapPanelWidth, kCompactMapSectionHeight)];
  if (self == nil) {
    return nil;
  }

  _observerEasting = easting;
  _observerNorthing = northing;
  _terrainEpsgCode = epsgCode;
  _maxDistance = maxDistance;
  _pointInfoHeight = kMinimumPointSectionHeight;
  _pointInfoView = pointInfoView;

  _mapSection = [[NSView alloc] initWithFrame:NSZeroRect];
  [self addSubview:_mapSection];
  [self addSubview:_pointInfoView];

  _mapStyleControl = [[NSPopUpButton alloc] initWithFrame:NSZeroRect pullsDown:NO];
  [_mapStyleControl addItemsWithTitles:@[
    @"Standard Terrain",
    @"Hybrid",
    @"Imagery",
  ]];
  for (auto title_url : kTileOverlays) {
    [_mapStyleControl addItemsWithTitles:@[ title_url.first ]];
  }
  _tileOverlay = nil;
  _mapStyleControl.target = self;
  _mapStyleControl.action = @selector(mapStyleChanged:);

  _coverageControl =
      [NSButton buttonWithImage:[NSImage imageWithSystemSymbolName:@"square.grid.3x3"
                                          accessibilityDescription:@"Show terrain coverage"]
                         target:self
                         action:@selector(toggleCoverage:)];
  _coverageControl.title = @"";
  _coverageControl.buttonType = NSButtonTypeToggle;
  _coverageControl.bordered = NO;
  _coverageControl.imagePosition = NSImageOnly;
  [_coverageControl.widthAnchor constraintEqualToConstant:22.0].active = YES;
  [_coverageControl.heightAnchor constraintEqualToConstant:22.0].active = YES;

  _returnToObserverControl =
      [NSButton buttonWithImage:[NSImage imageWithSystemSymbolName:@"location.fill"
                                          accessibilityDescription:@"Return to observer"]
                         target:self
                         action:@selector(returnToObserver:)];
  _returnToObserverControl.title = @"";
  _returnToObserverControl.bordered = NO;
  _returnToObserverControl.imagePosition = NSImageOnly;
  _returnToObserverControl.toolTip = @"Return to observer without changing map scale";
  [_returnToObserverControl setAccessibilityLabel:_returnToObserverControl.toolTip];
  [_returnToObserverControl.widthAnchor constraintEqualToConstant:22.0].active = YES;
  [_returnToObserverControl.heightAnchor constraintEqualToConstant:22.0].active = YES;

  _mapFocusControl = [NSButton
      buttonWithImage:[NSImage imageWithSystemSymbolName:@"scope"
                                accessibilityDescription:@"Center minimap on mouseover terrain"]
               target:self
               action:@selector(toggleMapFocus:)];
  _mapFocusControl.title = @"";
  _mapFocusControl.buttonType = NSButtonTypeToggle;
  _mapFocusControl.bordered = NO;
  _mapFocusControl.imagePosition = NSImageOnly;
  [_mapFocusControl.widthAnchor constraintEqualToConstant:22.0].active = YES;
  [_mapFocusControl.heightAnchor constraintEqualToConstant:22.0].active = YES;

  _mapSizeControl = [NSButton
      buttonWithImage:[NSImage imageWithSystemSymbolName:@"arrow.down.left.and.arrow.up.right"
                                accessibilityDescription:@"Enlarge minimap"]
               target:self
               action:@selector(toggleMapSize:)];
  _mapSizeControl.title = @"";
  _mapSizeControl.bordered = NO;
  _mapSizeControl.imagePosition = NSImageOnly;
  [_mapSizeControl.widthAnchor constraintEqualToConstant:22.0].active = YES;
  [_mapSizeControl.heightAnchor constraintEqualToConstant:22.0].active = YES;

  NSView *controlSpacer = [[NSView alloc] initWithFrame:NSZeroRect];
  [controlSpacer setContentHuggingPriority:NSLayoutPriorityDefaultLow
                            forOrientation:NSLayoutConstraintOrientationHorizontal];
  [controlSpacer setContentCompressionResistancePriority:NSLayoutPriorityDefaultLow
                                          forOrientation:NSLayoutConstraintOrientationHorizontal];

  NSStackView *controls = [NSStackView stackViewWithViews:@[
    _mapStyleControl,
    controlSpacer,
    _coverageControl,
    _returnToObserverControl,
    _mapFocusControl,
    _mapSizeControl,
  ]];
  controls.orientation = NSUserInterfaceLayoutOrientationHorizontal;
  controls.alignment = NSLayoutAttributeCenterY;
  controls.spacing = 4.0;
  [_mapStyleControl setContentHuggingPriority:NSLayoutPriorityDefaultHigh
                               forOrientation:NSLayoutConstraintOrientationHorizontal];
  controls.translatesAutoresizingMaskIntoConstraints = NO;
  [_mapSection addSubview:controls];

  _mapView = [[InteractiveMiniMapView alloc] initWithFrame:NSZeroRect];
  _mapView.interactionOwner = self;
  _mapView.delegate = self;
  _mapView.scrollEnabled = YES;
  _mapView.zoomEnabled = YES;
  _mapView.rotateEnabled = NO;
  _mapView.pitchEnabled = NO;
  _mapView.showsCompass = NO;
  _mapView.showsScale = NO;
  _mapView.wantsLayer = YES;
  _mapView.layer.cornerRadius = 10.0;
  _mapView.layer.masksToBounds = YES;
  _mapView.translatesAutoresizingMaskIntoConstraints = NO;
  [_mapSection addSubview:_mapView];

  _visibilityMask =
      std::make_shared<panorama::app::VisibilityMask>(metalDevice, commandQueue, library);
  _maskQueue = dispatch_queue_create("panorama.minimap-mask", DISPATCH_QUEUE_SERIAL);
  _visibilityOverlay = [[VisibilityImageOverlay alloc] init];
  _visibilityRenderer = [[VisibilityImageRenderer alloc] initWithOverlay:_visibilityOverlay];
  [_mapView addOverlay:_visibilityOverlay level:MKOverlayLevelAboveLabels];

  _scaleView = [[CompactMapScaleView alloc] initWithFrame:NSMakeRect(8.0, 8.0, 66.0, 28.0)];
  _scaleView.autoresizingMask = NSViewMaxXMargin | NSViewMaxYMargin;
  [_mapView addSubview:_scaleView];

  [NSLayoutConstraint activateConstraints:@[
    [controls.topAnchor constraintEqualToAnchor:_mapSection.topAnchor constant:12.0],
    [controls.leadingAnchor constraintEqualToAnchor:_mapSection.leadingAnchor constant:14.0],
    [controls.trailingAnchor constraintEqualToAnchor:_mapSection.trailingAnchor constant:-12.0],
    [_mapView.topAnchor constraintEqualToAnchor:controls.bottomAnchor constant:8.0],
    [_mapView.leadingAnchor constraintEqualToAnchor:_mapSection.leadingAnchor constant:12.0],
    [_mapView.trailingAnchor constraintEqualToAnchor:_mapSection.trailingAnchor constant:-12.0],
    [_mapView.bottomAnchor constraintEqualToAnchor:_mapSection.bottomAnchor constant:-6.0],
  ]];

  // Terrain and ray geometry use the dataset's projected CRS (for example,
  // EPSG:2056 Swiss LV95), whereas MapKit accepts WGS84 latitude/longitude.
  // The observer is therefore transformed before becoming the initial map
  // centre. Camera headings must *not* simply be treated as geographic
  // bearings: setCameraOrientation constructs endpoints in the terrain CRS
  // first and transforms those complete points to WGS84, preserving the
  // projected grid's local convergence relative to true north.
  const panorama::Crs terrainCrs = panorama::Crs::from_epsg(_terrainEpsgCode);
  const panorama::LatLon observerLatLon =
      terrainCrs.to_lat_lon({_observerEasting, _observerNorthing});
  const CLLocationCoordinate2D observerCoordinate =
      CLLocationCoordinate2DMake(observerLatLon.lat, observerLatLon.lon);
  [_mapView setRegion:MKCoordinateRegionMakeWithDistance(
                          observerCoordinate,
                          kInitialMapDistance,
                          kInitialMapDistance
                      )
             animated:NO];

  _observerAnnotation = [[MiniMapAnnotation alloc] init];
  _observerAnnotation.kind = AnnotationKind::Observer;
  _observerAnnotation.coordinate = observerCoordinate;
  [_mapView addAnnotation:_observerAnnotation];

  NSMutableArray<MKPolygon *> *coveragePolygons =
      [NSMutableArray arrayWithCapacity:coverage.tiles.size()];
  for (const panorama::TileKey key : coverage.tiles) {
    const double xMinimum =
        coverage.grid.origin_x + static_cast<double>(key.column) * coverage.grid.width;
    const double yMaximum =
        coverage.grid.origin_y - static_cast<double>(key.row) * coverage.grid.width;
    const double xMaximum = xMinimum + coverage.grid.width;
    const double yMinimum = yMaximum - coverage.grid.width;
    const auto mapCoordinate = [&](double x, double y) {
      const panorama::LatLon point = terrainCrs.to_lat_lon({x, y});
      return CLLocationCoordinate2DMake(point.lat, point.lon);
    };
    CLLocationCoordinate2D corners[4] = {
        mapCoordinate(xMinimum, yMinimum),
        mapCoordinate(xMinimum, yMaximum),
        mapCoordinate(xMaximum, yMaximum),
        mapCoordinate(xMaximum, yMinimum),
    };
    [coveragePolygons addObject:[MKPolygon polygonWithCoordinates:corners count:4U]];
  }
  _coverageOverlay = [[MKMultiPolygon alloc] initWithPolygons:coveragePolygons];
  [self mapStyleChanged:_mapStyleControl];

  _contentVisible = false;
  _coverageVisible = coverageVisible;
  _followInspection = false;
  _mapPointerInside = false;
  _largeMap = false;

  if (_coverageVisible) {
    [_mapView insertOverlay:_coverageOverlay belowOverlay:_visibilityOverlay];
  }
  [self updateCoverageControl];
  [self updateMapFocusControl];
  [self updateMapSizeControl];
  _mapSection.hidden = YES;
  _pointInfoView.hidden = YES;
  return self;
}

- (void)updateCoverageControl {
  NSString *symbol = _coverageVisible ? @"square.grid.3x3.fill" : @"square.grid.3x3";
  NSString *description = _coverageVisible ? @"Hide terrain coverage" : @"Show terrain coverage";
  _coverageControl.image = [NSImage imageWithSystemSymbolName:symbol
                                     accessibilityDescription:description];
  _coverageControl.state = _coverageVisible ? NSControlStateValueOn : NSControlStateValueOff;
  _coverageControl.toolTip = description;
  [_coverageControl setAccessibilityLabel:description];
}

- (void)toggleCoverage:(id)sender {
  (void)sender;
  _coverageVisible = !_coverageVisible;
  if (_coverageVisible) {
    [_mapView insertOverlay:_coverageOverlay belowOverlay:_visibilityOverlay];
  } else {
    [_mapView removeOverlay:_coverageOverlay];
  }
  [self updateCoverageControl];
}

- (void)returnToObserver:(id)sender {
  (void)sender;
  [_mapView setCenterCoordinate:_observerAnnotation.coordinate animated:YES];
}

- (void)centerOnObserver {
  if (!_contentVisible) {
    _centrePending = true;
    return;
  }
  [_mapView setCenterCoordinate:_observerAnnotation.coordinate animated:NO];
}

- (void)layout {
  [super layout];
  const CGFloat pointHeight = _contentVisible ? _pointInfoHeight : 0.0;
  _pointInfoView.frame = NSMakeRect(0.0, 0.0, self.bounds.size.width, pointHeight);
  _mapSection.frame = NSMakeRect(
      0.0,
      pointHeight,
      self.bounds.size.width,
      std::max(0.0, self.bounds.size.height - pointHeight)
  );
  if (_contentVisible) {
    [_scaleView updateForMapView:_mapView];
    [self updateVisibilityTransform];
  }
}

- (void)setMapAndPointInfoVisible:(bool)visible {
  if (_contentVisible == visible)
    return;
  _contentVisible = visible;
  ++_visibilityGeneration;
  _pendingMask.reset();
  _lastMask.reset();
  _visibilityPoints = nil;
  _mapSection.hidden = !visible;
  _pointInfoView.hidden = !visible;
  [_visibilityRenderer setImage:nullptr mapRect:MKMapRectNull];
  if (visible) {
    [self updateObserverGraphics];
    [self updateCameraGraphics];
    if (_centrePending) {
      _centrePending = false;
      [self centerOnObserver];
    }
    [self informationFooterContentDidChange];
  } else if (!_maskActive) {
    _visibilityMask->clear();
  }
  [self setNeedsLayout:YES];
}

- (NSSize)preferredPanelSize {
  const CGFloat mapWidth = _largeMap ? kLargeMapPanelWidth : kCompactMapPanelWidth;
  const CGFloat mapHeight = _largeMap ? kLargeMapSectionHeight : kCompactMapSectionHeight;
  return NSMakeSize(mapWidth, _contentVisible ? mapHeight + _pointInfoHeight : 0.0);
}

- (void)informationFooterContentDidChange {
  if (!_contentVisible)
    return;
  [_pointInfoView layoutSubtreeIfNeeded];
  const CGFloat nextHeight =
      std::max(kMinimumPointSectionHeight, _pointInfoView.fittingSize.height);
  if (std::abs(nextHeight - _pointInfoHeight) < 0.5) {
    return;
  }
  _pointInfoHeight = nextHeight;
  [self setNeedsLayout:YES];
  [_sizeDelegate miniMapPanelPreferredSizeDidChange:self];
}

- (void)updateMapSizeControl {
  NSString *symbol =
      _largeMap ? @"arrow.up.right.and.arrow.down.left" : @"arrow.down.left.and.arrow.up.right";
  NSString *description = _largeMap ? @"Restore compact minimap" : @"Enlarge minimap";
  _mapSizeControl.image = [NSImage imageWithSystemSymbolName:symbol
                                    accessibilityDescription:description];
  _mapSizeControl.toolTip = description;
  [_mapSizeControl setAccessibilityLabel:description];
}

- (void)updateMapFocusControl {
  NSString *description = _followInspection ? @"Stop following the panorama mouseover point"
                                            : @"Follow the panorama mouseover point";
  _mapFocusControl.image = [NSImage imageWithSystemSymbolName:@"scope"
                                     accessibilityDescription:description];
  _mapFocusControl.state = _followInspection ? NSControlStateValueOn : NSControlStateValueOff;
  _mapFocusControl.contentTintColor =
      _followInspection ? NSColor.controlAccentColor : NSColor.secondaryLabelColor;
  _mapFocusControl.toolTip = description;
  [_mapFocusControl setAccessibilityLabel:description];
}

- (void)toggleMapFocus:(id)sender {
  (void)sender;
  _followInspection = !_followInspection;
  [self updateMapFocusControl];
  if (_followInspection && !_mapPointerInside && _inspectionAnnotation != nil) {
    [_mapView setCenterCoordinate:_inspectionAnnotation.coordinate animated:NO];
  }
}

- (void)toggleMapSize:(id)sender {
  (void)sender;
  _largeMap = !_largeMap;
  [self updateMapSizeControl];
  [_sizeDelegate miniMapPanelPreferredSizeDidChange:self];
}

- (void)mapStyleChanged:(id)sender {
  (void)sender;
  if (_tileOverlay != nil) {
    [_mapView removeOverlay:_tileOverlay];
    _tileOverlay = nil;
  }
  _mapView.pointOfInterestFilter = nil;
  switch (_mapStyleControl.indexOfSelectedItem) {
  case 0:
    _mapView.preferredConfiguration =
        [[MKStandardMapConfiguration alloc] initWithElevationStyle:MKMapElevationStyleRealistic];
    break;
  case 1:
    _mapView.preferredConfiguration =
        [[MKHybridMapConfiguration alloc] initWithElevationStyle:MKMapElevationStyleRealistic];
    break;
  case 2:
    _mapView.preferredConfiguration =
        [[MKImageryMapConfiguration alloc] initWithElevationStyle:MKMapElevationStyleRealistic];
    break;
  // all other styles are custom tile overlays
  default:
    // set map to use WGS:3857 and turn off the standard apple labels
    _mapView.preferredConfiguration =
        [[MKStandardMapConfiguration alloc] initWithElevationStyle:MKMapElevationStyleRealistic];
    _mapView.pointOfInterestFilter = [MKPointOfInterestFilter filterExcludingAllCategories];

    const size_t index = _mapStyleControl.indexOfSelectedItem - 3;
    if (index >= kTileOverlays.size()) {
      throw std::invalid_argument(
          std::format("Map style index '{}' not handled", _mapStyleControl.indexOfSelectedItem)
      );
    }
    _tileOverlay = [[MKTileOverlay alloc] initWithURLTemplate:kTileOverlays[index].second];
    _tileOverlay.canReplaceMapContent = YES;
    [_mapView insertOverlay:_tileOverlay belowOverlay:_fieldOfViewOverlay];

    break;
  }
}

- (void)mapViewDidChangeVisibleRegion:(MKMapView *)mapView {
  if (!_contentVisible)
    return;
  [_scaleView updateForMapView:mapView];
  [self updateVisibilityTransform];
}

/// Capture geometry on AppKit; CRS grid construction happens on the worker.
/// The image retains its geographical rectangle while a newer job is pending.
- (void)updateVisibilityTransform {
  if (!_contentVisible || _visibilityPoints == nil)
    return;
  const NSSize size = [_mapView convertRectToBacking:_mapView.bounds].size;
  const MKMapRect rect = _mapView.visibleMapRect;
  if (size.width <= 0 || size.height <= 0 || rect.size.width <= 0 || rect.size.height <= 0)
    return;
  const uint32_t width = uint32_t(std::clamp(std::ceil(size.width), 1.0, 4096.0));
  const uint32_t height = uint32_t(std::clamp(std::ceil(size.height), 1.0, 4096.0));
  const panorama::app::VisibilityMaskParameters parameters = {
      0,
      0,
      0,
      0,
      0,
      0,
      width,
      height,
      _visibilityImage.width * _visibilityImage.height,
  };
  const panorama::app::VisibilityMapRegion region = {
      rect.origin.x,
      rect.origin.y,
      rect.size.width,
      rect.size.height,
      _observerEasting,
      _observerNorthing,
      _maxDistance,
      _terrainEpsgCode,
  };
  if (_lastMask && _lastMask->points == _visibilityPoints &&
      MKMapRectEqualToRect(_lastMask->rect, rect) &&
      std::memcmp(&_lastMask->parameters, &parameters, sizeof(parameters)) == 0)
    return;
  _pendingMask = VisibilityMaskRequest{_visibilityPoints,
                                       parameters,
                                       rect,
                                       _visibilityGeneration,
                                       region,
                                       panorama::app::diagnostics::seconds()};
  _lastMask = _pendingMask;
  [self startPendingMask];
}

- (void)startPendingMask {
  if (_maskActive || !_contentVisible || !_pendingMask)
    return;
  const VisibilityMaskRequest request = *_pendingMask;
  _pendingMask.reset();
  _maskActive = true;
  const auto mask = _visibilityMask;
  __weak MiniMapPanelView *weakSelf = self;
  // Only one block is submitted at a time, including its main-queue delivery.
  // Input while it runs replaces the single pending request, never a FIFO.
  dispatch_async(_maskQueue, ^{
    @autoreleasepool {
      CGImageRef image = nullptr;
      try {
        image = mask->render(request.points, request.parameters, &request.region);
      } catch (const std::exception &error) {
        std::fprintf(stderr, "Minimap visibility: %s\n", error.what());
      }
      dispatch_async(dispatch_get_main_queue(), ^{
        MiniMapPanelView *panel = weakSelf;
        if (panel != nil) {
          panel->_maskActive = false;
          if (panel->_contentVisible && request.generation == panel->_visibilityGeneration) {
            if (image != nullptr) {
              [panel->_visibilityRenderer setImage:image mapRect:request.rect];
              if (panorama::app::diagnostics::enabled) {
                ++panorama::app::diagnostics::minimap.presented;
                std::printf(
                    "Minimap publish: request-to-image %.3f ms\n",
                    (panorama::app::diagnostics::seconds() - request.requested_at) * 1000
                );
              }
            } else {
              panel->_lastMask.reset();
            }
          } else if (panorama::app::diagnostics::enabled) {
            ++panorama::app::diagnostics::minimap.stale;
          }
          if (!panel->_contentVisible)
            mask->clear();
          [panel startPendingMask];
        }
        CGImageRelease(image);
      });
    }
  });
}

- (void)setCameraOrientation:(panorama::CameraOrientation)orientation
         verticalFieldOfView:(double)verticalFieldOfView
                       image:(panorama::ImageSize)image {
  if (_cameraOrientation.heading == orientation.heading &&
      _cameraFieldOfView == verticalFieldOfView && _cameraImage.width == image.width &&
      _cameraImage.height == image.height)
    return;
  _cameraOrientation = orientation;
  _cameraFieldOfView = verticalFieldOfView;
  _cameraImage = image;
  _cameraDirty = true;
  [self updateCameraGraphics];
}

- (void)updateCameraGraphics {
  if (!_contentVisible || !_cameraDirty)
    return;
  const auto orientation = _cameraOrientation;
  const auto image = _cameraImage;
  const double verticalFieldOfView = _cameraFieldOfView;
  _cameraDirty = false;
  if (image.width == 0U || image.height == 0U) {
    return;
  }
  const double aspect = static_cast<double>(image.width) / image.height;
  const double horizontalFieldOfView =
      2.0 * std::atan(std::tan(verticalFieldOfView * 0.5) * aspect);
  const double leftHeading = orientation.heading - horizontalFieldOfView * 0.5;
  const double rightHeading = orientation.heading + horizontalFieldOfView * 0.5;

  // Heading is clockwise from projected grid north. Build both far endpoints
  // as (easting, northing) offsets in that same terrain CRS, then transform the
  // actual coordinates to WGS84 for MapKit. Converting heading directly to a
  // WGS84 bearing would silently ignore grid convergence and rotate the wedge
  // away from the rays for projected CRSs such as Swiss LV95.
  const auto endpoint = [&](double heading) {
    return panorama::Coord{
        _observerEasting + _maxDistance * std::sin(heading),
        _observerNorthing + _maxDistance * std::cos(heading),
    };
  };
  const panorama::Crs terrainCrs = panorama::Crs::from_epsg(_terrainEpsgCode);
  const auto mapCoordinate = [&](panorama::Coord projected) {
    const panorama::LatLon geographic = terrainCrs.to_lat_lon(projected);
    return CLLocationCoordinate2DMake(geographic.lat, geographic.lon);
  };
  CLLocationCoordinate2D wedge[3] = {
      _observerAnnotation.coordinate,
      mapCoordinate(endpoint(leftHeading)),
      mapCoordinate(endpoint(rightHeading)),
  };
  CLLocationCoordinate2D headingLine[2] = {
      _observerAnnotation.coordinate,
      mapCoordinate(endpoint(orientation.heading)),
  };

  // Install replacements before removing the old geometry. MapKit draws
  // overlays asynchronously, so remove-then-add creates a visible blank frame
  // while cruise updates the observer and camera continuously.
  MKPolygon *nextFieldOfView = [MKPolygon polygonWithCoordinates:wedge count:3];
  MKPolyline *nextHeading = [MKPolyline polylineWithCoordinates:headingLine count:2];
  [_mapView insertOverlay:nextFieldOfView belowOverlay:_visibilityOverlay];
  [_mapView addOverlay:nextHeading level:MKOverlayLevelAboveLabels];
  if (_fieldOfViewOverlay != nil)
    [_mapView removeOverlay:_fieldOfViewOverlay];
  if (_headingOverlay != nil)
    [_mapView removeOverlay:_headingOverlay];
  _fieldOfViewOverlay = nextFieldOfView;
  _headingOverlay = nextHeading;
}

- (void)setVisibilityPoints:(id<MTLBuffer>)points image:(panorama::ImageSize)image {
  if (!_contentVisible)
    return;
  const uint64_t count = uint64_t(image.width) * image.height;
  if (points == nil || count == 0 || count > UINT32_MAX || points.length < count * 8) {
    if (_visibilityPoints != nil) {
      ++_visibilityGeneration;
      _visibilityPoints = nil;
      _pendingMask.reset();
      _lastMask.reset();
      [_visibilityRenderer setImage:nullptr mapRect:MKMapRectNull];
    }
    return;
  }
  _visibilityPoints = points;
  _visibilityImage = image;
  [self updateVisibilityTransform];
}

- (panorama::Coord)projectedCoordinate:(CLLocationCoordinate2D)coordinate {
  const panorama::Crs terrainCrs = panorama::Crs::from_epsg(_terrainEpsgCode);
  return terrainCrs.from_lat_lon({coordinate.latitude, coordinate.longitude});
}

- (void)mapPointerDidEnter {
  _mapPointerInside = true;
}

- (void)mapPointerDidExit {
  _mapPointerInside = false;
}

- (void)mapDidHoverCoordinate:(CLLocationCoordinate2D)coordinate {
  _mapPointerInside = true;
  if (_interactionDelegate == nil) {
    return;
  }
  const panorama::Coord projected = [self projectedCoordinate:coordinate];
  [_interactionDelegate miniMapPanel:self didHoverEasting:projected.x northing:projected.y];
}

- (void)mapDidEndHover {
  [_interactionDelegate miniMapPanelDidEndHover:self];
}

- (void)mapDidSelectCoordinate:(CLLocationCoordinate2D)coordinate {
  const panorama::Coord projected = [self projectedCoordinate:coordinate];
  [_interactionDelegate miniMapPanel:self didSelectEasting:projected.x northing:projected.y];
}

- (void)mapDidRequestObserverMoveCoordinate:(CLLocationCoordinate2D)coordinate {
  const panorama::Coord projected = [self projectedCoordinate:coordinate];
  [_interactionDelegate miniMapPanel:self
      didRequestObserverMoveToEasting:projected.x
                             northing:projected.y];
}

- (void)showContextMenuForCoordinate:(CLLocationCoordinate2D)coordinate event:(NSEvent *)event {
  _contextCoordinate = coordinate;
  NSMenu *menu = [[NSMenu alloc] initWithTitle:@""];
  NSMenuItem *move = [[NSMenuItem alloc] initWithTitle:@"Move Observer Here"
                                                action:@selector(moveObserverFromContextMenu:)
                                         keyEquivalent:@""];
  move.target = self;
  [menu addItem:move];
  const NSPoint location = [_mapView convertPoint:event.locationInWindow fromView:nil];
  [menu popUpMenuPositioningItem:nil atLocation:location inView:_mapView];
}

- (void)moveObserverFromContextMenu:(id)sender {
  (void)sender;
  [self mapDidRequestObserverMoveCoordinate:_contextCoordinate];
}

- (void)setObserverEasting:(double)easting northing:(double)northing {
  if (_observerEasting == easting && _observerNorthing == northing)
    return;
  _observerEasting = easting;
  _observerNorthing = northing;
  _observerDirty = true;
  _cameraDirty = true;
  ++_visibilityGeneration;
  _visibilityPoints = nil;
  _pendingMask.reset();
  _lastMask.reset();
  // Keep the last complete mask visible while the replacement is rendered on
  // the serial worker. Its immutable map rectangle remains geographically
  // valid as the map follows the observer; generation still rejects work that
  // completed for an older observer.
  [self updateObserverGraphics];
  [self updateCameraGraphics];
}

- (void)updateObserverGraphics {
  if (!_contentVisible || !_observerDirty)
    return;
  _observerDirty = false;
  const double easting = _observerEasting, northing = _observerNorthing;
  const panorama::Crs terrainCrs = panorama::Crs::from_epsg(_terrainEpsgCode);
  const panorama::LatLon observer = terrainCrs.to_lat_lon({easting, northing});
  const CLLocationCoordinate2D coordinate = CLLocationCoordinate2DMake(observer.lat, observer.lon);
  _observerAnnotation.coordinate = coordinate;

  [self updateVisibilityTransform];
}

- (void)setInspectedPointEasting:(double)easting northing:(double)northing locked:(bool)locked {
  if (!_contentVisible)
    return;
  const panorama::Crs terrainCrs = panorama::Crs::from_epsg(_terrainEpsgCode);
  const panorama::LatLon geographic = terrainCrs.to_lat_lon({easting, northing});
  const CLLocationCoordinate2D coordinate =
      CLLocationCoordinate2DMake(geographic.lat, geographic.lon);
  if (_inspectionAnnotation == nil) {
    _inspectionAnnotation = [[MiniMapAnnotation alloc] init];
    _inspectionAnnotation.kind = locked ? AnnotationKind::Locked : AnnotationKind::Hover;
    _inspectionAnnotation.coordinate = coordinate;
    [_mapView addAnnotation:_inspectionAnnotation];
  } else {
    _inspectionAnnotation.kind = locked ? AnnotationKind::Locked : AnnotationKind::Hover;
    _inspectionAnnotation.coordinate = coordinate;
  }
  MKAnnotationView *view = [_mapView viewForAnnotation:_inspectionAnnotation];
  if (view != nil) {
    view.image = annotation_image(_inspectionAnnotation.kind);
    view.alphaValue = 1.0;
  }
  // Following a point sampled from the panorama is useful; following a point
  // sampled from the map itself creates a feedback loop because recentering
  // changes the coordinate beneath the stationary pointer.
  if (_followInspection && !_mapPointerInside) {
    [_mapView setCenterCoordinate:coordinate animated:NO];
  }
}

- (void)clearInspectedPoint {
  if (_inspectionAnnotation != nil) {
    [_mapView removeAnnotation:_inspectionAnnotation];
    _inspectionAnnotation = nil;
  }
}

- (MKAnnotationView *)mapView:(MKMapView *)mapView viewForAnnotation:(id<MKAnnotation>)annotation {
  (void)mapView;
  if (![annotation isKindOfClass:MiniMapAnnotation.class]) {
    return nil;
  }
  MiniMapAnnotation *marker = (MiniMapAnnotation *)annotation;
  MKAnnotationView *view = [[MKAnnotationView alloc] initWithAnnotation:annotation
                                                        reuseIdentifier:nil];
  view.canShowCallout = NO;
  view.image = annotation_image(marker.kind);
  view.alphaValue = 1.0;
  return view;
}

- (MKOverlayRenderer *)mapView:(MKMapView *)mapView rendererForOverlay:(id<MKOverlay>)overlay {
  (void)mapView;
  if (overlay == _visibilityOverlay)
    return _visibilityRenderer;
  if (overlay == _coverageOverlay) {
    MKMultiPolygonRenderer *renderer =
        [[MKMultiPolygonRenderer alloc] initWithMultiPolygon:_coverageOverlay];
    renderer.fillColor = [NSColor.systemRedColor colorWithAlphaComponent:0.22];
    renderer.strokeColor = NSColor.clearColor;
    renderer.lineWidth = 0.0;
    return renderer;
  }
  if ([overlay isKindOfClass:MKPolygon.class]) {
    MKPolygonRenderer *renderer = [[MKPolygonRenderer alloc] initWithPolygon:(MKPolygon *)overlay];
    renderer.fillColor = [NSColor.systemBlueColor colorWithAlphaComponent:0.14];
    renderer.strokeColor = NSColor.clearColor;
    renderer.lineWidth = 0.0;
    return renderer;
  }
  if ([overlay isKindOfClass:MKPolyline.class]) {
    MKPolylineRenderer *renderer =
        [[MKPolylineRenderer alloc] initWithPolyline:(MKPolyline *)overlay];
    renderer.strokeColor = NSColor.systemBlueColor;
    renderer.lineWidth = 2.0;
    return renderer;
  }
  if ([overlay isKindOfClass:MKTileOverlay.class]) {
    MKTileOverlayRenderer *renderer =
        [[MKTileOverlayRenderer alloc] initWithTileOverlay:(MKTileOverlay *)overlay];

    if (!overlay.canReplaceMapContent) {
      renderer.alpha = 0.4;
    }
    return renderer;
  }
  throw std::invalid_argument("Overlay type not handled");
}

@end

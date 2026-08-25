#include "minimap.h"

#include "crs.h"

#import <MapKit/MapKit.h>
#import <QuartzCore/QuartzCore.h>

#include <algorithm>
#include <cmath>

constexpr CGFloat kMapPanelWidth = 300.0;
constexpr CGFloat kMapSectionHeight = 286.0;
constexpr CGFloat kPointSectionHeight = 82.0;
constexpr double kInitialMapDistance = 50'000.0;
constexpr double kMinimumMapDistance = 500.0;
constexpr double kMaximumMapDistance = 2'000'000.0;

enum class AnnotationKind : NSInteger {
  Observer,
  Hover,
  Locked,
};

/// Return a small symbol-only marker rather than MapKit's full pin balloon.
[[nodiscard]] NSImage *annotation_image(AnnotationKind kind) {
  NSString *symbolName = nil;
  NSString *description = nil;
  CGFloat pointSize = 0.0;
  NSColor *color = nil;
  switch (kind) {
  case AnnotationKind::Observer:
    symbolName = @"location.fill";
    description = @"Observer";
    pointSize = 11.0;
    color = NSColor.systemPurpleColor;
    break;
  case AnnotationKind::Hover:
    symbolName = @"circle.fill";
    description = @"Inspected Terrain Point";
    pointSize = 7.0;
    color = NSColor.systemBlueColor;
    break;
  case AnnotationKind::Locked:
    symbolName = @"circle.circle.fill";
    description = @"Locked Terrain Point";
    pointSize = 10.0;
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

/// A north-up map whose centre is always the observer. MapKit's normal pan,
/// pitch, and rotation gestures are disabled; scrolling and pinching change
/// only the stored ground span, so navigation cannot lose the viewpoint.
@interface FixedCentreMapView : MKMapView
@property(nonatomic) CLLocationCoordinate2D fixedCentre;
@property(nonatomic) double visibleDistance;
@end

@implementation FixedCentreMapView

- (void)applyVisibleDistanceAnimated:(BOOL)animated {
  const double distance =
      std::clamp(self.visibleDistance, kMinimumMapDistance, kMaximumMapDistance);
  self.visibleDistance = distance;
  [self setRegion:MKCoordinateRegionMakeWithDistance(self.fixedCentre, distance, distance)
         animated:animated];
}

- (void)scrollWheel:(NSEvent *)event {
  const double coefficient = event.hasPreciseScrollingDeltas ? 0.012 : 0.08;
  self.visibleDistance *= std::exp(-event.scrollingDeltaY * coefficient);
  [self applyVisibleDistanceAnimated:NO];
}

- (void)magnifyWithEvent:(NSEvent *)event {
  self.visibleDistance *= std::exp(-event.magnification);
  [self applyVisibleDistanceAnimated:NO];
}

@end

@interface MiniMapPanelView () <MKMapViewDelegate> {
@private
  NSView *_mapSection;
  NSView *_pointInfoView;
  FixedCentreMapView *_mapView;
  CompactMapScaleView *_scaleView;
  NSPopUpButton *_mapStyleControl;
  MiniMapAnnotation *_observerAnnotation;
  MiniMapAnnotation *_inspectionAnnotation;
  id<MKOverlay> _fieldOfViewOverlay;
  id<MKOverlay> _headingOverlay;
  double _observerEasting;
  double _observerNorthing;
  double _maxDistance;
  uint32_t _terrainEpsgCode;
  bool _mapVisible;
  bool _pointInfoVisible;
}
@end

@implementation MiniMapPanelView

- (instancetype)initWithObserverEasting:(double)easting
                               northing:(double)northing
                        terrainEpsgCode:(uint32_t)epsgCode
                            maxDistance:(double)maxDistance
                          pointInfoView:(NSView *)pointInfoView {
  self = [super initWithFrame:NSMakeRect(0.0, 0.0, kMapPanelWidth, kMapSectionHeight)];
  if (self == nil) {
    return nil;
  }

  _observerEasting = easting;
  _observerNorthing = northing;
  _terrainEpsgCode = epsgCode;
  _maxDistance = maxDistance;
  _pointInfoView = pointInfoView;

  _mapSection = [[NSView alloc] initWithFrame:NSZeroRect];
  [self addSubview:_mapSection];
  [self addSubview:_pointInfoView];

  NSTextField *heading = [NSTextField labelWithString:@"Minimap"];
  heading.font = [NSFont boldSystemFontOfSize:NSFont.systemFontSize];
  _mapStyleControl = [[NSPopUpButton alloc] initWithFrame:NSZeroRect pullsDown:NO];
  [_mapStyleControl addItemsWithTitles:@[
    @"Standard Terrain",
    @"Hybrid",
    @"Imagery",
  ]];
  _mapStyleControl.target = self;
  _mapStyleControl.action = @selector(mapStyleChanged:);

  NSStackView *controls = [NSStackView stackViewWithViews:@[
    heading,
    _mapStyleControl,
  ]];
  controls.orientation = NSUserInterfaceLayoutOrientationHorizontal;
  controls.alignment = NSLayoutAttributeCenterY;
  controls.spacing = 6.0;
  [heading setContentHuggingPriority:NSLayoutPriorityDefaultHigh
                      forOrientation:NSLayoutConstraintOrientationHorizontal];
  [_mapStyleControl setContentHuggingPriority:NSLayoutPriorityDefaultLow
                               forOrientation:NSLayoutConstraintOrientationHorizontal];
  controls.translatesAutoresizingMaskIntoConstraints = NO;
  [_mapSection addSubview:controls];

  _mapView = [[FixedCentreMapView alloc] initWithFrame:NSZeroRect];
  _mapView.delegate = self;
  _mapView.scrollEnabled = NO;
  _mapView.zoomEnabled = NO;
  _mapView.rotateEnabled = NO;
  _mapView.pitchEnabled = NO;
  _mapView.showsCompass = NO;
  _mapView.showsScale = NO;
  _mapView.wantsLayer = YES;
  _mapView.layer.cornerRadius = 10.0;
  _mapView.layer.masksToBounds = YES;
  _mapView.translatesAutoresizingMaskIntoConstraints = NO;
  [_mapSection addSubview:_mapView];

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
    [_mapView.bottomAnchor constraintEqualToAnchor:_mapSection.bottomAnchor constant:-12.0],
  ]];

  // Terrain and ray geometry use the dataset's projected CRS (for example,
  // EPSG:2056 Swiss LV95), whereas MapKit accepts WGS84 latitude/longitude.
  // The observer is therefore transformed once before it becomes the fixed
  // map centre. Camera headings must *not* simply be treated as geographic
  // bearings: setCameraOrientation constructs endpoints in the terrain CRS
  // first and transforms those complete points to WGS84, preserving the
  // projected grid's local convergence relative to true north.
  const panorama::Crs terrainCrs = panorama::Crs::from_epsg(_terrainEpsgCode);
  const panorama::LatLon observerLatLon =
      terrainCrs.to_lat_lon({_observerEasting, _observerNorthing});
  const CLLocationCoordinate2D observerCoordinate =
      CLLocationCoordinate2DMake(observerLatLon.lat, observerLatLon.lon);
  _mapView.fixedCentre = observerCoordinate;
  _mapView.visibleDistance = kInitialMapDistance;
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
  [self mapStyleChanged:_mapStyleControl];

  _mapVisible = false;
  _pointInfoVisible = false;
  _mapSection.hidden = YES;
  _pointInfoView.hidden = YES;
  return self;
}

- (void)layout {
  [super layout];
  const CGFloat pointHeight = _pointInfoVisible ? kPointSectionHeight : 0.0;
  _pointInfoView.frame = NSMakeRect(0.0, 0.0, self.bounds.size.width, pointHeight);
  _mapSection.frame = NSMakeRect(
      0.0,
      pointHeight,
      self.bounds.size.width,
      std::max(0.0, self.bounds.size.height - pointHeight)
  );
  [_scaleView updateForMapView:_mapView];
}

- (void)setMapVisible:(bool)visible {
  _mapVisible = visible;
  _mapSection.hidden = !visible;
  [self setNeedsLayout:YES];
}

- (void)setPointInfoVisible:(bool)visible {
  _pointInfoVisible = visible;
  _pointInfoView.hidden = !visible;
  [self setNeedsLayout:YES];
}

- (bool)hasVisibleContent {
  return _mapVisible || _pointInfoVisible;
}

- (NSSize)preferredPanelSize {
  const CGFloat width = _mapVisible ? kMapPanelWidth : 230.0;
  const CGFloat height =
      (_mapVisible ? kMapSectionHeight : 0.0) + (_pointInfoVisible ? kPointSectionHeight : 0.0);
  return NSMakeSize(width, height);
}

- (void)mapStyleChanged:(id)sender {
  (void)sender;
  switch (_mapStyleControl.indexOfSelectedItem) {
  case 0:
    _mapView.preferredConfiguration =
        [[MKStandardMapConfiguration alloc] initWithElevationStyle:MKMapElevationStyleRealistic];
    break;
  case 1:
    _mapView.preferredConfiguration =
        [[MKHybridMapConfiguration alloc] initWithElevationStyle:MKMapElevationStyleRealistic];
    break;
  default:
    _mapView.preferredConfiguration =
        [[MKImageryMapConfiguration alloc] initWithElevationStyle:MKMapElevationStyleRealistic];
    break;
  }
}

- (void)mapViewDidChangeVisibleRegion:(MKMapView *)mapView {
  [_scaleView updateForMapView:mapView];
}

- (void)setCameraOrientation:(panorama::CameraOrientation)orientation
         verticalFieldOfView:(double)verticalFieldOfView
                       image:(panorama::ImageSize)image {
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

  if (_fieldOfViewOverlay != nil) {
    [_mapView removeOverlay:_fieldOfViewOverlay];
  }
  if (_headingOverlay != nil) {
    [_mapView removeOverlay:_headingOverlay];
  }
  _fieldOfViewOverlay = [MKPolygon polygonWithCoordinates:wedge count:3];
  _headingOverlay = [MKPolyline polylineWithCoordinates:headingLine count:2];
  [_mapView addOverlay:_fieldOfViewOverlay level:MKOverlayLevelAboveRoads];
  [_mapView addOverlay:_headingOverlay level:MKOverlayLevelAboveRoads];
}

- (void)setInspectedPointEasting:(double)easting northing:(double)northing locked:(bool)locked {
  const panorama::Crs terrainCrs = panorama::Crs::from_epsg(_terrainEpsgCode);
  const panorama::LatLon geographic = terrainCrs.to_lat_lon({easting, northing});
  if (_inspectionAnnotation == nil) {
    _inspectionAnnotation = [[MiniMapAnnotation alloc] init];
    _inspectionAnnotation.kind = locked ? AnnotationKind::Locked : AnnotationKind::Hover;
    _inspectionAnnotation.coordinate = CLLocationCoordinate2DMake(geographic.lat, geographic.lon);
    [_mapView addAnnotation:_inspectionAnnotation];
  } else {
    _inspectionAnnotation.kind = locked ? AnnotationKind::Locked : AnnotationKind::Hover;
    _inspectionAnnotation.coordinate = CLLocationCoordinate2DMake(geographic.lat, geographic.lon);
  }
  MKAnnotationView *view = [_mapView viewForAnnotation:_inspectionAnnotation];
  if (view != nil) {
    view.image = annotation_image(_inspectionAnnotation.kind);
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
  return view;
}

- (MKOverlayRenderer *)mapView:(MKMapView *)mapView rendererForOverlay:(id<MKOverlay>)overlay {
  (void)mapView;
  if ([overlay isKindOfClass:MKPolygon.class]) {
    MKPolygonRenderer *renderer = [[MKPolygonRenderer alloc] initWithPolygon:(MKPolygon *)overlay];
    renderer.fillColor = [NSColor.systemBlueColor colorWithAlphaComponent:0.14];
    renderer.strokeColor = NSColor.clearColor;
    renderer.lineWidth = 0.0;
    return renderer;
  }
  MKPolylineRenderer *renderer =
      [[MKPolylineRenderer alloc] initWithPolyline:(MKPolyline *)overlay];
  renderer.strokeColor = NSColor.systemBlueColor;
  renderer.lineWidth = 2.0;
  return renderer;
}

@end

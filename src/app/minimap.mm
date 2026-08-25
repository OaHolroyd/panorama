#include "minimap.h"

#include "crs.h"

#import <MapKit/MapKit.h>
#import <QuartzCore/QuartzCore.h>

#include <algorithm>
#include <cmath>

constexpr CGFloat kMapPanelWidth = 300.0;
constexpr CGFloat kMapSectionHeight = 286.0;
constexpr CGFloat kPointSectionHeight = 174.0;
constexpr double kInitialMapDistance = 50'000.0;
constexpr double kMinimumMapDistance = 500.0;
constexpr double kMaximumMapDistance = 2'000'000.0;

enum class AnnotationKind : NSInteger {
  Observer,
  Hover,
  Locked,
};

@interface MiniMapAnnotation : NSObject <MKAnnotation>
@property(nonatomic) CLLocationCoordinate2D coordinate;
@property(nonatomic) AnnotationKind kind;
@end

@implementation MiniMapAnnotation
@end

/// A north-up map whose centre is always the observer. MapKit's normal pan,
/// pitch, and rotation gestures are disabled; scrolling and pinching change
/// only the stored ground span, so navigation cannot lose the viewpoint.
@interface FixedCentreMapView : MKMapView
@property(nonatomic) CLLocationCoordinate2D fixedCentre;
@property(nonatomic) double visibleDistance;
- (void)zoomByFactor:(double)factor;
@end

@implementation FixedCentreMapView

- (void)applyVisibleDistanceAnimated:(BOOL)animated {
  const double distance =
      std::clamp(self.visibleDistance, kMinimumMapDistance, kMaximumMapDistance);
  self.visibleDistance = distance;
  [self setRegion:MKCoordinateRegionMakeWithDistance(self.fixedCentre, distance, distance)
         animated:animated];
}

- (void)zoomByFactor:(double)factor {
  self.visibleDistance *= factor;
  [self applyVisibleDistanceAnimated:YES];
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

  NSButton *zoomOut = [NSButton buttonWithImage:[NSImage imageWithSystemSymbolName:@"minus"
                                                          accessibilityDescription:@"Zoom Map Out"]
                                         target:self
                                         action:@selector(zoomMapOut:)];
  NSButton *zoomIn = [NSButton buttonWithImage:[NSImage imageWithSystemSymbolName:@"plus"
                                                         accessibilityDescription:@"Zoom Map In"]
                                        target:self
                                        action:@selector(zoomMapIn:)];
  zoomOut.bezelStyle = NSBezelStyleAccessoryBarAction;
  zoomIn.bezelStyle = NSBezelStyleAccessoryBarAction;
  NSStackView *controls = [NSStackView stackViewWithViews:@[
    heading,
    _mapStyleControl,
    zoomOut,
    zoomIn,
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
  _mapView.showsScale = YES;
  _mapView.wantsLayer = YES;
  _mapView.layer.cornerRadius = 10.0;
  _mapView.layer.masksToBounds = YES;
  _mapView.translatesAutoresizingMaskIntoConstraints = NO;
  [_mapSection addSubview:_mapView];

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

- (void)zoomMapOut:(id)sender {
  (void)sender;
  [_mapView zoomByFactor:2.0];
}

- (void)zoomMapIn:(id)sender {
  (void)sender;
  [_mapView zoomByFactor:0.5];
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
  if ([view isKindOfClass:MKMarkerAnnotationView.class]) {
    MKMarkerAnnotationView *marker = (MKMarkerAnnotationView *)view;
    marker.markerTintColor = locked ? NSColor.systemOrangeColor : NSColor.systemBlueColor;
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
  MKMarkerAnnotationView *view = [[MKMarkerAnnotationView alloc] initWithAnnotation:annotation
                                                                    reuseIdentifier:nil];
  view.canShowCallout = NO;
  if (marker.kind == AnnotationKind::Observer) {
    view.markerTintColor = NSColor.systemPurpleColor;
    view.glyphImage = [NSImage imageWithSystemSymbolName:@"location.fill"
                                accessibilityDescription:@"Observer"];
  } else {
    view.markerTintColor =
        marker.kind == AnnotationKind::Locked ? NSColor.systemOrangeColor : NSColor.systemBlueColor;
    view.glyphImage = [NSImage imageWithSystemSymbolName:@"scope"
                                accessibilityDescription:@"Inspected Terrain Point"];
  }
  return view;
}

- (MKOverlayRenderer *)mapView:(MKMapView *)mapView rendererForOverlay:(id<MKOverlay>)overlay {
  (void)mapView;
  if ([overlay isKindOfClass:MKPolygon.class]) {
    MKPolygonRenderer *renderer = [[MKPolygonRenderer alloc] initWithPolygon:(MKPolygon *)overlay];
    renderer.fillColor = [NSColor.systemBlueColor colorWithAlphaComponent:0.14];
    renderer.strokeColor = [NSColor.systemBlueColor colorWithAlphaComponent:0.75];
    renderer.lineWidth = 1.5;
    return renderer;
  }
  MKPolylineRenderer *renderer =
      [[MKPolylineRenderer alloc] initWithPolyline:(MKPolyline *)overlay];
  renderer.strokeColor = NSColor.systemBlueColor;
  renderer.lineWidth = 2.0;
  return renderer;
}

@end

#include "minimap.h"

#include "crs.h"

#import <MapKit/MapKit.h>
#import <MetalKit/MetalKit.h>
#import <QuartzCore/QuartzCore.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

constexpr CGFloat kCompactMapPanelWidth = 300.0;
constexpr CGFloat kCompactMapSectionHeight = 286.0;
constexpr CGFloat kLargeMapPanelWidth = 520.0;
constexpr CGFloat kLargeMapSectionHeight = 456.0;
constexpr CGFloat kPointSectionHeight = 36.0;
constexpr double kInitialMapDistance = 50'000.0;
constexpr double kVisibilityBasisDistance = 1'000.0;

/// Scalar-only ABI mirrored by `VisibilityMapParameters` in image_renderer.metal.
struct VisibilityMapParameters {
  float centre_x;
  float centre_y;
  float east_x_per_metre;
  float east_y_per_metre;
  float north_x_per_metre;
  float north_y_per_metre;
  float point_size;
  uint32_t ray_count;
};

static_assert(sizeof(VisibilityMapParameters) == 8U * sizeof(uint32_t));

enum class AnnotationKind : NSInteger {
  Observer,
  Hover,
  Locked,
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

/// Transparent, non-interactive Metal layer drawn over MapKit. It consumes the
/// immutable point buffer published with a completed frame, so hundreds of
/// thousands of collisions remain one GPU draw rather than becoming MapKit
/// annotations or host-side paths.
@interface VisibilityMapView : MTKView <MTKViewDelegate> {
@private
  id<MTLCommandQueue> _visibilityCommandQueue;
  id<MTLRenderPipelineState> _visibilityPipeline;
  id<MTLBuffer> _points;
  VisibilityMapParameters _mapParameters;
}
- (instancetype)initWithDevice:(id<MTLDevice>)device
                  commandQueue:(id<MTLCommandQueue>)commandQueue
                       library:(id<MTLLibrary>)library;
- (void)setPoints:(id<MTLBuffer>)points image:(panorama::ImageSize)image;
- (void)setMapParameters:(VisibilityMapParameters)parameters;
@end

@implementation VisibilityMapView

- (instancetype)initWithDevice:(id<MTLDevice>)device
                  commandQueue:(id<MTLCommandQueue>)commandQueue
                       library:(id<MTLLibrary>)library {
  self = [super initWithFrame:NSZeroRect device:device];
  if (self == nil) {
    return nil;
  }
  if (device == nil || commandQueue == nil || library == nil) {
    throw std::invalid_argument("Visibility map requires valid Metal resources");
  }

  id<MTLFunction> vertex = [library newFunctionWithName:@"visibility_point_vertex"];
  id<MTLFunction> fragment = [library newFunctionWithName:@"visibility_point_fragment"];
  if (vertex == nil || fragment == nil) {
    throw std::runtime_error("Visibility point shaders are missing from the Metal library");
  }

  self.colorPixelFormat = MTLPixelFormatBGRA8Unorm;
  MTLRenderPipelineDescriptor *descriptor = [[MTLRenderPipelineDescriptor alloc] init];
  descriptor.label = @"Minimap visibility points";
  descriptor.vertexFunction = vertex;
  descriptor.fragmentFunction = fragment;
  descriptor.colorAttachments[0].pixelFormat = self.colorPixelFormat;
  descriptor.colorAttachments[0].blendingEnabled = YES;
  // Maximum blending preserves a constant highlight opacity where many camera
  // rays land on the same minimap pixel.
  descriptor.colorAttachments[0].rgbBlendOperation = MTLBlendOperationMax;
  descriptor.colorAttachments[0].alphaBlendOperation = MTLBlendOperationMax;
  descriptor.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorOne;
  descriptor.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOne;
  descriptor.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
  descriptor.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOne;

  NSError *error = nil;
  _visibilityPipeline = [device newRenderPipelineStateWithDescriptor:descriptor error:&error];
  if (_visibilityPipeline == nil) {
    const char *detail = error == nil ? "unknown error" : error.localizedDescription.UTF8String;
    throw std::runtime_error(
        "Could not create minimap visibility pipeline: " + std::string(detail)
    );
  }

  _visibilityCommandQueue = commandQueue;
  _mapParameters.point_size = 2.0F;
  self.delegate = self;
  self.paused = YES;
  self.enableSetNeedsDisplay = YES;
  self.autoResizeDrawable = YES;
  self.framebufferOnly = YES;
  self.clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 0.0);
  self.wantsLayer = YES;
  self.layer.opaque = NO;
  self.layer.backgroundColor = NSColor.clearColor.CGColor;
  return self;
}

- (BOOL)isOpaque {
  return NO;
}

- (NSView *)hitTest:(NSPoint)point {
  (void)point;
  return nil;
}

- (void)setPoints:(id<MTLBuffer>)points image:(panorama::ImageSize)image {
  const uint64_t count = static_cast<uint64_t>(image.width) * image.height;
  const uint64_t pointBytes = count * 2U * sizeof(float);
  if (count == 0U || count > UINT32_MAX || points == nil || points.length < pointBytes) {
    _points = nil;
    _mapParameters.ray_count = 0U;
  } else {
    _points = points;
    _mapParameters.ray_count = static_cast<uint32_t>(count);
  }
  [self setNeedsDisplay:YES];
}

- (void)setMapParameters:(VisibilityMapParameters)parameters {
  const uint32_t rayCount = _mapParameters.ray_count;
  _mapParameters = parameters;
  _mapParameters.ray_count = rayCount;
  [self setNeedsDisplay:YES];
}

- (void)drawInMTKView:(MTKView *)view {
  id<CAMetalDrawable> drawable = view.currentDrawable;
  MTLRenderPassDescriptor *pass = view.currentRenderPassDescriptor;
  if (drawable == nil || pass == nil) {
    return;
  }
  pass.colorAttachments[0].loadAction = MTLLoadActionClear;
  pass.colorAttachments[0].storeAction = MTLStoreActionStore;
  pass.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 0.0);

  id<MTLCommandBuffer> command = [_visibilityCommandQueue commandBuffer];
  id<MTLRenderCommandEncoder> encoder = [command renderCommandEncoderWithDescriptor:pass];
  if (command == nil || encoder == nil) {
    return;
  }
  command.label = @"Draw minimap visibility";
  encoder.label = @"visibility points";
  if (_points != nil && _mapParameters.ray_count > 0U) {
    [encoder setRenderPipelineState:_visibilityPipeline];
    [encoder setVertexBuffer:_points offset:0 atIndex:0];
    [encoder setVertexBytes:&_mapParameters length:sizeof(_mapParameters) atIndex:1];
    [encoder drawPrimitives:MTLPrimitiveTypePoint
                vertexStart:0U
                vertexCount:_mapParameters.ray_count];
  }
  [encoder endEncoding];
  [command presentDrawable:drawable];
  [command commit];
}

- (void)mtkView:(MTKView *)view drawableSizeWillChange:(CGSize)size {
  (void)view;
  (void)size;
}

@end

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
  VisibilityMapView *_visibilityView;
  CompactMapScaleView *_scaleView;
  NSPopUpButton *_mapStyleControl;
  NSButton *_coverageControl;
  NSButton *_returnToObserverControl;
  NSButton *_mapFocusControl;
  NSButton *_mapSizeControl;
  MiniMapAnnotation *_observerAnnotation;
  MiniMapAnnotation *_inspectionAnnotation;
  CLLocationCoordinate2D _eastBasisCoordinate;
  CLLocationCoordinate2D _northBasisCoordinate;
  id<MKOverlay> _fieldOfViewOverlay;
  id<MKOverlay> _headingOverlay;
  MKMultiPolygon *_coverageOverlay;
  double _observerEasting;
  double _observerNorthing;
  double _maxDistance;
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
    heading,
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
  [heading setContentHuggingPriority:NSLayoutPriorityDefaultHigh
                      forOrientation:NSLayoutConstraintOrientationHorizontal];
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

  _visibilityView = [[VisibilityMapView alloc] initWithDevice:metalDevice
                                                 commandQueue:commandQueue
                                                      library:library];
  _visibilityView.frame = _mapView.bounds;
  _visibilityView.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
  [_mapView addSubview:_visibilityView];

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
  const panorama::LatLon eastBasis = terrainCrs.to_lat_lon(
      {
          _observerEasting + kVisibilityBasisDistance,
          _observerNorthing,
      }
  );
  const panorama::LatLon northBasis = terrainCrs.to_lat_lon(
      {
          _observerEasting,
          _observerNorthing + kVisibilityBasisDistance,
      }
  );
  _eastBasisCoordinate = CLLocationCoordinate2DMake(eastBasis.lat, eastBasis.lon);
  _northBasisCoordinate = CLLocationCoordinate2DMake(northBasis.lat, northBasis.lon);
  [self mapStyleChanged:_mapStyleControl];

  _contentVisible = false;
  _coverageVisible = coverageVisible;
  _followInspection = false;
  _mapPointerInside = false;
  _largeMap = false;
  if (_coverageVisible) {
    [_mapView addOverlay:_coverageOverlay level:MKOverlayLevelAboveRoads];
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
    [_mapView addOverlay:_coverageOverlay level:MKOverlayLevelAboveRoads];
  } else {
    [_mapView removeOverlay:_coverageOverlay];
  }
  [self updateCoverageControl];
}

- (void)returnToObserver:(id)sender {
  (void)sender;
  [_mapView setCenterCoordinate:_observerAnnotation.coordinate animated:YES];
}

- (void)layout {
  [super layout];
  const CGFloat pointHeight = _contentVisible ? kPointSectionHeight : 0.0;
  _pointInfoView.frame = NSMakeRect(0.0, 0.0, self.bounds.size.width, pointHeight);
  _mapSection.frame = NSMakeRect(
      0.0,
      pointHeight,
      self.bounds.size.width,
      std::max(0.0, self.bounds.size.height - pointHeight)
  );
  [_scaleView updateForMapView:_mapView];
  [self updateVisibilityTransform];
}

- (void)setMapAndPointInfoVisible:(bool)visible {
  _contentVisible = visible;
  _mapSection.hidden = !visible;
  _pointInfoView.hidden = !visible;
  if (visible) {
    [_visibilityView setNeedsDisplay:YES];
  }
  [self setNeedsLayout:YES];
}

- (NSSize)preferredPanelSize {
  const CGFloat mapWidth = _largeMap ? kLargeMapPanelWidth : kCompactMapPanelWidth;
  const CGFloat mapHeight = _largeMap ? kLargeMapSectionHeight : kCompactMapSectionHeight;
  return NSMakeSize(mapWidth, _contentVisible ? mapHeight + kPointSectionHeight : 0.0);
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
  [self updateVisibilityTransform];
}

/// Approximate the projected terrain CRS over this compact map with the local
/// east/north Jacobian at the observer. Deriving both basis endpoints through
/// the full CRS and MapKit conversion keeps grid convergence aligned with the
/// existing FOV wedge while the map zoom changes.
- (void)updateVisibilityTransform {
  const NSSize size = _visibilityView.bounds.size;
  if (size.width <= 0.0 || size.height <= 0.0) {
    return;
  }
  const NSPoint centre = [_mapView convertCoordinate:_observerAnnotation.coordinate
                                       toPointToView:_visibilityView];
  const NSPoint east = [_mapView convertCoordinate:_eastBasisCoordinate
                                     toPointToView:_visibilityView];
  const NSPoint north = [_mapView convertCoordinate:_northBasisCoordinate
                                      toPointToView:_visibilityView];
  const auto clip = [size](NSPoint point) {
    return NSMakePoint(2.0 * point.x / size.width - 1.0, 2.0 * point.y / size.height - 1.0);
  };
  const NSPoint centreClip = clip(centre);
  const NSPoint eastClip = clip(east);
  const NSPoint northClip = clip(north);
  const VisibilityMapParameters parameters = {
      static_cast<float>(centreClip.x),
      static_cast<float>(centreClip.y),
      static_cast<float>((eastClip.x - centreClip.x) / kVisibilityBasisDistance),
      static_cast<float>((eastClip.y - centreClip.y) / kVisibilityBasisDistance),
      static_cast<float>((northClip.x - centreClip.x) / kVisibilityBasisDistance),
      static_cast<float>((northClip.y - centreClip.y) / kVisibilityBasisDistance),
      2.0F,
      0U,
  };
  [_visibilityView setMapParameters:parameters];
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

- (void)setVisibilityPoints:(id<MTLBuffer>)points image:(panorama::ImageSize)image {
  [_visibilityView setPoints:points image:image];
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
  _observerEasting = easting;
  _observerNorthing = northing;
  const panorama::Crs terrainCrs = panorama::Crs::from_epsg(_terrainEpsgCode);
  const panorama::LatLon observer = terrainCrs.to_lat_lon({easting, northing});
  const CLLocationCoordinate2D coordinate = CLLocationCoordinate2DMake(observer.lat, observer.lon);
  _observerAnnotation.coordinate = coordinate;

  const panorama::LatLon east =
      terrainCrs.to_lat_lon({easting + kVisibilityBasisDistance, northing});
  const panorama::LatLon north =
      terrainCrs.to_lat_lon({easting, northing + kVisibilityBasisDistance});
  _eastBasisCoordinate = CLLocationCoordinate2DMake(east.lat, east.lon);
  _northBasisCoordinate = CLLocationCoordinate2DMake(north.lat, north.lon);
  [self updateVisibilityTransform];
}

- (void)setInspectedPointEasting:(double)easting northing:(double)northing locked:(bool)locked {
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
  MKPolylineRenderer *renderer =
      [[MKPolylineRenderer alloc] initWithPolyline:(MKPolyline *)overlay];
  renderer.strokeColor = NSColor.systemBlueColor;
  renderer.lineWidth = 2.0;
  return renderer;
}

@end

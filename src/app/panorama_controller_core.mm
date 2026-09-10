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

NSString *format_movement_speed(double metres_per_second) {
  const double kilometres_per_hour = metres_per_second * 3.6;
  return std::abs(kilometres_per_hour) < 100.0
             ? [NSString stringWithFormat:@"%.1f km/h", kilometres_per_hour]
             : [NSString stringWithFormat:@"%.0f km/h", kilometres_per_hour];
}

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

@end

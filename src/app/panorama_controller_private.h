#pragma once

#include "panorama_controller.h"

#include "coordinate_input.h"
#include "metalfx_upscaler.h"

#import <MapKit/MapKit.h>

#include <chrono>
#include <optional>

[[nodiscard]] NSString *format_movement_speed(double metres_per_second);

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
  NSPopUpButton *_peakLabelControl;
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
  panorama::app::PeakLabelMode _peakLabelMode;
  NSTimer *_metalfxSettleTimer;
}
@end

@interface PanoramaController (Private)
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
- (void)clearTargetVisibility;
- (void)requestTargetVisibilityForPoint:(panorama::app::TerrainPoint)point;
- (void)lookAtTerrainPoint:(panorama::app::TerrainPoint)point;
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
- (void)updateSettingsControlAvailability;
- (void)setVerticalFieldOfViewDegrees:(double)degrees;
- (void)updateZoomControls;
- (void)updateAspectLockAppearance;
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
- (void)astronomicalInputChanged:(NSTextField *)sender;
- (BOOL)publishAstronomicalLighting;
- (BOOL)publishTerrainControls;
- (BOOL)commitResolutionControls;
- (void)metalfxChanged:(id)sender;
- (void)peakLabelsChanged:(id)sender;
- (void)requestMetalFxInteraction;
- (void)settleMetalFx:(NSTimer *)timer;
- (NSViewController *)makeSettingsViewController;
- (NSViewController *)makePositioningViewController;
- (NSViewController *)makeDebugViewController;
- (NSViewController *)makePointInfoViewController;
- (void)updatePointInfo:(std::optional<panorama::app::PointInspection>)inspection;
- (void)updateLockedPointIndicatorWithOrientation:(panorama::CameraOrientation)orientation
                              verticalFieldOfView:(double)verticalFieldOfView
                                            image:(panorama::ImageSize)image;
- (void)updateDebugInfoWithOrientation:(panorama::CameraOrientation)orientation
                   verticalFieldOfView:(double)verticalFieldOfView
                                 image:(panorama::ImageSize)image
                          milliseconds:(double)milliseconds
                       gpuMilliseconds:(double)gpuMilliseconds
                              streamed:(BOOL)streamed
                              revision:(uint64_t)revision;
@end

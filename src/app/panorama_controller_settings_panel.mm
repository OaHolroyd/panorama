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

@implementation PanoramaController (SettingsPanel)

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

  _peakLabelControl = [[NSPopUpButton alloc] initWithFrame:NSZeroRect pullsDown:NO];
  [_peakLabelControl addItemsWithTitles:@[ @"Off", @"On", @"Near pointer" ]];
  [_peakLabelControl selectItemAtIndex:static_cast<NSInteger>(_peakLabelMode)];
  _peakLabelControl.target = self;
  _peakLabelControl.action = @selector(peakLabelsChanged:);
  _peakLabelControl.toolTip = @"Show names and elevations for visible Alpine peaks";
  _peakLabelControl.enabled = _renderer->peak_labels_available();

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
                                           make_row(@"Peak labels", _peakLabelControl),
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

@end

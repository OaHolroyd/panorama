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

@implementation PanoramaController (PositioningPanel)

- (NSViewController *)makePositioningViewController {
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

  NSTextField *heading = [NSTextField labelWithString:@"Position & Movement"];
  heading.font = [NSFont boldSystemFontOfSize:NSFont.systemFontSize];

  _coordinateSystemControl = [[NSPopUpButton alloc] initWithFrame:NSZeroRect pullsDown:NO];
  [_coordinateSystemControl addItemWithTitle:@"Auto"];
  _coordinateSystemControl.lastItem.tag = -1;
  [_coordinateSystemControl.menu addItem:NSMenuItem.separatorItem];
  [_coordinateSystemControl addItemWithTitle:@"Latitude / longitude"];
  _coordinateSystemControl.lastItem.tag =
      static_cast<NSInteger>(panorama::app::CoordinateInputSystem::Wgs84);
  [_coordinateSystemControl addItemWithTitle:@"Swiss LV95"];
  _coordinateSystemControl.lastItem.tag =
      static_cast<NSInteger>(panorama::app::CoordinateInputSystem::SwissLv95);
  [_coordinateSystemControl addItemWithTitle:@"OS National Grid"];
  _coordinateSystemControl.lastItem.tag =
      static_cast<NSInteger>(panorama::app::CoordinateInputSystem::BritishNationalGrid);
  [_coordinateSystemControl.menu addItem:NSMenuItem.separatorItem];
  [_coordinateSystemControl
      addItemWithTitle:[NSString
                           stringWithFormat:@"Dataset grid — %s", _renderer->terrain_crs().name()]];
  _coordinateSystemControl.lastItem.tag =
      static_cast<NSInteger>(panorama::app::CoordinateInputSystem::Terrain);
  _coordinateSystemControl.target = self;
  _coordinateSystemControl.action = @selector(coordinateSystemChanged:);
  _coordinateSystemControl.toolTip =
      @"Auto detects the coordinate system; choose one explicitly to resolve ambiguity";
  // Cap the row at the inspector's 268-point content width. Pop-up buttons use
  // their longest menu item as an intrinsic width; without this constraint the
  // dataset-grid title can force the whole inset stack beyond the panel edge.
  [_coordinateSystemControl.widthAnchor constraintEqualToConstant:178.0].active = YES;
  NSTextField *coordinateSystemLabel = [NSTextField labelWithString:@"System"];
  [coordinateSystemLabel.widthAnchor constraintEqualToConstant:82.0].active = YES;
  NSStackView *coordinateSystemRow =
      [NSStackView stackViewWithViews:@[ coordinateSystemLabel, _coordinateSystemControl ]];
  coordinateSystemRow.orientation = NSUserInterfaceLayoutOrientationHorizontal;
  coordinateSystemRow.alignment = NSLayoutAttributeCenterY;
  coordinateSystemRow.spacing = 8.0;

  _coordinateInputControl = [[NSTextField alloc] initWithFrame:NSZeroRect];
  _coordinateInputControl.delegate = self;
  _coordinateInputControl.placeholderString = @"Enter or paste a coordinate";
  _coordinateInputControl.toolTip = @"The coordinate system will be detected automatically";
  _coordinateInputControl.target = self;
  _coordinateInputControl.action = @selector(moveToCoordinate:);
  [_coordinateInputControl.widthAnchor constraintEqualToConstant:178.0].active = YES;
  NSTextField *coordinateLabel = [NSTextField labelWithString:@"Coordinate"];
  [coordinateLabel.widthAnchor constraintEqualToConstant:82.0].active = YES;
  NSStackView *coordinateRow =
      [NSStackView stackViewWithViews:@[ coordinateLabel, _coordinateInputControl ]];
  coordinateRow.orientation = NSUserInterfaceLayoutOrientationHorizontal;
  coordinateRow.alignment = NSLayoutAttributeCenterY;
  coordinateRow.spacing = 8.0;

  _coordinateStatusLabel = [NSTextField labelWithString:@"Format will be detected automatically"];
  _coordinateStatusLabel.font = [NSFont systemFontOfSize:NSFont.smallSystemFontSize];
  _coordinateStatusLabel.textColor = NSColor.secondaryLabelColor;
  _coordinateStatusLabel.maximumNumberOfLines = 2;
  _coordinateStatusLabel.lineBreakMode = NSLineBreakByWordWrapping;
  [_coordinateStatusLabel.widthAnchor constraintEqualToConstant:268.0].active = YES;

  _coordinateMoveControl = [NSButton buttonWithTitle:@"Move"
                                              target:self
                                              action:@selector(moveToCoordinate:)];
  _coordinateMoveControl.image = [NSImage imageWithSystemSymbolName:@"location.fill"
                                           accessibilityDescription:@"Move observer to coordinate"];
  _coordinateMoveControl.imagePosition = NSImageLeading;
  _coordinateMoveControl.enabled = NO;
  NSView *coordinateSpacer = [[NSView alloc] initWithFrame:NSZeroRect];
  [coordinateSpacer setContentHuggingPriority:NSLayoutPriorityDefaultLow
                               forOrientation:NSLayoutConstraintOrientationHorizontal];
  [coordinateSpacer
      setContentCompressionResistancePriority:NSLayoutPriorityDefaultLow
                               forOrientation:NSLayoutConstraintOrientationHorizontal];
  NSStackView *coordinateActionRow =
      [NSStackView stackViewWithViews:@[ coordinateSpacer, _coordinateMoveControl ]];
  coordinateActionRow.orientation = NSUserInterfaceLayoutOrientationHorizontal;
  coordinateActionRow.alignment = NSLayoutAttributeCenterY;
  [coordinateActionRow.widthAnchor constraintEqualToConstant:268.0].active = YES;

  InspectorSectionView *destinationSection =
      [[InspectorSectionView alloc] initWithTitle:@"Destination"
                                         controls:@[
                                           coordinateSystemRow,
                                           coordinateRow,
                                           _coordinateStatusLabel,
                                           coordinateActionRow,
                                         ]
                                      defaultsKey:@"panorama.inspector.destination.expanded"];

  const auto makeMovementRow = [](NSString *title, NSView *control) {
    NSTextField *label = [NSTextField labelWithString:title];
    [label.widthAnchor constraintEqualToConstant:82.0].active = YES;
    [control.widthAnchor constraintEqualToConstant:178.0].active = YES;
    NSStackView *row = [NSStackView stackViewWithViews:@[ label, control ]];
    row.orientation = NSUserInterfaceLayoutOrientationHorizontal;
    row.alignment = NSLayoutAttributeCenterY;
    row.spacing = 8.0;
    return row;
  };

  _movementModeControl = [[NSSegmentedControl alloc] initWithFrame:NSZeroRect];
  _movementModeControl.segmentCount = 3;
  [_movementModeControl setLabel:@"Browse" forSegment:0];
  [_movementModeControl setLabel:@"Roam" forSegment:1];
  [_movementModeControl setLabel:@"Cruise" forSegment:2];
  _movementModeControl.selectedSegment = 0;
  _movementModeControl.trackingMode = NSSegmentSwitchTrackingSelectOne;
  _movementModeControl.target = self;
  _movementModeControl.action = @selector(movementModeChanged:);
  _movementModeControl.toolTip = @"Browse looks around; Roam uses WASD; Cruise moves forward "
                                  "continuously under mouse control";
  NSView *movementModeRow = makeMovementRow(@"Mode", _movementModeControl);

  _roamTurningModeControl = [[NSSegmentedControl alloc] initWithFrame:NSZeroRect];
  _roamTurningModeControl.segmentCount = 2;
  [_roamTurningModeControl setLabel:@"Arrow keys" forSegment:0];
  [_roamTurningModeControl setLabel:@"Mouse" forSegment:1];
  _roamTurningModeControl.selectedSegment = 0;
  _roamTurningModeControl.trackingMode = NSSegmentSwitchTrackingSelectOne;
  _roamTurningModeControl.target = self;
  _roamTurningModeControl.action = @selector(roamTurningModeChanged:);
  _roamTurningModeControl.toolTip =
      @"Turn with the arrow keys or by moving the pointer over the panorama";
  _roamTurningModeRow = makeMovementRow(@"Turning", _roamTurningModeControl);

  _roamMouseSensitivityControl = [NSSlider sliderWithValue:1.0
                                                  minValue:0.25
                                                  maxValue:3.0
                                                    target:self
                                                    action:@selector(roamMouseSensitivityChanged:)];
  _roamMouseSensitivityControl.continuous = YES;
  _roamMouseSensitivityControl.toolTip = @"Mouse turning sensitivity";
  _roamMouseSensitivityLabel = [NSTextField labelWithString:@"1.00×"];
  _roamMouseSensitivityLabel.alignment = NSTextAlignmentRight;
  [_roamMouseSensitivityLabel.widthAnchor constraintEqualToConstant:54.0].active = YES;
  NSStackView *roamMouseSensitivitySetting = [NSStackView
      stackViewWithViews:@[ _roamMouseSensitivityControl, _roamMouseSensitivityLabel ]];
  roamMouseSensitivitySetting.orientation = NSUserInterfaceLayoutOrientationHorizontal;
  roamMouseSensitivitySetting.alignment = NSLayoutAttributeCenterY;
  roamMouseSensitivitySetting.spacing = 6.0;
  _roamMouseSensitivityRow = makeMovementRow(@"Sensitivity", roamMouseSensitivitySetting);

  _roamAltitudeModeControl = [[NSSegmentedControl alloc] initWithFrame:NSZeroRect];
  _roamAltitudeModeControl.segmentCount = 2;
  [_roamAltitudeModeControl setLabel:@"Terrain" forSegment:0];
  [_roamAltitudeModeControl setLabel:@"Altitude" forSegment:1];
  _roamAltitudeModeControl.selectedSegment = 0;
  _roamAltitudeModeControl.trackingMode = NSSegmentSwitchTrackingSelectOne;
  _roamAltitudeModeControl.target = self;
  _roamAltitudeModeControl.action = @selector(roamAltitudeModeChanged:);
  _roamAltitudeModeControl.toolTip =
      @"Maintain height above terrain or hold absolute elevation while moving";
  NSView *roamAltitudeRow = makeMovementRow(@"Height mode", _roamAltitudeModeControl);

  _aircraftDynamicsControl = [NSButton checkboxWithTitle:@"Enabled"
                                                  target:self
                                                  action:@selector(aircraftDynamicsChanged:)];
  _aircraftDynamicsControl.toolTip =
      @"Use coordinated banked turns and exchange airspeed with climbs and dives";
  _aircraftDynamicsRow = makeMovementRow(@"Aircraft", _aircraftDynamicsControl);

  _roamSpeedControl = [NSSlider sliderWithValue:panorama::app::kDefaultRoamSpeed
                                       minValue:panorama::app::kMinimumMovementSpeed
                                       maxValue:panorama::app::kMaximumRoamSpeed
                                         target:self
                                         action:@selector(roamSpeedChanged:)];
  _roamSpeedControl.continuous = YES;
  _roamSpeedControl.toolTip = @"Horizontal roaming speed";
  _roamSpeedLabel = [NSTextField labelWithString:@"72.0 km/h"];
  _roamSpeedLabel.alignment = NSTextAlignmentRight;
  [_roamSpeedLabel.widthAnchor constraintEqualToConstant:62.0].active = YES;
  NSStackView *roamSpeedSetting =
      [NSStackView stackViewWithViews:@[ _roamSpeedControl, _roamSpeedLabel ]];
  roamSpeedSetting.orientation = NSUserInterfaceLayoutOrientationHorizontal;
  roamSpeedSetting.alignment = NSLayoutAttributeCenterY;
  roamSpeedSetting.spacing = 6.0;
  _roamSpeedRowLabel = [NSTextField labelWithString:@"Speed"];
  [_roamSpeedRowLabel.widthAnchor constraintEqualToConstant:82.0].active = YES;
  [roamSpeedSetting.widthAnchor constraintEqualToConstant:178.0].active = YES;
  NSStackView *roamSpeedRow =
      [NSStackView stackViewWithViews:@[ _roamSpeedRowLabel, roamSpeedSetting ]];
  roamSpeedRow.orientation = NSUserInterfaceLayoutOrientationHorizontal;
  roamSpeedRow.alignment = NSLayoutAttributeCenterY;
  roamSpeedRow.spacing = 8.0;

  _roamUpdateRateControl = [NSSlider sliderWithValue:30.0
                                            minValue:1.0
                                            maxValue:60.0
                                              target:self
                                              action:@selector(roamUpdateRateChanged:)];
  _roamUpdateRateControl.continuous = YES;
  _roamUpdateRateControl.toolTip =
      @"Maximum observer-position requests per second; rendering may complete more slowly";
  _roamUpdateRateLabel = [NSTextField labelWithString:@"30 Hz"];
  _roamUpdateRateLabel.alignment = NSTextAlignmentRight;
  [_roamUpdateRateLabel.widthAnchor constraintEqualToConstant:54.0].active = YES;
  NSStackView *roamUpdateRateSetting =
      [NSStackView stackViewWithViews:@[ _roamUpdateRateControl, _roamUpdateRateLabel ]];
  roamUpdateRateSetting.orientation = NSUserInterfaceLayoutOrientationHorizontal;
  roamUpdateRateSetting.alignment = NSLayoutAttributeCenterY;
  roamUpdateRateSetting.spacing = 6.0;
  NSView *roamUpdateRateRow = makeMovementRow(@"Updates", roamUpdateRateSetting);

  _roamStatusLabel = [NSTextField labelWithString:@"WASD move • arrow keys look"];
  _roamStatusLabel.font = [NSFont systemFontOfSize:NSFont.smallSystemFontSize];
  _roamStatusLabel.textColor = NSColor.secondaryLabelColor;
  [_roamStatusLabel.widthAnchor constraintEqualToConstant:268.0].active = YES;
  _roamRows = @[
    _roamTurningModeRow,
    _roamMouseSensitivityRow,
    roamAltitudeRow,
    _aircraftDynamicsRow,
    roamSpeedRow,
    roamUpdateRateRow,
    _roamStatusLabel,
  ];

  InspectorSectionView *movementSection =
      [[InspectorSectionView alloc] initWithTitle:@"Movement"
                                         controls:@[
                                           movementModeRow,
                                           _roamTurningModeRow,
                                           _roamMouseSensitivityRow,
                                           roamAltitudeRow,
                                           _aircraftDynamicsRow,
                                           roamSpeedRow,
                                           roamUpdateRateRow,
                                           _roamStatusLabel,
                                         ]
                                      defaultsKey:@"panorama.inspector.movement.expanded"];

  _groundClearanceDecreaseControl =
      [NSButton buttonWithImage:[NSImage imageWithSystemSymbolName:@"minus"
                                          accessibilityDescription:@"Lower observer"]
                         target:self
                         action:@selector(adjustGroundClearance:)];
  _groundClearanceDecreaseControl.tag = -1;
  _groundClearanceDecreaseControl.controlSize = NSControlSizeSmall;
  _groundClearanceDecreaseControl.bezelStyle = NSBezelStyleTexturedRounded;
  _groundClearanceDecreaseControl.toolTip = @"Lower eye height by 1 m (Option: 0.1 m; Shift: 10 m)";
  [_groundClearanceDecreaseControl.widthAnchor constraintEqualToConstant:24.0].active = YES;

  _groundClearanceControl = [[NSTextField alloc] initWithFrame:NSZeroRect];
  _groundClearanceControl.delegate = self;
  _groundClearanceControl.alignment = NSTextAlignmentRight;
  _groundClearanceControl.font = [NSFont monospacedDigitSystemFontOfSize:12.0
                                                                  weight:NSFontWeightRegular];
  _groundClearanceControl.stringValue = [NSString stringWithFormat:@"%.1f", _groundClearance];
  _groundClearanceControl.toolTip = @"Observer height above the terrain directly beneath it";
  [_groundClearanceControl.widthAnchor constraintEqualToConstant:56.0].active = YES;

  _observerHeightUnit = [NSTextField labelWithString:@"m AGL"];
  _observerHeightUnit.textColor = NSColor.secondaryLabelColor;
  _observerHeightUnit.toolTip = @"Metres above ground level";

  _groundClearanceIncreaseControl =
      [NSButton buttonWithImage:[NSImage imageWithSystemSymbolName:@"plus"
                                          accessibilityDescription:@"Raise observer"]
                         target:self
                         action:@selector(adjustGroundClearance:)];
  _groundClearanceIncreaseControl.tag = 1;
  _groundClearanceIncreaseControl.controlSize = NSControlSizeSmall;
  _groundClearanceIncreaseControl.bezelStyle = NSBezelStyleTexturedRounded;
  _groundClearanceIncreaseControl.toolTip = @"Raise eye height by 1 m (Option: 0.1 m; Shift: 10 m)";
  [_groundClearanceIncreaseControl.widthAnchor constraintEqualToConstant:24.0].active = YES;

  NSStackView *heightSetting = [NSStackView stackViewWithViews:@[
    _groundClearanceDecreaseControl,
    _groundClearanceControl,
    _observerHeightUnit,
    _groundClearanceIncreaseControl,
  ]];
  heightSetting.orientation = NSUserInterfaceLayoutOrientationHorizontal;
  heightSetting.alignment = NSLayoutAttributeCenterY;
  heightSetting.spacing = 6.0;
  [heightSetting.widthAnchor constraintEqualToConstant:178.0].active = YES;

  _observerHeightLabel = [NSTextField labelWithString:@"Eye height"];
  _observerHeightLabel.toolTip = @"Observer height above the terrain directly beneath it";
  [_observerHeightLabel.widthAnchor constraintEqualToConstant:82.0].active = YES;
  NSStackView *heightRow =
      [NSStackView stackViewWithViews:@[ _observerHeightLabel, heightSetting ]];
  heightRow.orientation = NSUserInterfaceLayoutOrientationHorizontal;
  heightRow.alignment = NSLayoutAttributeCenterY;
  heightRow.spacing = 8.0;

  InspectorSectionView *observerSection =
      [[InspectorSectionView alloc] initWithTitle:@"Observer"
                                         controls:@[ heightRow ]
                                      defaultsKey:@"panorama.inspector.observer.expanded"];
  NSStackView *settings = [NSStackView
      stackViewWithViews:@[ heading, destinationSection, movementSection, observerSection ]];
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
  [self coordinateSystemChanged:_coordinateSystemControl];
  [self updateRoamControls];
  return viewController;
}

/// Build the read-only diagnostics displayed over the leading side of the
/// rendered scene. Camera values change with completed revisions; inspected
/// point details update independently as hover samples arrive.

@end

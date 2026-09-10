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

@implementation PanoramaController (InformationPanels)

- (NSViewController *)makeDebugViewController {
  NSViewController *viewController = [[NSViewController alloc] init];
  NSView *content = [[NSView alloc] initWithFrame:NSMakeRect(0.0, 0.0, 240.0, 430.0)];
  viewController.view = content;

  NSTextField *heading = [NSTextField labelWithString:@"Viewer Debug Info"];
  heading.font = [NSFont boldSystemFontOfSize:NSFont.systemFontSize];

  _debugInfoLabel = [NSTextField labelWithString:@""];
  _debugInfoLabel.font = [NSFont monospacedSystemFontOfSize:12.0 weight:NSFontWeightRegular];
  _debugInfoLabel.maximumNumberOfLines = 0;
  _debugInfoLabel.lineBreakMode = NSLineBreakByClipping;

  NSTextField *pointHeading = [NSTextField labelWithString:@"Inspected Point"];
  pointHeading.font = [NSFont boldSystemFontOfSize:NSFont.systemFontSize];
  _debugPointInfoLabel = [NSTextField labelWithString:@"No point selected."];
  _debugPointInfoLabel.font = [NSFont monospacedSystemFontOfSize:12.0 weight:NSFontWeightRegular];
  _debugPointInfoLabel.maximumNumberOfLines = 0;
  _debugPointInfoLabel.lineBreakMode = NSLineBreakByClipping;

  NSStackView *debugInfo = [NSStackView
      stackViewWithViews:@[ heading, _debugInfoLabel, pointHeading, _debugPointInfoLabel ]];
  debugInfo.orientation = NSUserInterfaceLayoutOrientationVertical;
  debugInfo.alignment = NSLayoutAttributeLeading;
  debugInfo.spacing = 12.0;
  debugInfo.translatesAutoresizingMaskIntoConstraints = NO;
  [content addSubview:debugInfo];
  [NSLayoutConstraint activateConstraints:@[
    [debugInfo.topAnchor constraintEqualToAnchor:content.topAnchor constant:16.0],
    [debugInfo.leadingAnchor constraintEqualToAnchor:content.leadingAnchor constant:16.0],
    [debugInfo.trailingAnchor constraintLessThanOrEqualToAnchor:content.trailingAnchor
                                                       constant:-16.0],
  ]];

  [self updateDebugInfoWithOrientation:_orientation
                   verticalFieldOfView:_verticalFieldOfView
                                 image:_image
                          milliseconds:0.0
                       gpuMilliseconds:0.0
                              streamed:NO
                              revision:0U];
  return viewController;
}

- (void)updateDebugInfoWithOrientation:(panorama::CameraOrientation)orientation
                   verticalFieldOfView:(double)verticalFieldOfView
                                 image:(panorama::ImageSize)image
                          milliseconds:(double)milliseconds
                       gpuMilliseconds:(double)gpuMilliseconds
                              streamed:(BOOL)streamed
                              revision:(uint64_t)revision {
  if (_debugInfoLabel == nil) {
    return;
  }

  const panorama::ObserverLocation observer = _renderer->observer();
  double heading = std::fmod(orientation.heading * panorama::app::kRadiansToDegrees, 360.0);
  if (heading < 0.0) {
    heading += 360.0;
  }
  const double fps = milliseconds > 0.0 ? 1'000.0 / milliseconds : 0.0;
  NSString *performance =
      milliseconds > 0.0
          ? [NSString
                stringWithFormat:@"FPS          %8.2f\nWall latency %8.2f ms", fps, milliseconds]
          : @"FPS                 —\nWall latency        —";
  performance = [performance
      stringByAppendingString:streamed ? @"\nGPU frame      streaming"
                                       : [NSString stringWithFormat:@"\nGPU frame    %8.2f ms",
                                                                    gpuMilliseconds]];
  _debugInfoLabel.stringValue = [NSString
      stringWithFormat:@"%@\nRevision     %8llu\n\n"
                        "Easting    %11.2f m\nNorthing   %11.2f m\nElevation  %11.2f m\n\n"
                        "Heading      %8.2f°\nPitch        %8.2f°\nRoll         %8.2f°\n"
                        "V. FOV       %8.2f°\n\nResolution   %4u × %4u\nMax range  %10.0f m",
                       performance,
                       static_cast<unsigned long long>(revision),
                       observer.easting,
                       observer.northing,
                       observer.elevation,
                       heading,
                       orientation.pitch * panorama::app::kRadiansToDegrees,
                       orientation.roll * panorama::app::kRadiansToDegrees,
                       verticalFieldOfView * panorama::app::kRadiansToDegrees,
                       image.width,
                       image.height,
                       _renderer->max_distance()];
}

/// Build the compact hover readout shown while point inspection is enabled.
- (NSViewController *)makePointInfoViewController {
  NSViewController *viewController = [[NSViewController alloc] init];
  NSView *content = [[NSView alloc] initWithFrame:NSMakeRect(0.0, 0.0, 230.0, 36.0)];
  viewController.view = content;

  _observerInfoLabel = [NSTextField labelWithString:@""];
  _observerInfoLabel.font = [NSFont monospacedSystemFontOfSize:10.0 weight:NSFontWeightRegular];
  _observerInfoLabel.textColor = NSColor.secondaryLabelColor;
  _observerInfoLabel.maximumNumberOfLines = 2;
  _observerInfoLabel.lineBreakMode = NSLineBreakByClipping;
  _movementInfoLabel = [NSTextField labelWithString:@""];
  _movementInfoLabel.font = [NSFont monospacedSystemFontOfSize:10.0 weight:NSFontWeightRegular];
  _movementInfoLabel.textColor = NSColor.secondaryLabelColor;
  _movementInfoLabel.maximumNumberOfLines = 2;
  _movementInfoLabel.lineBreakMode = NSLineBreakByClipping;
  _movementInfoLabel.hidden = YES;

  _pointInfoHeading = [NSTextField labelWithString:@" "];
  _pointInfoHeading.font = [NSFont boldSystemFontOfSize:NSFont.systemFontSize];
  _pointInfoHeading.maximumNumberOfLines = 1;
  _pointInfoHeading.lineBreakMode = NSLineBreakByTruncatingTail;
  _pointInfoLabel = [NSTextField labelWithString:@""];
  _pointInfoLabel.font = [NSFont monospacedSystemFontOfSize:12.0 weight:NSFontWeightRegular];
  _pointInfoLabel.maximumNumberOfLines = 1;
  _pointInfoLabel.lineBreakMode = NSLineBreakByClipping;

  _pointVisibilityIcon = [[NSImageView alloc] initWithFrame:NSZeroRect];
  _pointVisibilityIcon.contentTintColor = NSColor.secondaryLabelColor;
  _pointVisibilityIcon.hidden = YES;
  _pointLockIcon = [[NSImageView alloc] initWithFrame:NSZeroRect];
  _pointLockIcon.contentTintColor = NSColor.secondaryLabelColor;
  _pointLockIcon.hidden = YES;

  _moveToLockedPointControl = [NSButton buttonWithTitle:@"Move here"
                                                 target:self
                                                 action:@selector(moveToLockedPoint:)];
  _moveToLockedPointControl.controlSize = NSControlSizeSmall;
  _moveToLockedPointControl.bezelStyle = NSBezelStyleRounded;
  _moveToLockedPointControl.toolTip = @"Move the observer to the locked terrain point";
  _moveToLockedPointControl.hidden = YES;

  NSView *pointSpacer = [[NSView alloc] initWithFrame:NSZeroRect];
  [pointSpacer setContentHuggingPriority:NSLayoutPriorityDefaultLow
                          forOrientation:NSLayoutConstraintOrientationHorizontal];
  [pointSpacer setContentCompressionResistancePriority:NSLayoutPriorityDefaultLow
                                        forOrientation:NSLayoutConstraintOrientationHorizontal];

  _pointInfoRow = [NSStackView stackViewWithViews:@[
    _pointInfoHeading,
    _pointInfoLabel,
    _pointVisibilityIcon,
    _pointLockIcon,
    pointSpacer,
    _moveToLockedPointControl,
  ]];
  _pointInfoRow.orientation = NSUserInterfaceLayoutOrientationHorizontal;
  _pointInfoRow.alignment = NSLayoutAttributeCenterY;
  _pointInfoRow.spacing = 7.0;

  NSStackView *footer = [NSStackView stackViewWithViews:@[
    _observerInfoLabel,
    _movementInfoLabel,
    _pointInfoRow,
  ]];
  footer.orientation = NSUserInterfaceLayoutOrientationVertical;
  footer.alignment = NSLayoutAttributeLeading;
  footer.spacing = 2.0;
  footer.translatesAutoresizingMaskIntoConstraints = NO;
  [content addSubview:footer];
  [NSLayoutConstraint activateConstraints:@[
    [footer.topAnchor constraintEqualToAnchor:content.topAnchor constant:6.0],
    [footer.bottomAnchor constraintEqualToAnchor:content.bottomAnchor constant:-6.0],
    [footer.leadingAnchor constraintEqualToAnchor:content.leadingAnchor constant:14.0],
    [footer.trailingAnchor constraintEqualToAnchor:content.trailingAnchor constant:-14.0],
    [_observerInfoLabel.widthAnchor constraintEqualToAnchor:footer.widthAnchor],
    [_movementInfoLabel.widthAnchor constraintEqualToAnchor:footer.widthAnchor],
    [_pointInfoRow.widthAnchor constraintEqualToAnchor:footer.widthAnchor],
    [_pointInfoRow.heightAnchor constraintEqualToConstant:24.0],
  ]];
  return viewController;
}

@end

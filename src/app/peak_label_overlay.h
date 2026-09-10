#pragma once

#include "peak_catalogue.h"

#import <AppKit/AppKit.h>

#include <optional>

@interface PeakLabelOverlayView : NSView
- (void)setLabelMode:(panorama::app::PeakLabelMode)mode;
- (void)setPeakFrame:(const panorama::app::PeakLabelFrame &)frame;
- (void)setPointerLocation:(std::optional<NSPoint>)location;
@end

#include "peak_label_overlay.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

@interface PeakLabelOverlayView () {
  panorama::app::PeakLabelMode _mode;
  panorama::app::PeakLabelFrame _frame;
  std::optional<NSPoint> _pointer;
}
@end

@implementation PeakLabelOverlayView

- (BOOL)isOpaque {
  return NO;
}
- (NSView *)hitTest:(NSPoint)point {
  (void)point;
  return nil;
}
- (void)setLabelMode:(panorama::app::PeakLabelMode)mode {
  _mode = mode;
  self.needsDisplay = YES;
}
- (void)setPeakFrame:(const panorama::app::PeakLabelFrame &)frame {
  _frame = frame;
  self.needsDisplay = YES;
}
- (void)setPointerLocation:(std::optional<NSPoint>)location {
  _pointer = location;
  self.needsDisplay = YES;
}

- (void)drawRect:(NSRect)dirtyRect {
  (void)dirtyRect;
  if (_mode == panorama::app::PeakLabelMode::Off || _frame.output_image.width == 0 ||
      _frame.output_image.height == 0 ||
      (_mode == panorama::app::PeakLabelMode::NearPointer && !_pointer.has_value()))
    return;
  constexpr CGFloat kPointerRadius = 140.0;
  const NSUInteger limit = _mode == panorama::app::PeakLabelMode::NearPointer ? 12 : 30;
  NSMutableParagraphStyle *paragraph = [[NSMutableParagraphStyle alloc] init];
  paragraph.alignment = NSTextAlignmentCenter;
  NSDictionary *attributes = @{
    NSFontAttributeName : [NSFont systemFontOfSize:11.0 weight:NSFontWeightSemibold],
    NSForegroundColorAttributeName : NSColor.whiteColor,
    NSParagraphStyleAttributeName : paragraph,
  };
  std::vector<NSRect> occupied;
  NSUInteger displayed = 0;
  for (const auto &peak : _frame.peaks) {
    const NSPoint anchor = NSMakePoint(
        peak.pixel_x / _frame.output_image.width * self.bounds.size.width,
        self.bounds.size.height -
            peak.pixel_y / _frame.output_image.height * self.bounds.size.height
    );
    if (_mode == panorama::app::PeakLabelMode::NearPointer &&
        std::hypot(anchor.x - _pointer->x, anchor.y - _pointer->y) > kPointerRadius)
      continue;
    NSString *name = [NSString stringWithUTF8String:peak.name.c_str()];
    NSString *label = [NSString stringWithFormat:@"%@ · %.0f m", name, peak.elevation];
    NSSize size = [label sizeWithAttributes:attributes];
    size.width += 12.0;
    size.height += 6.0;
    // Prefer successively higher placements so labels can rise clear of a
    // crowded mountain skyline. Side and lower placements remain fallbacks
    // for peaks close to the top of the view.
    const std::array<NSPoint, 7> origins = {
        NSMakePoint(anchor.x - size.width * 0.5, anchor.y + 12.0),
        NSMakePoint(anchor.x - size.width * 0.5, anchor.y + 40.0),
        NSMakePoint(anchor.x - size.width * 0.5, anchor.y + 72.0),
        NSMakePoint(anchor.x - size.width * 0.5, anchor.y + 108.0),
        NSMakePoint(anchor.x + 12.0, anchor.y - size.height * 0.5),
        NSMakePoint(anchor.x - size.width - 12.0, anchor.y - size.height * 0.5),
        NSMakePoint(anchor.x - size.width * 0.5, anchor.y - size.height - 12.0),
    };
    std::optional<NSRect> placement;
    for (const NSPoint origin : origins) {
      const NSRect candidate = NSMakeRect(origin.x, origin.y, size.width, size.height);
      if (!NSContainsRect(NSInsetRect(self.bounds, 4.0, 4.0), candidate))
        continue;
      bool overlaps = false;
      for (NSRect other : occupied)
        overlaps = overlaps || NSIntersectsRect(NSInsetRect(candidate, -3.0, -3.0), other);
      if (!overlaps) {
        placement = candidate;
        break;
      }
    }
    if (!placement.has_value())
      continue;
    const NSRect box = *placement;
    occupied.push_back(box);
    const NSPoint leaderEnd = NSMakePoint(
        std::clamp(anchor.x, NSMinX(box), NSMaxX(box)),
        std::clamp(anchor.y, NSMinY(box), NSMaxY(box))
    );
    [[NSColor colorWithWhite:1.0 alpha:0.72] setStroke];
    NSBezierPath *leader = [NSBezierPath bezierPath];
    leader.lineWidth = 1.0;
    [leader moveToPoint:anchor];
    [leader lineToPoint:leaderEnd];
    [leader stroke];
    [[NSColor colorWithWhite:0.05 alpha:0.72] setFill];
    [[NSBezierPath bezierPathWithRoundedRect:box xRadius:5.0 yRadius:5.0] fill];
    [NSColor.whiteColor setFill];
    [[NSBezierPath bezierPathWithOvalInRect:NSMakeRect(anchor.x - 2.5, anchor.y - 2.5, 5.0, 5.0)]
        fill];
    [label drawInRect:NSInsetRect(box, 6.0, 3.0) withAttributes:attributes];
    if (++displayed >= limit)
      break;
  }
}

@end

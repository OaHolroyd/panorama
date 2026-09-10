#include "viewer_overlay.h"

#include <algorithm>
#include <cmath>

/// Letterbox the render view at its current output aspect ratio without
/// imposing a fitting size on its parent. The parent can therefore resize
/// freely, and a resolution change can update the ratio without distorting the
/// Metal output.
@interface AspectFitContainerView () {
  NSView *_renderView;
  CGFloat _aspectRatio;
}
@end

@implementation AspectFitContainerView

- (instancetype)initWithFrame:(NSRect)frame
                   renderView:(NSView *)renderView
                  aspectRatio:(CGFloat)aspectRatio {
  self = [super initWithFrame:frame];
  if (self != nil) {
    _renderView = renderView;
    _aspectRatio = aspectRatio;
    self.wantsLayer = YES;
    self.layer.backgroundColor = NSColor.blackColor.CGColor;
    [self addSubview:_renderView];
  }
  return self;
}

- (void)setAspectRatio:(CGFloat)aspectRatio {
  if (aspectRatio <= 0.0 || std::abs(aspectRatio - _aspectRatio) <= 1e-9) {
    return;
  }
  _aspectRatio = aspectRatio;
  [self setNeedsLayout:YES];
}

- (void)layout {
  [super layout];
  const NSRect bounds = self.bounds;
  CGFloat width = bounds.size.width;
  CGFloat height = width / _aspectRatio;
  if (height > bounds.size.height) {
    height = bounds.size.height;
    width = height * _aspectRatio;
  }
  _renderView.frame = NSMakeRect(
      bounds.origin.x + (bounds.size.width - width) * 0.5,
      bounds.origin.y + (bounds.size.height - height) * 0.5,
      width,
      height
  );
}

@end

/// A flipped document view keeps the first inspector section at the top when
/// its intrinsic height grows beyond the scroll view's visible area.
@implementation InspectorDocumentView

- (BOOL)isFlipped {
  return YES;
}

@end

/// A compact inspector section whose disclosure state survives app launches.
/// Hiding the content stack removes it from its parent stack's fitting height,
/// so collapsed sections also shorten the scrollable document immediately.
@interface InspectorSectionView () {
  NSButton *_disclosureButton;
  NSStackView *_contentStack;
  NSString *_defaultsKey;
}
@end

@implementation InspectorSectionView

- (instancetype)initWithTitle:(NSString *)title
                     controls:(NSArray<NSView *> *)controls
                  defaultsKey:(NSString *)defaultsKey {
  self = [super initWithFrame:NSZeroRect];
  if (self != nil) {
    _defaultsKey = [defaultsKey copy];
    _disclosureButton = [NSButton buttonWithTitle:title
                                           target:self
                                           action:@selector(toggleDisclosure:)];
    _disclosureButton.bordered = NO;
    _disclosureButton.imagePosition = NSImageLeading;
    _disclosureButton.alignment = NSTextAlignmentLeft;
    _disclosureButton.font = [NSFont systemFontOfSize:NSFont.smallSystemFontSize
                                               weight:NSFontWeightSemibold];

    _contentStack = [NSStackView stackViewWithViews:controls];
    _contentStack.orientation = NSUserInterfaceLayoutOrientationVertical;
    _contentStack.alignment = NSLayoutAttributeLeading;
    _contentStack.spacing = 8.0;

    self.orientation = NSUserInterfaceLayoutOrientationVertical;
    self.alignment = NSLayoutAttributeLeading;
    self.spacing = 8.0;
    [self addArrangedSubview:_disclosureButton];
    [self addArrangedSubview:_contentStack];

    NSNumber *saved = [NSUserDefaults.standardUserDefaults objectForKey:_defaultsKey];
    [self setExpanded:saved == nil || saved.boolValue];
  }
  return self;
}

- (void)setExpanded:(BOOL)expanded {
  _contentStack.hidden = !expanded;
  _disclosureButton.image =
      [NSImage imageWithSystemSymbolName:expanded ? @"chevron.down" : @"chevron.right"
                accessibilityDescription:expanded ? @"Collapse section" : @"Expand section"];
  _disclosureButton.accessibilityValue = expanded ? @"Expanded" : @"Collapsed";
}

- (void)toggleDisclosure:(id)sender {
  (void)sender;
  const BOOL expanded = _contentStack.hidden;
  [self setExpanded:expanded];
  [NSUserDefaults.standardUserDefaults setBool:expanded forKey:_defaultsKey];
}

@end

/// Wrap custom overlay content in the current platform's native translucent
/// material. The returned view owns `contentView` through either the modern
/// Liquid Glass API or the pre-macOS 26 visual-effect fallback.
static NSView *makeOverlayPanel(NSView *contentView) {
  NSView *panel = nil;
  if (@available(macOS 26.0, *)) {
    NSGlassEffectView *glass = [[NSGlassEffectView alloc] initWithFrame:NSZeroRect];
    glass.style = NSGlassEffectViewStyleRegular;
    glass.cornerRadius = 18.0;
    glass.contentView = contentView;
    panel = glass;
  } else {
    NSVisualEffectView *material = [[NSVisualEffectView alloc] initWithFrame:NSZeroRect];
    material.material = NSVisualEffectMaterialSidebar;
    material.blendingMode = NSVisualEffectBlendingModeWithinWindow;
    material.state = NSVisualEffectStateActive;
    material.wantsLayer = YES;
    material.layer.cornerRadius = 18.0;
    material.layer.masksToBounds = YES;
    [material addSubview:contentView];
    panel = material;
  }
  contentView.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
  return panel;
}

/// Keep the render at the full window size and position auxiliary panels above
/// it. Frame-based layout is deliberate: overlapping children do not define a
/// useful Auto Layout fitting size for an NSWindow content view.
@interface ViewerOverlayView () {
  NSView *_contentView;
  NSView *_inspectorView;
  NSView *_settingsView;
  NSView *_debugView;
  NSView *_debugContentView;
  NSView *_mapPanelView;
  MiniMapPanelView *_mapPanelContentView;
  CGFloat _inspectorWidth;
  NSSize _debugSize;
  CGFloat _panelMargin;
  bool _inspectorVisible;
  bool _debugVisible;
  bool _mapAndPointInfoVisible;
}
@end

@implementation ViewerOverlayView

- (instancetype)initWithFrame:(NSRect)frame
                  contentView:(NSView *)contentView
                 settingsView:(NSView *)settingsView
               inspectorWidth:(CGFloat)inspectorWidth
                    debugView:(NSView *)debugView
                    debugSize:(NSSize)debugSize
                 mapPanelView:(MiniMapPanelView *)mapPanelView {
  self = [super initWithFrame:frame];
  if (self != nil) {
    _contentView = contentView;
    _settingsView = settingsView;
    _debugContentView = debugView;
    _mapPanelContentView = mapPanelView;
    _mapPanelContentView.sizeDelegate = self;
    _inspectorWidth = inspectorWidth;
    _debugSize = debugSize;
    _panelMargin = 12.0;
    _inspectorVisible = true;
    _debugVisible = false;
    _mapAndPointInfoVisible = false;
    self.wantsLayer = YES;
    self.layer.masksToBounds = YES;
    [self addSubview:_contentView];

    _inspectorView = makeOverlayPanel(_settingsView);
    _debugView = makeOverlayPanel(_debugContentView);
    _mapPanelView = makeOverlayPanel(_mapPanelContentView);
    [self addSubview:_inspectorView];
    [self addSubview:_debugView];
    [self addSubview:_mapPanelView];
  }
  return self;
}

- (NSRect)inspectorFrameForVisible:(bool)visible {
  const NSRect bounds = self.bounds;
  // Full-size window content extends beneath the titlebar so the native
  // toolbar can float over it. Keep the inspector inside AppKit's safe area,
  // clear of the toolbar and window controls, while allowing the terrain view
  // itself to fill the window.
  const NSEdgeInsets safeArea = self.safeAreaInsets;
  const CGFloat x = visible ? NSMaxX(bounds) - safeArea.right - _inspectorWidth - _panelMargin
                            : NSMaxX(bounds) + _panelMargin;
  const CGFloat bottom = safeArea.bottom + _panelMargin;
  const CGFloat top = safeArea.top + _panelMargin;
  const CGFloat height = std::max(0.0, bounds.size.height - bottom - top);
  return NSMakeRect(x, bounds.origin.y + bottom, _inspectorWidth, height);
}

- (NSRect)debugFrameForVisible:(bool)visible {
  const NSRect bounds = self.bounds;
  const NSEdgeInsets safeArea = self.safeAreaInsets;
  const CGFloat availableHeight =
      std::max(0.0, bounds.size.height - safeArea.top - safeArea.bottom - 2.0 * _panelMargin);
  const CGFloat height = std::min(_debugSize.height, availableHeight);
  const CGFloat x = visible ? NSMinX(bounds) + safeArea.left + _panelMargin
                            : NSMinX(bounds) - _debugSize.width - _panelMargin;
  const CGFloat y = NSMaxY(bounds) - safeArea.top - _panelMargin - height;
  return NSMakeRect(x, y, _debugSize.width, height);
}

- (NSRect)mapPanelFrameForVisible:(bool)visible {
  const NSRect bounds = self.bounds;
  const NSEdgeInsets safeArea = self.safeAreaInsets;
  NSSize preferredSize = [_mapPanelContentView preferredPanelSize];
  // Preserve the outgoing panel's size while its final visible section slides
  // away; the content reports zero height once both sections are disabled.
  if (!visible && preferredSize.height == 0.0 && _mapPanelView.frame.size.height > 0.0) {
    preferredSize = _mapPanelView.frame.size;
  }
  const CGFloat availableHeight =
      std::max(0.0, bounds.size.height - safeArea.top - safeArea.bottom - 2.0 * _panelMargin);
  const CGFloat availableWidth =
      std::max(0.0, bounds.size.width - safeArea.left - safeArea.right - 2.0 * _panelMargin);
  const CGFloat width = std::min(preferredSize.width, availableWidth);
  const CGFloat height = std::min(preferredSize.height, availableHeight);
  const CGFloat x = visible ? NSMinX(bounds) + safeArea.left + _panelMargin
                            : NSMinX(bounds) - width - _panelMargin;
  const CGFloat y = NSMinY(bounds) + safeArea.bottom + _panelMargin;
  return NSMakeRect(x, y, width, height);
}

- (void)layout {
  [super layout];
  _contentView.frame = self.bounds;
  _inspectorView.frame = [self inspectorFrameForVisible:_inspectorVisible];
  _settingsView.frame = _inspectorView.bounds;
  _debugView.frame = [self debugFrameForVisible:_debugVisible];
  _debugContentView.frame = _debugView.bounds;
  _mapPanelView.frame = [self mapPanelFrameForVisible:_mapAndPointInfoVisible];
  _mapPanelContentView.frame = _mapPanelView.bounds;
}

- (void)miniMapPanelPreferredSizeDidChange:(MiniMapPanelView *)panel {
  if (panel != _mapPanelContentView) {
    return;
  }
  const NSRect targetFrame = [self mapPanelFrameForVisible:_mapAndPointInfoVisible];
  [NSAnimationContext runAnimationGroup:^(NSAnimationContext *context) {
    context.duration = 0.25;
    _mapPanelView.animator.frame = targetFrame;
    _mapPanelContentView.animator.frame =
        NSMakeRect(0.0, 0.0, targetFrame.size.width, targetFrame.size.height);
  }];
}

- (void)toggleInspector:(id)sender {
  (void)sender;
  _inspectorVisible = !_inspectorVisible;
  [NSAnimationContext runAnimationGroup:^(NSAnimationContext *context) {
    context.duration = 0.25;
    _inspectorView.animator.frame = [self inspectorFrameForVisible:_inspectorVisible];
  }];
}

- (void)toggleDebugOverlay:(id)sender {
  (void)sender;
  _debugVisible = !_debugVisible;
  [NSAnimationContext runAnimationGroup:^(NSAnimationContext *context) {
    context.duration = 0.25;
    _debugView.animator.frame = [self debugFrameForVisible:_debugVisible];
  }];
}

- (void)setMapAndPointInfoVisible:(bool)visible {
  if (_mapAndPointInfoVisible == visible) {
    return;
  }
  _mapAndPointInfoVisible = visible;
  [_mapPanelContentView setMapAndPointInfoVisible:visible];
  const NSRect targetFrame = [self mapPanelFrameForVisible:visible];
  [NSAnimationContext runAnimationGroup:^(NSAnimationContext *context) {
    context.duration = 0.25;
    _mapPanelView.animator.frame = targetFrame;
    _mapPanelContentView.animator.frame =
        NSMakeRect(0.0, 0.0, targetFrame.size.width, targetFrame.size.height);
  }];
}

@end

#pragma once

#include "minimap.h"

#import <AppKit/AppKit.h>

/// Letterboxes the Metal view at the selected render aspect ratio.
@interface AspectFitContainerView : NSView
- (instancetype)initWithFrame:(NSRect)frame
                   renderView:(NSView *)renderView
                  aspectRatio:(CGFloat)aspectRatio;
- (void)setAspectRatio:(CGFloat)aspectRatio;
@end

@interface InspectorDocumentView : NSView
@end

@interface InspectorSectionView : NSStackView
- (instancetype)initWithTitle:(NSString *)title
                     controls:(NSArray<NSView *> *)controls
                  defaultsKey:(NSString *)defaultsKey;
@end

/// Owns the terrain content and independently animated inspector, diagnostics,
/// and minimap panels layered above it.
@interface ViewerOverlayView : NSView <MiniMapPanelViewSizeDelegate>
- (instancetype)initWithFrame:(NSRect)frame
                  contentView:(NSView *)contentView
                 settingsView:(NSView *)settingsView
               inspectorWidth:(CGFloat)inspectorWidth
                    debugView:(NSView *)debugView
                    debugSize:(NSSize)debugSize
                 mapPanelView:(MiniMapPanelView *)mapPanelView;
- (void)toggleInspector:(id)sender;
- (void)toggleDebugOverlay:(id)sender;
- (void)setMapAndPointInfoVisible:(bool)visible;
@end

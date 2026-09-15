#pragma once

#include "minimap.h"

#import <AppKit/AppKit.h>

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
- (bool)isInspectorEnabled;
- (void)toggleDebugOverlay:(id)sender;
- (bool)isDebugInfoEnabled;
- (void)setMapAndPointInfoVisible:(bool)visible;
@end

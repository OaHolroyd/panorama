#pragma once

#include "ray_projection.h"

#import <AppKit/AppKit.h>
#import <Metal/Metal.h>

#include <cstdint>

@class MiniMapPanelView;

@protocol MiniMapPanelViewSizeDelegate <NSObject>
- (void)miniMapPanelPreferredSizeDidChange:(MiniMapPanelView *)panel;
@end

/// Combined fixed-centre map and terrain-point readout used by the viewer's
/// leading overlay. MapKit implementation details remain in minimap.mm so the
/// application controller only publishes camera and inspected-world state.
@interface MiniMapPanelView : NSView

@property(nonatomic, weak) id<MiniMapPanelViewSizeDelegate> sizeDelegate;

- (instancetype)initWithObserverEasting:(double)easting
                               northing:(double)northing
                        terrainEpsgCode:(uint32_t)epsgCode
                            maxDistance:(double)maxDistance
                          pointInfoView:(NSView *)pointInfoView
                            metalDevice:(id<MTLDevice>)metalDevice
                           commandQueue:(id<MTLCommandQueue>)commandQueue
                                library:(id<MTLLibrary>)library;

- (void)setMapVisible:(bool)visible;
- (void)setPointInfoVisible:(bool)visible;
- (bool)hasVisibleContent;
- (NSSize)preferredPanelSize;

/// Update the projected heading wedge to match the displayed camera view.
- (void)setCameraOrientation:(panorama::CameraOrientation)orientation
         verticalFieldOfView:(double)verticalFieldOfView
                       image:(panorama::ImageSize)image;

/// Display one projected point for every valid collision in a completed trace.
- (void)setVisibilityPoints:(id<MTLBuffer>)points image:(panorama::ImageSize)image;

- (void)setInspectedPointEasting:(double)easting northing:(double)northing locked:(bool)locked;
- (void)clearInspectedPoint;

@end

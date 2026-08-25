#pragma once

#include "ray_projection.h"

#import <AppKit/AppKit.h>

#include <cstdint>

/// Combined fixed-centre map and terrain-point readout used by the viewer's
/// leading overlay. MapKit implementation details remain in minimap.mm so the
/// application controller only publishes camera and inspected-world state.
@interface MiniMapPanelView : NSView

- (instancetype)initWithObserverEasting:(double)easting
                               northing:(double)northing
                        terrainEpsgCode:(uint32_t)epsgCode
                            maxDistance:(double)maxDistance
                          pointInfoView:(NSView *)pointInfoView;

- (void)setMapVisible:(bool)visible;
- (void)setPointInfoVisible:(bool)visible;
- (bool)hasVisibleContent;
- (NSSize)preferredPanelSize;

/// Update the projected heading wedge to match the displayed camera view.
- (void)setCameraOrientation:(panorama::CameraOrientation)orientation
         verticalFieldOfView:(double)verticalFieldOfView
                       image:(panorama::ImageSize)image;

- (void)setInspectedPointEasting:(double)easting northing:(double)northing locked:(bool)locked;
- (void)clearInspectedPoint;

@end

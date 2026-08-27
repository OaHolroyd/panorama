#pragma once

#include "ray_projection.h"
#include "terrain_catalogue.h"

#import <AppKit/AppKit.h>
#import <Metal/Metal.h>

#include <cstdint>

@class MiniMapPanelView;

@protocol MiniMapPanelViewSizeDelegate <NSObject>
- (void)miniMapPanelPreferredSizeDidChange:(MiniMapPanelView *)panel;
@end

@protocol MiniMapPanelViewInteractionDelegate <NSObject>
/// Hover previews a terrain point; primary click selects it for viewing; a
/// secondary click or Option-click requests observer movement.
- (void)miniMapPanel:(MiniMapPanelView *)panel
     didHoverEasting:(double)easting
            northing:(double)northing;
- (void)miniMapPanelDidEndHover:(MiniMapPanelView *)panel;
- (void)miniMapPanel:(MiniMapPanelView *)panel
    didSelectEasting:(double)easting
            northing:(double)northing;
- (void)miniMapPanel:(MiniMapPanelView *)panel
    didRequestObserverMoveToEasting:(double)easting
                           northing:(double)northing;
@end

/// Combined freely navigable map and terrain-point readout used by the viewer's
/// leading overlay. MapKit implementation details remain in minimap.mm so the
/// application controller only publishes camera and inspected-world state.
@interface MiniMapPanelView : NSView

@property(nonatomic, weak) id<MiniMapPanelViewSizeDelegate> sizeDelegate;
@property(nonatomic, weak) id<MiniMapPanelViewInteractionDelegate> interactionDelegate;

- (instancetype)initWithObserverEasting:(double)easting
                               northing:(double)northing
                        terrainEpsgCode:(uint32_t)epsgCode
                        terrainCoverage:(const panorama::TerrainCoverage &)coverage
               coverageInitiallyVisible:(bool)coverageVisible
                            maxDistance:(double)maxDistance
                          pointInfoView:(NSView *)pointInfoView
                            metalDevice:(id<MTLDevice>)metalDevice
                           commandQueue:(id<MTLCommandQueue>)commandQueue
                                library:(id<MTLLibrary>)library;

/// Show or hide the map and information footer as one coupled surface.
- (void)setMapAndPointInfoVisible:(bool)visible;
/// Return the compact or expanded size, including the information footer.
- (NSSize)preferredPanelSize;
/// Re-measure the footer after its observer, movement, or point rows change.
- (void)informationFooterContentDidChange;

/// Update the projected heading wedge to match the displayed camera view.
- (void)setCameraOrientation:(panorama::CameraOrientation)orientation
         verticalFieldOfView:(double)verticalFieldOfView
                       image:(panorama::ImageSize)image;

/// Display one projected point for every valid collision in a completed trace.
- (void)setVisibilityPoints:(id<MTLBuffer>)points image:(panorama::ImageSize)image;

/// Move all observer-relative map graphics after an interactive relocation.
- (void)setObserverEasting:(double)easting northing:(double)northing;
/// Recenter on the observer without changing the current map scale.
- (void)centerOnObserver;

/// Display the hover or locked point using the viewer's blue/orange convention.
- (void)setInspectedPointEasting:(double)easting northing:(double)northing locked:(bool)locked;
- (void)clearInspectedPoint;

@end

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
    didHoverLatitude:(double)latitude
           longitude:(double)longitude;
- (void)miniMapPanelDidEndHover:(MiniMapPanelView *)panel;
- (void)miniMapPanel:(MiniMapPanelView *)panel
    didSelectLatitude:(double)latitude
            longitude:(double)longitude;
- (void)miniMapPanel:(MiniMapPanelView *)panel
    didRequestObserverMoveToLatitude:(double)latitude
                           longitude:(double)longitude;
@end

/// Combined freely navigable map and terrain-point readout used by the viewer's
/// leading overlay. MapKit implementation details remain in minimap.mm so the
/// application controller only publishes camera and inspected-world state.
@interface MiniMapPanelView : NSView

@property(nonatomic, weak) id<MiniMapPanelViewSizeDelegate> sizeDelegate;
@property(nonatomic, weak) id<MiniMapPanelViewInteractionDelegate> interactionDelegate;

- (instancetype)initWithObserverLatitude:(double)latitude
                               longitude:(double)longitude
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

/// Coalesce a coverage-mask update from a completed, immutable hit snapshot.
/// Hidden panels discard it; map-only navigation reuses the visible snapshot.
- (void)setVisibilityPoints:(id<MTLBuffer>)points
                      image:(panorama::ImageSize)image
                renderFrame:(panorama::TerrainRenderFrame)frame
                   observer:(panorama::LatLon)observer;

/// Move all observer-relative map graphics after an interactive relocation.
- (void)setObserverLatitude:(double)latitude longitude:(double)longitude;
/// Recenter on the observer without changing the current map scale.
- (void)centerOnObserver;

/// Display the hover or locked point using the viewer's blue/orange convention.
- (void)setInspectedPointLatitude:(double)latitude longitude:(double)longitude locked:(bool)locked;
- (void)clearInspectedPoint;

@end

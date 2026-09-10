#pragma once

#include "viewer_renderer.h"

#import <MetalKit/MetalKit.h>

#include <optional>

@protocol PanoramaViewController <NSObject>
- (void)pointerMovedOverOccludingView:(NSView *)view;
- (void)pointerMovedOverPanorama;
- (void)setCruiseSteeringX:(double)x y:(double)y;
- (void)mouseTurnForCurrentZoomHeading:(double)heading pitch:(double)pitch;
- (void)invalidatePanoramaHover;
- (void)inspectLocationX:(double)x y:(double)y;
- (void)togglePointLockAtLocationX:(double)x y:(double)y;
- (void)panoramaPointerExited;
- (void)panForCurrentZoomHeading:(double)heading pitch:(double)pitch;
- (void)zoomWithScrollDelta:(double)delta precise:(BOOL)precise;
- (BOOL)isMouseTurningEnabled;
- (BOOL)isCruisingEnabled;
- (BOOL)isRoamingEnabled;
- (void)rotateForCurrentZoomHeading:(double)heading pitch:(double)pitch;
- (void)adjustCruiseSpeedBy:(double)delta;
- (void)setRoamKey:(panorama::app::RoamKey)key pressed:(BOOL)pressed;
@end

@interface PanoramaView : MTKView
@property(nonatomic, weak) id<PanoramaViewController> panoramaController;
- (void)setPointInspectionEnabled:(bool)enabled;
- (void)setMouseTurningEnabled:(bool)enabled;
- (void)setCruiseSteeringEnabled:(bool)enabled;
- (void)setCruiseHUDHeading:(double)heading
                      pitch:(double)pitch
                       bank:(double)bank
        verticalFieldOfView:(double)verticalFieldOfView
               aircraftMode:(bool)aircraftMode;
- (void)setViewerPaused:(bool)paused recoveryMessage:(NSString *)recoveryMessage;
- (void)setTerrainPointIndicator:(std::optional<panorama::app::LockedPointProjection>)projection
                           image:(panorama::ImageSize)image
                          locked:(bool)locked
                        occluded:(bool)occluded;
@end

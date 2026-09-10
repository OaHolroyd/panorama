#pragma once

#include "minimap.h"
#include "panorama_view.h"
#include "viewer_overlay.h"
#include "viewer_renderer.h"

@interface PanoramaController : NSObject <
                                    MTKViewDelegate,
                                    NSTextFieldDelegate,
                                    MiniMapPanelViewInteractionDelegate,
                                    PanoramaViewController>
- (instancetype)initWithRenderer:(panorama::app::ViewerRenderer *)renderer
                          window:(NSWindow *)window;
- (void)selectRaytracer:(NSMenuItem *)sender;
- (void)attachPanoramaView:(PanoramaView *)panoramaView
               overlayView:(ViewerOverlayView *)overlayView
             aspectFitView:(AspectFitContainerView *)aspectFitView
              miniMapPanel:(MiniMapPanelView *)miniMapPanel;
- (void)toggleMapAndPointInspection:(id)sender;
- (BOOL)isMapAndPointInspectionEnabled;
- (NSViewController *)makeSettingsViewController;
- (NSViewController *)makePositioningViewController;
- (NSViewController *)makeDebugViewController;
- (NSViewController *)makePointInfoViewController;
@end

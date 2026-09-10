#pragma once

#include "minimap.h"
#include "panorama_view.h"
#include "viewer_overlay.h"
#include "viewer_renderer.h"

@interface PanoramaController : NSObject
- (instancetype)initWithRenderer:(panorama::app::ViewerRenderer *)renderer
                          window:(NSWindow *)window;
- (void)attachPanoramaView:(PanoramaView *)panoramaView
               overlayView:(ViewerOverlayView *)overlayView
             aspectFitView:(AspectFitContainerView *)aspectFitView
              miniMapPanel:(MiniMapPanelView *)miniMapPanel;
@end

@interface PanoramaController (PanoramaInput) <PanoramaViewController>
- (void)selectRaytracer:(NSMenuItem *)sender;
@end

@interface PanoramaController (MiniMapInteraction) <MiniMapPanelViewInteractionDelegate>
- (void)toggleMapAndPointInspection:(id)sender;
- (BOOL)isMapAndPointInspectionEnabled;
@end

@interface PanoramaController (TextEditing) <NSTextFieldDelegate>
@end

@interface PanoramaController (Rendering) <MTKViewDelegate>
@end

@interface PanoramaController (SettingsPanel)
- (NSViewController *)makeSettingsViewController;
@end

@interface PanoramaController (PositioningPanel)
- (NSViewController *)makePositioningViewController;
@end

@interface PanoramaController (InformationPanels)
- (NSViewController *)makeDebugViewController;
- (NSViewController *)makePointInfoViewController;
@end

#pragma once

#include "panorama_controller.h"

/// Owns the expandable toolbar field and its suggestions above the terrain.
@interface LocationSearchControl : NSObject
- (instancetype)initWithRenderer:(panorama::app::ViewerRenderer *)renderer
                      controller:(PanoramaController *)controller
                          window:(NSWindow *)window;
- (NSToolbarItem *)makeToolbarItemWithIdentifier:(NSToolbarItemIdentifier)identifier;
@end

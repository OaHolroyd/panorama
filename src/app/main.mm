#include "minimap.h"
#include "panorama_controller.h"
#include "panorama_view.h"
#include "trace_diagnostics.h"
#include "viewer_overlay.h"
#include "viewer_renderer.h"

#import <AppKit/AppKit.h>
#import <MetalKit/MetalKit.h>

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <utility>

static NSToolbarItemIdentifier const kDebugToolbarItemIdentifier = @"panorama.debug-info";
static NSToolbarItemIdentifier const kMapToolbarItemIdentifier = @"panorama.minimap";

@interface PanoramaAppDelegate : NSObject <NSApplicationDelegate, NSToolbarDelegate> {
@private
  std::unique_ptr<panorama::app::ViewerRenderer> _renderer;
  NSWindow *_window;
  PanoramaController *_controller;
  NSTabViewController *_inspectorController;
  ViewerOverlayView *_overlayView;
}
- (instancetype)initWithSettings:(panorama::app::ViewerSettings)settings;
@end

@implementation PanoramaAppDelegate

- (instancetype)initWithSettings:(panorama::app::ViewerSettings)settings {
  self = [super init];
  if (self != nil) {
    _renderer = panorama::app::make_viewer_renderer(std::move(settings));
  }
  return self;
}

/// Put the overlay controls at the trailing edge, matching native macOS apps.
- (NSArray<NSToolbarItemIdentifier> *)toolbarDefaultItemIdentifiers:(NSToolbar *)toolbar {
  (void)toolbar;
  return @[
    NSToolbarFlexibleSpaceItemIdentifier,
    kMapToolbarItemIdentifier,
    kDebugToolbarItemIdentifier,
    NSToolbarToggleInspectorItemIdentifier,
  ];
}

- (NSArray<NSToolbarItemIdentifier> *)toolbarAllowedItemIdentifiers:(NSToolbar *)toolbar {
  (void)toolbar;
  return @[
    NSToolbarFlexibleSpaceItemIdentifier,
    NSToolbarSpaceItemIdentifier,
    kMapToolbarItemIdentifier,
    kDebugToolbarItemIdentifier,
    NSToolbarToggleInspectorItemIdentifier,
  ];
}

- (NSToolbarItem *)toolbar:(NSToolbar *)toolbar
        itemForItemIdentifier:(NSToolbarItemIdentifier)itemIdentifier
    willBeInsertedIntoToolbar:(BOOL)willBeInserted {
  (void)toolbar;
  (void)willBeInserted;
  const BOOL isInspector = [itemIdentifier isEqualToString:NSToolbarToggleInspectorItemIdentifier];
  const BOOL isDebug = [itemIdentifier isEqualToString:kDebugToolbarItemIdentifier];
  const BOOL isMap = [itemIdentifier isEqualToString:kMapToolbarItemIdentifier];
  if (!isInspector && !isDebug && !isMap) {
    return nil;
  }

  // Viewless bordered items receive the native toolbar appearance, including
  // Liquid Glass on supported macOS releases. The inspector also uses AppKit's
  // standard semantic identifier even though this delegate instantiates it.
  NSToolbarItem *item = [[NSToolbarItem alloc] initWithItemIdentifier:itemIdentifier];
  item.bordered = YES;
  if (isInspector) {
    item.target = _overlayView;
    item.label = @"Inspector";
    item.paletteLabel = @"Inspector";
    item.toolTip = @"Show or hide the render settings inspector";
    item.image = [NSImage imageWithSystemSymbolName:@"sidebar.right"
                           accessibilityDescription:@"Toggle Inspector"];
    item.action = @selector(toggleInspector:);
  } else if (isDebug) {
    item.target = _overlayView;
    item.label = @"Debug Info";
    item.paletteLabel = @"Debug Info";
    item.toolTip = @"Show or hide viewer debugging information";
    item.image = [NSImage imageWithSystemSymbolName:@"info.circle"
                           accessibilityDescription:@"Toggle Debug Info"];
    item.action = @selector(toggleDebugOverlay:);
  } else if (isMap) {
    item.target = _controller;
    item.label = @"Map & Inspect";
    item.paletteLabel = @"Map & Inspect";
    item.toolTip = @"Show the minimap and enable terrain-point inspection";
    item.image = [NSImage imageWithSystemSymbolName:@"map"
                           accessibilityDescription:@"Toggle Map and Point Inspection"];
    item.action = @selector(toggleMapAndPointInspection:);
    if (@available(macOS 26.0, *)) {
      item.style = [_controller isMapAndPointInspectionEnabled] ? NSToolbarItemStyleProminent
                                                                : NSToolbarItemStylePlain;
    }
  }
  return item;
}

- (void)applicationDidFinishLaunching:(NSNotification *)notification {
  (void)notification;
  const panorama::ImageSize image = _renderer->initial_image();
  constexpr CGFloat kInspectorWidth = 300.0;
  constexpr NSSize kDebugSize = {240.0, 430.0};
  const NSRect windowFrame = NSMakeRect(0.0, 0.0, image.width, image.height);
  const NSRect imageFrame = NSMakeRect(0.0, 0.0, image.width, image.height);
  _window = [[NSWindow alloc]
      initWithContentRect:windowFrame
                styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                          NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable |
                          NSWindowStyleMaskFullSizeContentView
                  backing:NSBackingStoreBuffered
                    defer:NO];
  _window.title = @"panorama-app — drag or use WASD/arrow keys to look around; scroll to zoom";

  PanoramaView *view = [[PanoramaView alloc] initWithFrame:imageFrame device:_renderer->device()];
  view.colorPixelFormat = MTLPixelFormatBGRA8Unorm;
  view.framebufferOnly = NO;
  // The full-screen pass samples any completed frame into this native backing
  // drawable. During live resize no tracing or MetalFX resources are rebuilt.
  view.autoResizeDrawable = YES;
  view.preferredFramesPerSecond = 30;
  view.clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0);
  _controller = [[PanoramaController alloc] initWithRenderer:_renderer.get() window:_window];

  // Keep the traced ray field and Metal drawable at the same aspect ratio when
  // the inspector, window, or requested resolution changes; otherwise AppKit
  // scales the drawable non-uniformly and distorts terrain.
  const CGFloat imageAspect =
      static_cast<CGFloat>(image.width) / static_cast<CGFloat>(image.height);
  AspectFitContainerView *imageContainer =
      [[AspectFitContainerView alloc] initWithFrame:imageFrame
                                         renderView:view
                                        aspectRatio:imageAspect];

  NSViewController *settingsController = [_controller makeSettingsViewController];
  settingsController.title = @"Viewer";
  NSViewController *positioningController = [_controller makePositioningViewController];
  positioningController.title = @"Position";
  _inspectorController = [[NSTabViewController alloc] init];
  _inspectorController.tabStyle = NSTabViewControllerTabStyleSegmentedControlOnTop;
  [_inspectorController addChildViewController:settingsController];
  [_inspectorController addChildViewController:positioningController];
  NSViewController *debugController = [_controller makeDebugViewController];
  NSViewController *pointInfoController = [_controller makePointInfoViewController];
  const panorama::ObserverLocation observer = _renderer->observer();
  MiniMapPanelView *miniMapPanel =
      [[MiniMapPanelView alloc] initWithObserverEasting:observer.easting
                                               northing:observer.northing
                                        terrainEpsgCode:_renderer->terrain_crs().epsg_code()
                                        terrainCoverage:_renderer->terrain_coverage()
                               coverageInitiallyVisible:_renderer->observer_used_fallback()
                                            maxDistance:_renderer->max_distance()
                                          pointInfoView:pointInfoController.view
                                            metalDevice:_renderer->device()
                                           commandQueue:_renderer->command_queue()
                                                library:_renderer->library()];
  _overlayView = [[ViewerOverlayView alloc] initWithFrame:imageFrame
                                              contentView:imageContainer
                                             settingsView:_inspectorController.view
                                           inspectorWidth:kInspectorWidth
                                                debugView:debugController.view
                                                debugSize:kDebugSize
                                             mapPanelView:miniMapPanel];
  [_controller attachPanoramaView:view
                      overlayView:_overlayView
                    aspectFitView:imageContainer
                     miniMapPanel:miniMapPanel];

  NSToolbar *toolbar = [[NSToolbar alloc] initWithIdentifier:@"panorama.toolbar"];
  toolbar.delegate = self;
  toolbar.displayMode = NSToolbarDisplayModeIconOnly;
  toolbar.allowsUserCustomization = NO;
  _window.toolbar = toolbar;
  // Standard AppKit chrome adopts Liquid Glass on current macOS releases.
  // Let the terrain extend beneath it instead of drawing an opaque titlebar
  // background that visually separates the toolbar from the scene.
  _window.toolbarStyle = NSWindowToolbarStyleAutomatic;
  _window.titleVisibility = NSWindowTitleHidden;
  _window.titlebarAppearsTransparent = YES;
  _window.titlebarSeparatorStyle = NSTitlebarSeparatorStyleNone;

  // This source-only AppKit application has no nib to supply its main menu.
  NSMenu *mainMenu = [[NSMenu alloc] initWithTitle:@""];
  NSMenuItem *applicationItem = [[NSMenuItem alloc] initWithTitle:@"" action:nil keyEquivalent:@""];
  [mainMenu addItem:applicationItem];
  NSMenu *applicationMenu = [[NSMenu alloc] initWithTitle:@"panorama-app"];
  [applicationMenu addItem:[[NSMenuItem alloc] initWithTitle:@"Quit panorama-app"
                                                      action:@selector(terminate:)
                                               keyEquivalent:@"q"]];
  applicationItem.submenu = applicationMenu;

  NSMenuItem *raytracerItem = [[NSMenuItem alloc] initWithTitle:@"Raytracer"
                                                         action:nil
                                                  keyEquivalent:@""];
  NSMenu *raytracerMenu = [[NSMenu alloc] initWithTitle:@"Raytracer"];
  for (const auto backend : {panorama::Raytracer::MetalBvh, panorama::Raytracer::Software}) {
    NSMenuItem *item = [[NSMenuItem alloc]
        initWithTitle:backend == panorama::Raytracer::MetalBvh ? @"BVH" : @"Mipmap"
               action:@selector(selectRaytracer:)
        keyEquivalent:@""];
    item.tag = static_cast<NSInteger>(backend);
    item.target = _controller;
    [raytracerMenu addItem:item];
  }
  raytracerItem.submenu = raytracerMenu;
  [mainMenu addItem:raytracerItem];

  NSApp.mainMenu = mainMenu;

  view.panoramaController = _controller;
  view.delegate = _controller;
  NSView *windowContent = _overlayView;
  windowContent.translatesAutoresizingMaskIntoConstraints = YES;
  windowContent.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
  _window.contentView = windowContent;
  [_window makeKeyAndOrderFront:nil];
  // Adding the unified toolbar settles its final content geometry when the
  // window is first shown. Apply the requested render size after that step.
  [_window setContentSize:imageFrame.size];
  [_window center];
  [_window makeFirstResponder:view];
  [NSApp activateIgnoringOtherApps:YES];
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)sender {
  (void)sender;
  return YES;
}

@end

int main(int argc, const char *argv[]) {
  try {
    @autoreleasepool {
      panorama::app::ViewerSettings settings = panorama::app::parse_arguments(argc, argv);
      panorama::app::diagnostics::enabled = settings.trace_diagnostics;
      NSApplication *application = NSApplication.sharedApplication;
      application.activationPolicy = NSApplicationActivationPolicyRegular;
      PanoramaAppDelegate *delegate =
          [[PanoramaAppDelegate alloc] initWithSettings:std::move(settings)];
      application.delegate = delegate;
      [application run];
    }
    return EXIT_SUCCESS;
  } catch (const std::exception &exception) {
    std::fprintf(stderr, "%s\n", exception.what());
    return EXIT_FAILURE;
  }
}

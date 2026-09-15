#include "location_search_control.h"
#include "location_search.h"

#import <MapKit/MapKit.h>

#include <algorithm>
#include <stdexcept>

using namespace panorama;
using namespace panorama::app;

@interface LocationSearchToolbarView : NSView
@end

@implementation LocationSearchToolbarView
- (NSSize)intrinsicContentSize {
  return NSMakeSize(300.0, 30.0);
}
@end

/// Keep the field editor active while clicking a suggestion.
@interface LocationSuggestionsTable : NSTableView
@end
@implementation LocationSuggestionsTable
- (BOOL)acceptsFirstResponder {
  return NO;
}
@end

@interface LocationSearchControl () <
    NSSearchFieldDelegate,
    MKLocalSearchCompleterDelegate,
    NSTableViewDataSource,
    NSTableViewDelegate> {
  ViewerRenderer *_renderer;
  __weak PanoramaController *_controller;
  __weak NSWindow *_window;
  __weak NSResponder *_previousResponder;
  LocationSearchToolbarView *_toolbarView;
  NSToolbarItem *_toolbarItem;
  NSButton *_closeButton;
  NSSearchField *_field;
  NSView *_fieldPanel;
  NSVisualEffectView *_suggestions;
  NSScrollView *_scrollView;
  NSTableView *_table;
  id _dismissSearchMonitor;
  NSTextField *_status;
  MKLocalSearchCompleter *_completer;
  MKLocalSearch *_search;
  NSArray<MKLocalSearchCompletion *> *_appleMatches;
  std::vector<uint32_t> _peakMatches;
  uint64_t _queryToken;
  BOOL _expanded;
  BOOL _editing;
  BOOL _submitting;
}
@end

@implementation LocationSearchControl

- (instancetype)initWithRenderer:(ViewerRenderer *)renderer
                      controller:(PanoramaController *)controller
                          window:(NSWindow *)window {
  self = [super init];
  if (self != nil) {
    _renderer = renderer;
    _controller = controller;
    _window = window;
    _appleMatches = @[];
    _toolbarView = [[LocationSearchToolbarView alloc] initWithFrame:NSMakeRect(0, 0, 300, 30)];
    _toolbarView.translatesAutoresizingMaskIntoConstraints = NO;
    [_toolbarView.widthAnchor constraintEqualToConstant:300].active = YES;
    [_toolbarView.heightAnchor constraintEqualToConstant:30].active = YES;

    _field = [[NSSearchField alloc] initWithFrame:NSMakeRect(0, 0, 300, 30)];
    _field.placeholderString = @"Coordinates, peak, or place";
    _field.accessibilityLabel = @"Search locations";
    _field.delegate = self;
    _field.target = self;
    _field.action = @selector(submitSearch:);
    _field.sendsWholeSearchString = YES;
    _field.sendsSearchStringImmediately = NO;
    ((NSSearchFieldCell *)_field.cell).cancelButtonCell = nil;
    _field.hidden = YES;
    // Match the background to the field itself, rather than the whole toolbar
    // item (which also contains the close button). AppKit suppresses a search
    // field's own background when it is hosted in a toolbar.
    if (@available(macOS 26.0, *)) {
      NSGlassEffectView *glass =
          [[NSGlassEffectView alloc] initWithFrame:NSMakeRect(0, 0, 274, 30)];
      glass.style = NSGlassEffectViewStyleRegular;
      glass.cornerRadius = 15;
      glass.contentView = _field;
      _fieldPanel = glass;
    } else {
      NSVisualEffectView *material =
          [[NSVisualEffectView alloc] initWithFrame:NSMakeRect(0, 0, 274, 30)];
      material.material = NSVisualEffectMaterialHeaderView;
      material.blendingMode = NSVisualEffectBlendingModeWithinWindow;
      material.state = NSVisualEffectStateActive;
      material.wantsLayer = YES;
      material.layer.cornerRadius = 15;
      material.layer.masksToBounds = YES;
      [material addSubview:_field];
      _fieldPanel = material;
    }
    _field.frame = _fieldPanel.bounds;
    _field.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    _fieldPanel.translatesAutoresizingMaskIntoConstraints = NO;
    [_toolbarView addSubview:_fieldPanel];
    _closeButton = [NSButton buttonWithImage:[NSImage imageWithSystemSymbolName:@"xmark"
                                                       accessibilityDescription:@"Close search"]
                                      target:self
                                      action:@selector(closeSearch:)];
    _closeButton.bordered = NO;
    _closeButton.toolTip = @"Clear and close search";
    _closeButton.translatesAutoresizingMaskIntoConstraints = NO;
    [_toolbarView addSubview:_closeButton];
    [NSLayoutConstraint activateConstraints:@[
      [_fieldPanel.leadingAnchor constraintEqualToAnchor:_toolbarView.leadingAnchor],
      [_fieldPanel.topAnchor constraintEqualToAnchor:_toolbarView.topAnchor],
      [_fieldPanel.bottomAnchor constraintEqualToAnchor:_toolbarView.bottomAnchor],
      [_fieldPanel.trailingAnchor constraintEqualToAnchor:_closeButton.leadingAnchor],
      [_closeButton.widthAnchor constraintEqualToConstant:26],
      [_closeButton.trailingAnchor constraintEqualToAnchor:_toolbarView.trailingAnchor],
      [_closeButton.topAnchor constraintEqualToAnchor:_toolbarView.topAnchor],
      [_closeButton.bottomAnchor constraintEqualToAnchor:_toolbarView.bottomAnchor],
    ]];

    _suggestions = [[NSVisualEffectView alloc] initWithFrame:NSZeroRect];
    _suggestions.material = NSVisualEffectMaterialPopover;
    _suggestions.blendingMode = NSVisualEffectBlendingModeWithinWindow;
    _suggestions.state = NSVisualEffectStateActive;
    _suggestions.wantsLayer = YES;
    _suggestions.layer.cornerRadius = 12;
    _suggestions.layer.masksToBounds = YES;
    _suggestions.hidden = YES;
    _status = [NSTextField wrappingLabelWithString:@""];
    _status.font = [NSFont systemFontOfSize:NSFont.smallSystemFontSize];
    [_suggestions addSubview:_status];
    _table = [[LocationSuggestionsTable alloc] initWithFrame:NSZeroRect];
    [_table addTableColumn:[[NSTableColumn alloc] initWithIdentifier:@"location"]];
    _table.headerView = nil;
    _table.rowHeight = 44;
    _table.intercellSpacing = NSMakeSize(0, 0);
    _table.backgroundColor = NSColor.clearColor;
    _table.allowsEmptySelection = YES;
    _table.dataSource = self;
    _table.delegate = self;
    _table.target = self;
    _table.action = @selector(selectSuggestion:);
    _table.accessibilityLabel = @"Location suggestions";
    _scrollView = [[NSScrollView alloc] initWithFrame:NSZeroRect];
    _scrollView.drawsBackground = NO;
    _scrollView.hasVerticalScroller = YES;
    _scrollView.documentView = _table;
    [_suggestions addSubview:_scrollView];
    [NSNotificationCenter.defaultCenter addObserver:self
                                           selector:@selector(windowLayoutChanged:)
                                               name:NSWindowDidResizeNotification
                                             object:window];
    __weak LocationSearchControl *weakSelf = self;
    _dismissSearchMonitor = [NSEvent
        addLocalMonitorForEventsMatchingMask:NSEventMaskLeftMouseDown | NSEventMaskRightMouseDown
                                     handler:^NSEvent *(NSEvent *event) {
                                       LocationSearchControl *control = weakSelf;
                                       if (control == nil || !control->_expanded ||
                                           event.window != control->_window) {
                                         return event;
                                       }
                                       const NSPoint fieldPoint = [control->_toolbarView
                                           convertPoint:event.locationInWindow
                                               fromView:nil];
                                       if (NSPointInRect(
                                               fieldPoint,
                                               control->_toolbarView.bounds
                                           )) {
                                         return event;
                                       }
                                       const NSPoint suggestionsPoint = [control->_suggestions
                                           convertPoint:event.locationInWindow
                                               fromView:nil];
                                       if (!control->_suggestions.hidden &&
                                           NSPointInRect(
                                               suggestionsPoint,
                                               control->_suggestions.bounds
                                           )) {
                                         return event;
                                       }
                                       [control setExpanded:NO];
                                       return event;
                                     }];
    [NSNotificationCenter.defaultCenter addObserver:self
                                           selector:@selector(windowDeactivated:)
                                               name:NSWindowDidResignKeyNotification
                                             object:window];
  }
  return self;
}

- (void)dealloc {
  if (_dismissSearchMonitor != nil) {
    [NSEvent removeMonitor:_dismissSearchMonitor];
  }
  [NSNotificationCenter.defaultCenter removeObserver:self];
  _completer.delegate = nil;
  [_completer cancel];
  [_search cancel];
  [_suggestions removeFromSuperview];
}

- (NSToolbarItem *)makeToolbarItemWithIdentifier:(NSToolbarItemIdentifier)identifier {
  NSToolbarItem *item = [[NSToolbarItem alloc] initWithItemIdentifier:identifier];
  item.label = @"Search";
  item.paletteLabel = @"Search Locations";
  item.toolTip = @"Search coordinates, peaks, or places";
  _toolbarItem = item;
  [self configureSearchButton];
  return item;
}

- (void)configureSearchButton {
  _toolbarItem.image = [NSImage imageWithSystemSymbolName:@"magnifyingglass"
                                 accessibilityDescription:@"Search locations"];
  _toolbarItem.target = self;
  _toolbarItem.action = @selector(toggleSearch:);
  _toolbarItem.bordered = YES;
}

- (void)restoreCollapsedAppearance {
  _expanded = NO;
  _field.hidden = YES;
  _toolbarItem.view = nil;
  [self configureSearchButton];
}

- (void)cancelRequests {
  ++_queryToken;
  _completer.delegate = nil;
  [_completer cancel];
  _completer = nil;
  [_search cancel];
  _search = nil;
  _submitting = NO;
}

- (void)setExpanded:(BOOL)expanded {
  if (_expanded == expanded) {
    return;
  }
  if (expanded && !_expanded) {
    _previousResponder = _window.firstResponder;
    // AppKit shares one field editor between controls. Restore the original
    // control when search was opened while another text field had focus.
    if ([_previousResponder isKindOfClass:NSTextView.class]) {
      id delegate = ((NSTextView *)_previousResponder).delegate;
      if ([delegate isKindOfClass:NSResponder.class]) {
        _previousResponder = delegate;
      }
    }
  }
  _expanded = expanded;
  if (expanded) {
    _field.hidden = NO;
    _toolbarItem.bordered = NO;
    _toolbarItem.view = _toolbarView;
    // Let AppKit install and size the custom item before creating its field
    // editor and focus ring. The field's constraints then own every resize.
    __weak LocationSearchControl *weakSelf = self;
    dispatch_async(dispatch_get_main_queue(), ^{
      LocationSearchControl *control = weakSelf;
      if (control == nil || !control->_expanded) {
        return;
      }
      [control->_window.contentView.superview layoutSubtreeIfNeeded];
      [control->_toolbarView layoutSubtreeIfNeeded];
      [control->_window makeFirstResponder:control->_field];
      control->_editing = YES;
      [control updateSuggestions];
    });
  } else {
    _editing = NO;
    [self cancelRequests];
    _suggestions.hidden = YES;
    // End text editing before detaching the field so its editor and focus
    // ring cannot remain in the toolbar after the native button returns.
    [_window makeFirstResponder:_previousResponder];
    [_window endEditingFor:_field];
    [self restoreCollapsedAppearance];
  }
}

- (void)toggleSearch:(id)sender {
  (void)sender;
  [self setExpanded:!_expanded];
}

- (NSString *)query {
  return [_field.stringValue
      stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceAndNewlineCharacterSet];
}

- (MKCoordinateRegion)searchRegion {
  const auto observer = _renderer->observer().position;
  return MKCoordinateRegionMakeWithDistance(
      CLLocationCoordinate2DMake(observer.lat, observer.lon),
      200000,
      200000
  );
}

- (void)windowLayoutChanged:(NSNotification *)notification {
  (void)notification;
  // Toolbar layout follows the window notification.
  __weak LocationSearchControl *weakSelf = self;
  dispatch_async(dispatch_get_main_queue(), ^{
    [weakSelf layoutSuggestions];
  });
}

- (void)windowDeactivated:(NSNotification *)notification {
  (void)notification;
  _suggestions.hidden = YES;
}

- (void)layoutSuggestions {
  if (!_expanded || !_editing || _field.window == nil) {
    return;
  }
  NSView *content = _window.contentView;
  if (_suggestions.superview != content) {
    [content addSubview:_suggestions];
  }
  const NSRect anchor = [content convertRect:_field.bounds fromView:_field];
  const CGFloat width = std::min(380.0, std::max(0.0, content.bounds.size.width - 24.0));
  const CGFloat rowsHeight =
      std::min(8.0, static_cast<double>([self numberOfRowsInTableView:_table])) * 44;
  const CGFloat height = std::min(rowsHeight + 38, std::max(0.0, NSMinY(anchor) - 12));
  const CGFloat x =
      std::clamp(NSMinX(anchor), 12.0, std::max(12.0, NSMaxX(content.bounds) - width - 12));
  _suggestions.frame = NSMakeRect(x, NSMinY(anchor) - height - 4, width, height);
  _status.frame = NSMakeRect(12, height - 30, width - 24, 24);
  _scrollView.frame = NSMakeRect(4, 4, width - 8, std::max(0.0, height - 38));
  _table.tableColumns.firstObject.width = width - 8;
  _suggestions.hidden = self.query.length == 0;
}

- (void)setStatus:(NSString *)status error:(BOOL)error {
  _status.stringValue = status;
  _status.textColor = error ? NSColor.systemRedColor : NSColor.secondaryLabelColor;
  _field.toolTip = status;
  [self layoutSuggestions];
}

- (void)controlTextDidBeginEditing:(NSNotification *)notification {
  (void)notification;
  _editing = YES;
  [self updateSuggestions];
}

- (void)controlTextDidEndEditing:(NSNotification *)notification {
  (void)notification;
  _editing = NO;
  _suggestions.hidden = YES;
  [self cancelRequests];
  // Finish AppKit's responder change before detaching the field, and leave
  // keyboard focus on the control the user selected.
  __weak LocationSearchControl *weakSelf = self;
  dispatch_async(dispatch_get_main_queue(), ^{
    LocationSearchControl *control = weakSelf;
    if (control != nil && control->_expanded && !control->_editing) {
      [control restoreCollapsedAppearance];
    }
  });
}

- (void)controlTextDidChange:(NSNotification *)notification {
  (void)notification;
  [self updateSuggestions];
}

- (void)closeSearch:(id)sender {
  (void)sender;
  _field.stringValue = @"";
  [self setExpanded:NO];
}

- (void)updateSuggestions {
  [self cancelRequests];
  _peakMatches.clear();
  _appleMatches = @[];
  [_table deselectAll:nil];
  [_table reloadData];
  NSString *query = self.query;
  if (query.length == 0) {
    _suggestions.hidden = YES;
    return;
  }
  if (is_coordinate_search(query.UTF8String)) {
    try {
      const auto coordinate = parse_search_coordinate(query.UTF8String, _renderer->terrain_crs());
      [self setStatus:[NSString
                          stringWithFormat:@"%s • Return to move", coordinate.source_name.c_str()]
                error:NO];
    } catch (const std::exception &error) {
      [self setStatus:[NSString stringWithUTF8String:error.what()] error:YES];
    }
    return;
  }
  if (const PeakCatalogue *catalogue = _renderer->peak_catalogue()) {
    _peakMatches = search_peaks(*catalogue, query.UTF8String, _renderer->observer().position);
  }
  [_table reloadData];
  [self setStatus:_peakMatches.empty() ? @"Searching places…" : @"Peaks • searching places…"
            error:NO];
  // A fresh completer lets identity checks reject callbacks for earlier text.
  _completer = [[MKLocalSearchCompleter alloc] init];
  _completer.delegate = self;
  _completer.region = [self searchRegion];
  _completer.resultTypes =
      MKLocalSearchCompleterResultTypeAddress | MKLocalSearchCompleterResultTypePointOfInterest;
  if (@available(macOS 15.0, *)) {
    _completer.resultTypes |= MKLocalSearchCompleterResultTypePhysicalFeature;
  }
  _completer.queryFragment = query;
}

- (void)completerDidUpdateResults:(MKLocalSearchCompleter *)completer {
  if (completer != _completer || !_editing || _submitting) {
    return;
  }
  _appleMatches = [completer.results
      subarrayWithRange:NSMakeRange(0, std::min(completer.results.count, NSUInteger(8)))];
  [_table reloadData];
  [self setStatus:(_peakMatches.empty() && _appleMatches.count == 0)
                      ? @"No matches • Return to search places"
                      : @"Choose a location • Return to move"
            error:NO];
}

- (void)completer:(MKLocalSearchCompleter *)completer didFailWithError:(NSError *)error {
  if (completer != _completer || !_editing || _submitting) {
    return;
  }
  [self setStatus:_peakMatches.empty() ? @"Place suggestions unavailable • Return to retry"
                                       : @"Peak matches • place suggestions unavailable"
            error:NO];
  (void)error;
}

- (NSInteger)numberOfRowsInTableView:(NSTableView *)tableView {
  (void)tableView;
  return static_cast<NSInteger>(_peakMatches.size() + _appleMatches.count);
}

- (NSView *)tableView:(NSTableView *)tableView
    viewForTableColumn:(NSTableColumn *)column
                   row:(NSInteger)row {
  (void)tableView;
  NSView *view = [[NSView alloc] initWithFrame:NSMakeRect(0, 0, column.width, 44)];
  NSString *title;
  NSString *detail;
  if (static_cast<size_t>(row) < _peakMatches.size()) {
    const PeakRecord &peak =
        _renderer->peak_catalogue()->at(_peakMatches[static_cast<size_t>(row)]);
    title = [NSString stringWithUTF8String:peak.name.c_str()];
    detail = [NSString stringWithFormat:@"Peak • %.0f m • %.4f°, %.4f°",
                                        peak.elevation,
                                        peak.position.lat,
                                        peak.position.lon];
  } else {
    MKLocalSearchCompletion *completion =
        _appleMatches[static_cast<NSUInteger>(row) - _peakMatches.size()];
    title = completion.title;
    detail = [@"Apple Maps • " stringByAppendingString:completion.subtitle];
  }
  NSTextField *name = [NSTextField labelWithString:title];
  name.frame = NSMakeRect(8, 23, column.width - 16, 18);
  name.lineBreakMode = NSLineBreakByTruncatingTail;
  NSTextField *subtitle = [NSTextField labelWithString:detail];
  subtitle.font = [NSFont systemFontOfSize:NSFont.smallSystemFontSize];
  subtitle.textColor = NSColor.secondaryLabelColor;
  subtitle.lineBreakMode = NSLineBreakByTruncatingTail;
  subtitle.frame = NSMakeRect(8, 4, column.width - 16, 16);
  [view addSubview:name];
  [view addSubview:subtitle];
  return view;
}

- (BOOL)control:(NSControl *)control
               textView:(NSTextView *)textView
    doCommandBySelector:(SEL)command {
  (void)control;
  (void)textView;
  if (command == @selector(cancelOperation:)) {
    [self setExpanded:NO];
    return YES;
  }
  if (command == @selector(insertNewline:)) {
    [self submitSearch:nil];
    return YES;
  }
  if (command == @selector(moveDown:) || command == @selector(moveUp:)) {
    const NSInteger count = [self numberOfRowsInTableView:_table];
    if (count > 0) {
      const NSInteger row = _table.selectedRow;
      const NSInteger next = row < 0 ? (command == @selector(moveDown:) ? 0 : count - 1)
                                     : std::clamp(
                                           row + (command == @selector(moveDown:) ? 1 : -1),
                                           NSInteger(0),
                                           count - 1
                                       );
      [_table selectRowIndexes:[NSIndexSet indexSetWithIndex:static_cast<NSUInteger>(next)]
          byExtendingSelection:NO];
      [_table scrollRowToVisible:next];
    }
    return YES;
  }
  return NO;
}

- (void)selectSuggestion:(id)sender {
  (void)sender;
  if (_table.selectedRow >= 0) {
    [self submitSearch:nil];
  }
}

- (void)moveToLocation:(LatLon)location token:(uint64_t)token {
  [self moveToLocation:location snapToSummit:NO token:token];
}

- (void)moveToLocation:(LatLon)location snapToSummit:(BOOL)snapToSummit token:(uint64_t)token {
  [self setStatus:snapToSummit ? @"Finding summit within 100 m…" : @"Locating terrain…" error:NO];
  __weak LocationSearchControl *weakSelf = self;
  [_controller moveObserverToLocation:location
                         snapToSummit:snapToSummit
                           completion:^(NSString *error) {
                             LocationSearchControl *control = weakSelf;
                             if (control == nil || control->_queryToken != token) {
                               return;
                             }
                             control->_submitting = NO;
                             if (error != nil) {
                               [control setStatus:error error:YES];
                             } else {
                               [control setExpanded:NO];
                             }
                           }];
}

- (void)submitSearch:(id)sender {
  (void)sender;
  NSString *query = self.query;
  if (query.length == 0) {
    [self setExpanded:NO];
    return;
  }
  if (_submitting) {
    return;
  }
  [self cancelRequests];
  _submitting = YES;
  const uint64_t token = _queryToken;
  if (is_coordinate_search(query.UTF8String)) {
    try {
      const auto coordinate = parse_search_coordinate(query.UTF8String, _renderer->terrain_crs());
      [self moveToLocation:coordinate.geographic token:token];
    } catch (const std::exception &error) {
      _submitting = NO;
      [self setStatus:[NSString stringWithUTF8String:error.what()] error:YES];
    }
    return;
  }
  NSInteger row = _table.selectedRow;
  // Submitting free text follows peak lookup before falling back to Apple.
  if (row < 0 && !_peakMatches.empty()) {
    row = 0;
  }
  if (row >= 0 && static_cast<size_t>(row) < _peakMatches.size()) {
    const PeakRecord &peak =
        _renderer->peak_catalogue()->at(_peakMatches[static_cast<size_t>(row)]);
    _field.stringValue = [NSString stringWithUTF8String:peak.name.c_str()];
    [self moveToLocation:peak.position snapToSummit:YES token:token];
    return;
  }
  MKLocalSearchRequest *request;
  if (row >= 0 && static_cast<size_t>(row) - _peakMatches.size() < _appleMatches.count) {
    MKLocalSearchCompletion *completion =
        _appleMatches[static_cast<size_t>(row) - _peakMatches.size()];
    request = [[MKLocalSearchRequest alloc] initWithCompletion:completion];
  } else {
    request = [[MKLocalSearchRequest alloc] initWithNaturalLanguageQuery:query];
  }
  request.region = [self searchRegion];
  _search = [[MKLocalSearch alloc] initWithRequest:request];
  [self setStatus:@"Searching places…" error:NO];
  __weak LocationSearchControl *weakSelf = self;
  [_search startWithCompletionHandler:^(MKLocalSearchResponse *response, NSError *error) {
    dispatch_async(dispatch_get_main_queue(), ^{
      LocationSearchControl *control = weakSelf;
      if (control == nil || control->_queryToken != token) {
        return;
      }
      control->_search = nil;
      MKMapItem *item = response.mapItems.firstObject;
      if (error != nil || item == nil) {
        control->_submitting = NO;
        [control setStatus:error != nil ? @"Place search failed • try again" : @"No locations found"
                     error:YES];
        return;
      }
      CLLocationCoordinate2D coordinate;
      if (@available(macOS 26.0, *)) {
        coordinate = item.location.coordinate;
      } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        coordinate = item.placemark.coordinate;
#pragma clang diagnostic pop
      }
      [control moveToLocation:{coordinate.latitude, coordinate.longitude} token:token];
    });
  }];
}
@end

#include "panorama_controller_private.h"

#include "solar_position.h"
#include "timer.h"
#include "trace_diagnostics.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

@implementation PanoramaController (Lighting)

- (void)setDaylightStatus:(NSString *)status {
  _daylightTimesLabel.stringValue = status;
  _daylightSymbolsRow.hidden = YES;
}

- (void)resolveObserverTimeZone {
  const uint64_t requestToken = ++_timeZoneRequestToken;
  _observerTimeZone = nil;
  _timeZoneLookupInProgress = true;
  [self setDaylightStatus:@"Finding observer time zone…"];
  [self updateSettingsControlAvailability];

  [_timeZoneRequest cancel];
  const panorama::LatLon geographic =
      _renderer->terrain_crs().to_lat_lon({_observer.easting, _observer.northing});
  CLLocation *location = [[CLLocation alloc] initWithLatitude:geographic.lat
                                                    longitude:geographic.lon];
  _timeZoneRequest = [[MKReverseGeocodingRequest alloc] initWithLocation:location];
  __weak PanoramaController *weakSelf = self;
  [_timeZoneRequest
      getMapItemsWithCompletionHandler:^(NSArray<MKMapItem *> *mapItems, NSError *error) {
        // Geocoding may finish after another observer move. Marshal UI
        // work to the main queue and discard superseded responses.
        dispatch_async(dispatch_get_main_queue(), ^{
          PanoramaController *strongSelf = weakSelf;
          if (strongSelf == nil || requestToken != strongSelf->_timeZoneRequestToken) {
            return;
          }
          strongSelf->_timeZoneLookupInProgress = false;
          NSTimeZone *timeZone = mapItems.firstObject.timeZone;
          if (error != nil || timeZone == nil) {
            [strongSelf setDaylightStatus:@"Observer time zone unavailable"];
            [strongSelf updateSettingsControlAvailability];
            return;
          }

          strongSelf->_observerTimeZone = timeZone;
          strongSelf->_astronomicalTimeControl.toolTip = [NSString
              stringWithFormat:@"Local time in %@ at one-minute resolution", timeZone.name];
          strongSelf->_daylightTimesLabel.toolTip =
              [NSString stringWithFormat:@"Local geometric-horizon crossings in %@", timeZone.name];

          // Populate the initial controls with the current civil time at
          // the observer. Later observer moves preserve the user's chosen
          // wall-clock date and time, but reinterpret them at the new site.
          if (!strongSelf->_astronomicalControlsUseObserverTime) {
            NSDate *now = [NSDate date];
            NSDateFormatter *dateFormatter = [[NSDateFormatter alloc] init];
            dateFormatter.locale = [[NSLocale alloc] initWithLocaleIdentifier:@"en_US_POSIX"];
            dateFormatter.timeZone = timeZone;
            dateFormatter.dateFormat = @"dd-MM-yyyy";
            strongSelf->_astronomicalDateControl.stringValue = [dateFormatter stringFromDate:now];

            NSCalendar *calendar =
                [[NSCalendar alloc] initWithCalendarIdentifier:NSCalendarIdentifierGregorian];
            calendar.timeZone = timeZone;
            NSDateComponents *components =
                [calendar components:NSCalendarUnitHour | NSCalendarUnitMinute fromDate:now];
            const double minutes = static_cast<double>(components.hour * 60 + components.minute);
            strongSelf->_astronomicalTimeControl.doubleValue = minutes;
            strongSelf->_astronomicalTimeLabel.stringValue =
                panorama::app::format_clock_minutes(minutes);
            strongSelf->_astronomicalControlsUseObserverTime = true;
          }

          [strongSelf updateSettingsControlAvailability];
          if (strongSelf->_sunModeControl.selectedSegment == 1) {
            [strongSelf publishAstronomicalLighting];
          } else {
            [strongSelf setDaylightStatus:[NSString stringWithFormat:@"Observer time · %@",
                                                                     timeZone.abbreviation]];
          }
        });
      }];
}

/// Publish the astronomical direction as grid azimuth and altitude without
/// retracing the terrain.
- (BOOL)publishAstronomicalLighting {
  if (_observerTimeZone == nil) {
    [self setDaylightStatus:_timeZoneLookupInProgress ? @"Finding observer time zone…"
                                                      : @"Observer time zone unavailable"];
    return NO;
  }

  const char *date = _astronomicalDateControl.stringValue.UTF8String;
  NSString *timeValue = panorama::app::format_clock_minutes(_astronomicalTimeControl.doubleValue);
  const char *time = timeValue.UTF8String;
  const std::optional<panorama::app::CalendarDateTime> local = panorama::app::parse_date_time(
      date == nullptr ? std::string_view{} : std::string_view(date),
      time == nullptr ? std::string_view{} : std::string_view(time)
  );
  _astronomicalDateControl.textColor =
      local.has_value() ? NSColor.labelColor : NSColor.systemRedColor;
  if (!local.has_value()) {
    [self setDaylightStatus:@"Enter a valid DD-MM-YYYY date"];
    return NO;
  }
  const std::optional<panorama::app::CalendarDateTime> utc =
      panorama::app::local_date_time_to_utc(*local, _observerTimeZone);
  if (!utc.has_value()) {
    [self setDaylightStatus:@"This local time does not exist"];
    return NO;
  }

  const panorama::app::DaylightTimes daylightInfo = panorama::app::daylight_times(
      _renderer->terrain_crs(),
      {_observer.easting, _observer.northing},
      *local
  );
  NSString *timeZoneSummary =
      panorama::app::format_time_zone_summary(_observerTimeZone, utc.value());
  switch (daylightInfo.state) {
  case panorama::app::DaylightState::Normal:
    _daylightTimesLabel.stringValue = timeZoneSummary;
    _sunriseTimeLabel.stringValue = panorama::app::format_local_daylight_time(
        *local,
        daylightInfo.sunrise_minutes,
        _observerTimeZone
    );
    _sunsetTimeLabel.stringValue = panorama::app::format_local_daylight_time(
        *local,
        daylightInfo.sunset_minutes,
        _observerTimeZone
    );
    _daylightSymbolsRow.hidden = NO;
    break;
  case panorama::app::DaylightState::PolarDay:
    [self setDaylightStatus:[NSString stringWithFormat:@"%@\nSun above horizon all day",
                                                       timeZoneSummary]];
    break;
  case panorama::app::DaylightState::PolarNight:
    [self setDaylightStatus:[NSString stringWithFormat:@"%@\nSun below horizon all day",
                                                       timeZoneSummary]];
    break;
  }

  const panorama::app::SolarPosition sun = panorama::app::solar_position(
      _renderer->terrain_crs(),
      {_observer.easting, _observer.northing},
      utc.value()
  );
  const double azimuthDegrees = sun.azimuth * panorama::app::kRadiansToDegrees;
  const double altitudeDegrees = sun.elevation * panorama::app::kRadiansToDegrees;
  _sunAzimuthControl.doubleValue = azimuthDegrees;
  _sunAltitudeControl.doubleValue = altitudeDegrees;
  _sunAzimuthLabel.stringValue = [NSString stringWithFormat:@"%.1f°", azimuthDegrees];
  _sunAltitudeLabel.stringValue = [NSString stringWithFormat:@"%.1f°", altitudeDegrees];
  _presentation.appearance.sun_azimuth = sun.azimuth;
  _presentation.appearance.sun_elevation = sun.elevation;
  _renderer->request_presentation(_presentation);
  return YES;
}

- (void)publishLightingControls {
  _presentation.appearance.sun_azimuth =
      _sunAzimuthControl.doubleValue * panorama::app::kDegreesToRadians;
  _presentation.appearance.sun_elevation =
      _sunAltitudeControl.doubleValue * panorama::app::kDegreesToRadians;
  _presentation.appearance.diffusivity = static_cast<float>(_diffusivityControl.doubleValue);
  _presentation.appearance.ambient_light = static_cast<float>(_skyStrengthControl.doubleValue);
  _presentation.appearance.ambient_detail = static_cast<float>(_skyDetailControl.doubleValue);
  _renderer->request_presentation(_presentation);
}

- (void)sunModeChanged:(NSSegmentedControl *)sender {
  [self updateSettingsControlAvailability];
  if (sender.selectedSegment == 1) {
    _manualSunAzimuthDegrees = _sunAzimuthControl.doubleValue;
    _manualSunAltitudeDegrees = _sunAltitudeControl.doubleValue;
    if (_observerTimeZone == nil) {
      if (!_timeZoneLookupInProgress) {
        [self resolveObserverTimeZone];
      }
    } else if (![self publishAstronomicalLighting]) {
      NSBeep();
    }
  } else {
    _sunAzimuthControl.doubleValue = _manualSunAzimuthDegrees;
    _sunAltitudeControl.doubleValue = _manualSunAltitudeDegrees;
    _sunAzimuthLabel.stringValue = [NSString stringWithFormat:@"%.0f°", _manualSunAzimuthDegrees];
    _sunAltitudeLabel.stringValue = [NSString stringWithFormat:@"%.0f°", _manualSunAltitudeDegrees];
    [self publishLightingControls];
  }
}

- (void)astronomicalInputChanged:(NSTextField *)sender {
  (void)sender;
  if (_sunModeControl.selectedSegment == 1 && ![self publishAstronomicalLighting]) {
    NSBeep();
  }
}

- (void)astronomicalTimeChanged:(NSSlider *)sender {
  sender.doubleValue = std::round(sender.doubleValue);
  _astronomicalTimeLabel.stringValue = panorama::app::format_clock_minutes(sender.doubleValue);
  [self updateSettingsControlAvailability];
  if (_sunModeControl.selectedSegment == 1) {
    [self publishAstronomicalLighting];
  }
}

- (void)adjustAstronomicalTime:(NSButton *)sender {
  const double minutes = std::clamp(
      std::round(_astronomicalTimeControl.doubleValue) + static_cast<double>(sender.tag),
      _astronomicalTimeControl.minValue,
      _astronomicalTimeControl.maxValue
  );
  _astronomicalTimeControl.doubleValue = minutes;
  _astronomicalTimeLabel.stringValue = panorama::app::format_clock_minutes(minutes);
  [self updateSettingsControlAvailability];
  if (_sunModeControl.selectedSegment == 1) {
    [self publishAstronomicalLighting];
  }
}

- (void)sunAzimuthChanged:(NSSlider *)sender {
  constexpr double kDetentRadiusDegrees = 3.0;
  if (std::abs(sender.doubleValue - panorama::app::kDefaultSunAzimuthDegrees) <=
      kDetentRadiusDegrees) {
    sender.doubleValue = panorama::app::kDefaultSunAzimuthDegrees;
  }
  _manualSunAzimuthDegrees = sender.doubleValue;
  _sunAzimuthLabel.stringValue = [NSString stringWithFormat:@"%.0f°", sender.doubleValue];
  [self publishLightingControls];
}

- (void)sunAltitudeChanged:(NSSlider *)sender {
  constexpr double kDetentRadiusDegrees = 2.0;
  if (std::abs(sender.doubleValue - panorama::app::kDefaultSunAltitudeDegrees) <=
      kDetentRadiusDegrees) {
    sender.doubleValue = panorama::app::kDefaultSunAltitudeDegrees;
  }
  _manualSunAltitudeDegrees = sender.doubleValue;
  _sunAltitudeLabel.stringValue = [NSString stringWithFormat:@"%.0f°", sender.doubleValue];
  [self publishLightingControls];
}

- (void)diffusivityChanged:(NSSlider *)sender {
  constexpr double kDetentRadius = 0.02;
  if (std::abs(sender.doubleValue - panorama::app::kDefaultDiffusivity) <= kDetentRadius) {
    sender.doubleValue = panorama::app::kDefaultDiffusivity;
  }
  _diffusivityLabel.stringValue = [NSString stringWithFormat:@"%.2f", sender.doubleValue];
  [self publishLightingControls];
}

- (void)skyStrengthChanged:(NSSlider *)sender {
  constexpr double kDetentRadius = 0.02;
  if (std::abs(sender.doubleValue - panorama::app::kDefaultSkyStrength) <= kDetentRadius) {
    sender.doubleValue = panorama::app::kDefaultSkyStrength;
  }
  _skyStrengthLabel.stringValue = [NSString stringWithFormat:@"%.2f", sender.doubleValue];
  [self publishLightingControls];
}

- (void)skyDetailChanged:(NSSlider *)sender {
  constexpr double kDetentRadius = 0.02;
  if (std::abs(sender.doubleValue - panorama::app::kDefaultSkyDetail) <= kDetentRadius) {
    sender.doubleValue = panorama::app::kDefaultSkyDetail;
  }
  _skyDetailLabel.stringValue = [NSString stringWithFormat:@"%.2f", sender.doubleValue];
  [self publishLightingControls];
}

- (void)normalLightingChanged:(NSButton *)sender {
  _presentation.use_surface_normals = sender.state == NSControlStateValueOn;
  [self updateSettingsControlAvailability];
  _renderer->request_presentation(_presentation);
}

- (void)bilinearCollisionChanged:(NSButton *)sender {
  _bilinearCollisions = sender.state == NSControlStateValueOn;
  _renderer->request_collision_settings(_bilinearCollisions, _c1Normals);
}

- (void)c1NormalsChanged:(NSButton *)sender {
  _c1Normals = sender.state == NSControlStateValueOn;
  _renderer->request_collision_settings(_bilinearCollisions, _c1Normals);
}

- (void)raytracedShadowsChanged:(NSButton *)sender {
  _presentation.appearance.raytraced_shadows = sender.state == NSControlStateValueOn;
  _renderer->request_presentation(_presentation);
}

/// Feature outlines are presentation-only, like lighting, so both controls
/// update the current trace immediately.
- (void)publishFeatureOutlineControls {
  _presentation.appearance.feature_outlines =
      _featureOutlinesControl.state == NSControlStateValueOn;
  _presentation.appearance.feature_outline_detail =
      static_cast<float>(_featureOutlineDetailControl.doubleValue / 10.0);
  _renderer->request_presentation(_presentation);
}

- (void)featureOutlinesChanged:(NSButton *)sender {
  (void)sender;
  [self updateSettingsControlAvailability];
  [self publishFeatureOutlineControls];
}

- (void)featureOutlineDetailChanged:(NSSlider *)sender {
  sender.doubleValue = std::round(sender.doubleValue);
  _featureOutlineDetailLabel.stringValue = [NSString stringWithFormat:@"%.0f", sender.doubleValue];
  [self publishFeatureOutlineControls];
}

/// LOD is a trace setting: zero retains LOD 1 everywhere, while positive
/// values permit a tile representation no wider than this pixel-footprint
/// multiplier.
- (void)lodScaleChanged:(NSSlider *)sender {
  const double scale = sender.doubleValue;
  _lodScaleLabel.stringValue = scale == 0.0 ? @"Off" : [NSString stringWithFormat:@"%.1f×", scale];
  _renderer->request_lod_scale(static_cast<float>(scale));
}

@end

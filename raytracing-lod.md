# Level-of-detail terrain ray tracing

## Objective

Terrain far from the observer is sampled much more densely than the rendered
image can show. A grid cell which covers several pixels nearby may cover only a
small fraction of one pixel at long range. Level-of-detail (LOD) tracing should
therefore use progressively coarser terrain to:

- reduce the number of cells visited by camera and shadow rays;
- reduce GPU atlas traffic and memory use;
- avoid loading full-resolution distant terrain from disk; and
- retain full detail near the observer and wherever coarse data is not
  sufficiently accurate.

Distance thresholds such as level 0 to 1 km, level 1 to 2 km, and so on would
be a useful first experiment, but the appropriate thresholds depend on the
source cell size and the angular resolution of the output. Selecting LOD from
the projected size of a terrain cell gives predictable quality at different
render resolutions and fields of view.

## Existing hierarchy is not terrain LOD

The ray tracer already builds a maximum-elevation mipmap for every resident
tile. Each value is an upper bound for a square group of level-0 cells. The DDA
uses this hierarchy to reject large regions which are below the ray, descending
to level 0 only where a collision may occur.

This is a traversal hierarchy, not a coarse representation of the surface. A
maximum value cannot be treated as a coarse collision surface: doing so would
raise valleys to the height of nearby summits, thicken ridges, and create false
occlusion. LOD collision testing instead needs a real terrain surface at every
LOD, normally made from decimated or filtered vertices. The maximum hierarchy
can continue to accelerate traversal over that surface.

To avoid confusion with the existing one-indexed maximum-mipmap levels, this
document uses `LOD 0` for the original DEM resolution, `LOD 1` for twice the
cell spacing, `LOD 2` for four times the spacing, and so on.

## Choosing an LOD

Let:

- `s0` be the LOD-0 cell spacing in metres;
- `theta` be the angular width of a pixel in radians;
- `r` be the range from the camera to the terrain region; and
- `q` be a quality factor specifying how much of a pixel one cell may cover.

LOD `l` has cell spacing:

```text
s(l) = s0 * 2^l
```

The approximate width of one pixel at range `r` is:

```text
pixel_footprint = r * theta
```

The coarsest acceptable level is therefore approximately:

```text
l = floor(log2(q * r * theta / s0))
```

clamped to the available LOD range. A smaller `q` is more conservative. For a
panorama with different horizontal and vertical angular steps, the smaller
projected footprint should be used so that neither axis is undersampled.

Horizontal distance is adequate for an initial implementation. Slant range,
the three-dimensional straight-line distance from the camera to the terrain
point, is better when turning a height error into an angular screen error. The
two are almost identical for distant terrain viewed close to the horizon but
can differ substantially for steep upward or downward views.

LOD should be selected continuously as a ray advances rather than once for an
entire ray. It should also change only at aligned cell boundaries. A small
hysteresis band around each transition can prevent levels oscillating because
of floating-point noise.

## Geometric accuracy

Projected cell size alone limits horizontal sampling, but a coarse surface may
still omit a narrow or prominent peak. Each coarse patch should therefore have
an error bound describing the greatest vertical residual between the coarse
surface and the finer data beneath it. A useful representation contains:

- the coarse surface vertices;
- a conservative minimum residual or lower bound; and
- a conservative maximum residual or upper bound.

For a camera ray, the error can be projected into pixels approximately as:

```text
screen_error_pixels = vertical_error / (slant_range * vertical_pixel_angle)
```

This is an approximation; a more exact implementation can project the error
perpendicular to the viewing direction. If the bound exceeds the chosen screen
error, traversal refines to the next finer LOD. This preserves sharp silhouettes
and high-relief features even when the distance rule alone would choose coarse
terrain.

The bounds also let traversal classify many intervals without reading finer
geometry:

- a ray above the upper bound definitely misses the terrain;
- a ray below the lower bound definitely encounters terrain; and
- a ray between the bounds is ambiguous and must refine.

Final camera collisions should be refined until their projected error is below
the rendering tolerance. An early prototype can omit the bounds and compare
the output against LOD-0 traces, but it should not be treated as a
correctness-preserving implementation.

## GPU traversal

The existing hierarchical DDA can be extended rather than replaced:

1. Compute the permitted collision LOD from the ray's current distance and the
   output pixel angle.
2. Use conservative maximum values to skip empty regions as today.
3. When traversal descends, stop at the permitted LOD rather than always
   descending to LOD 0.
4. Intersect the ray with the bilinear patch from that LOD's actual terrain
   vertices.
5. Refine if the patch's error bound is too large or the intersection remains
   ambiguous.
6. Return the collision position, elevation, normal, selected LOD, and error
   estimate.

Normals should be calculated from the surface used for the final collision.
Coarse normals are generally appropriate for sub-pixel terrain, although
silhouettes and feature outlines may need refinement to avoid visible LOD
transitions.

As a low-risk proof of concept, the current full-resolution vertex atlas could
be sampled at strides of `2^l`. This would test traversal cost and image quality,
but it would not reduce disk traffic or atlas residency because the complete
tile would still have to be loaded.

## Tile generation and disk I/O

Reducing disk reads requires each LOD to be independently loadable. A single
compressed file containing all levels does not help if it must be read and
decompressed in full before one coarse level can be used.

`panorama-tile-gen` should produce, for each terrain region:

- collision vertices for every supported LOD;
- conservative elevation and approximation-error bounds;
- the traversal hierarchy required by that LOD; and
- metadata linking the levels to the same projected extent and registration.

The simplest layout is one file or independently compressed block per
`(tile, LOD)`. The catalogue can retain lightweight per-tile upper bounds so
that the scheduler can reject a tile before loading any payload. The resident
cache key then becomes `(terrain source, LOD)`, and either separate fixed-stride
atlases or a suitable variable allocation is needed for the different payload
sizes.

Independent levels allow a distant ray to request only a small coarse payload.
If it later finds an ambiguous or visually important region, the frontier can
defer that work while the next finer payload is loaded, using the existing
asynchronous preparation and reactivation model.

A more ambitious format could use a spatial quadtree in which coarse parent
tiles cover the extent of several fine tiles. This reduces both bytes and tile
count at long range, but complicates catalogue lookup, seams, cache management,
and cross-level traversal. Per-tile LOD is the better first implementation.

All levels must share samples along tile boundaries. Otherwise adjacent tiles
or different LODs can produce cracks, duplicate intersections, or light leaks.
Generation should derive shared coarse boundary vertices deterministically from
the same source samples. The tracer may also need a transition rule when two
neighbouring regions use different LODs.

## Shadow rays

Shadow tracing must participate in LOD selection. If camera rays use coarse
distant data but every shadow ray requests LOD 0, most of the intended I/O and
cache benefit disappears.

Hard shadows are especially sensitive to missed peaks because one small error
can change visibility from fully lit to fully shadowed. Conservative bounds are
therefore more valuable than simply intersecting a coarse approximation:

- intervals where the sun ray is above the upper terrain bound are definitely
  clear;
- intervals where terrain's lower bound is above the sun ray are definitely
  blocked; and
- ambiguous intervals request finer data.

The starting LOD can be based on the camera-space footprint of the receiving
pixel, then made more conservative for potential blockers close to the
receiver or near a terrain silhouette. This gives shadows a route to remain
correct without routinely loading the finest version of every crossed tile.

## Other consumers

LOD affects more than the primary colour image:

- **Surface shading:** use the final collision LOD's geometric normal.
- **Feature outlines:** depth changes caused only by an LOD transition must not
  become black feature lines. The outline pass can use collision error and LOD
  metadata, or request refinement around candidate discontinuities.
- **Minimap visibility:** coarse collision positions are sufficient when their
  projected map error is smaller than the visibility-point marker.
- **Terrain inspection:** locked or interactively selected points should be
  refined to LOD 0, or to a separately defined measurement tolerance, because
  the user may read their elevation and distance numerically.
- **Tile availability:** missing fine data should not invalidate a result which
  has already met its error tolerance using a coarser level.

## Suggested implementation stages

1. **Measure a software LOD experiment.** Sample aligned vertices from the
   existing full-resolution atlas, stop collision descent at a selected LOD,
   and compare images and timings against LOD 0.
2. **Select LOD by pixel footprint.** Pass horizontal and vertical pixel angles
   to the trace and choose a level as distance changes along each ray.
3. **Add error bounds.** Generate conservative residual bounds, refine
   ambiguous intersections, and report collision error and LOD for validation.
4. **Make LOD payloads independently loadable.** Extend the tile format,
   catalogue, loader, cache key, and GPU atlas organisation.
5. **Integrate frontier scheduling.** Permit a coarse resident tile to enqueue a
   request for a finer version without disturbing unrelated active rays.
6. **Apply LOD to shadow rays.** Use conservative clear/blocked tests and refine
   only ambiguous potential occluders.
7. **Handle presentation consumers.** Prevent outline artifacts and explicitly
   refine inspection queries.
8. **Tune and expose quality.** Choose a conservative default screen-error
   tolerance; only add a user-facing quality control if benchmarks show it is
   useful.

## Validation and benchmarks

Correctness comparisons should trace identical views at forced LOD 0 and with
adaptive LOD. Tests should cover:

- isolated narrow peaks and ridges;
- deep valleys hidden beneath high maximum-mipmap values;
- silhouettes and grazing camera rays;
- seams between tiles and between LODs;
- low-angle sun rays and long-distance shadow casters;
- effective-Earth curvature over long ranges;
- Float32 and quantized terrain;
- missing or deferred finer levels; and
- feature-outline stability.

Record GPU traversal time, cells visited, frontier iterations, bytes read,
tiles or LOD blocks loaded, atlas residency, and cache evictions. The feature
is successful only if it reduces both tracing work and terrain I/O without
introducing visible silhouette, seam, or shadow errors.

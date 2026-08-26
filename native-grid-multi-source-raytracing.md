# Native-grid multi-source terrain ray tracing

## Objective

Allow one render to traverse terrain from several DEM datasets without first
resampling their elevation grids into a common projection. Typical source
stacks include a high-resolution national model near the observer and one or
more lower-resolution national or global models outside its coverage:

```text
SwissALTI3D
    -> neighbouring national DEM
        -> SRTM fallback
```

The design must:

- retain each source's native horizontal grid and elevation samples;
- support projected and geographic CRSs, including latitude/longitude SRTM;
- keep one physical ray continuous when crossing between sources;
- preserve asynchronous tile loading and bounded GPU residency;
- support camera and terrain-shadow rays;
- remain numerically useful to a maximum range of 600 km;
- define deterministic priority and no-data fallback between overlapping
  sources; and
- integrate with independently loadable terrain levels of detail.

The feature deliberately avoids full-tile horizontal reprojection. It does not
avoid coordinate transformation altogether: each native tile needs a compact,
error-bounded mapping between its raster grid and the global ray geometry.

## Non-goals

The first implementation need not:

- blend arbitrary source boundaries perfectly;
- support every GDAL raster layout or CRS operation;
- perform time-varying atmospheric ray bending;
- choose source quality automatically from dataset names;
- preserve the current on-disk tile format unchanged; or
- combine native-grid support and adaptive LOD in one initial patch.

The architecture must leave room for these, particularly boundary blending and
LOD, without requiring another ray-coordinate redesign.

## Current constraints

The current renderer establishes one projected, metre-based terrain grid from
the observer tile. Several assumptions follow from that:

- `ObserverLocation` contains easting, northing, and elevation in the terrain
  CRS.
- `TerrainCatalogue` owns one `TileGrid` and maps one row/column key to one
  source file.
- `AsyncTilePreparer` rejects tiles whose CRS, cell size, dimensions, or mipmap
  layout differ from the observer tile.
- `RaytraceParameters` contains one dispatch-wide cell size and level count.
- `ResidentTile` needs only an observer-relative projected origin, maximum
  elevation, and global grid key.
- the Metal continuation kernels derive the successor row and column from one
  rectangular catalogue grid;
- the resident atlas has one fixed payload stride; and
- headings and sun azimuths are expressed relative to the common projected
  grid north.

Native-grid tracing must move physical geometry out of these dispatch-wide
assumptions and into dataset or resident-tile metadata. The maximum-elevation
mipmap and bilinear heightfield intersection remain useful because both operate
on raster cell indices once a ray has been expressed in that grid.

## Terminology

The implementation should distinguish concepts which are currently all called
sources or tiles:

- **Dataset:** one coherent DEM with a CRS, vertical reference, resolution,
  registration, coverage, and quality priority.
- **Native tile:** one independently loadable rectangular chunk whose vertices
  remain on its dataset's raster grid.
- **Terrain address:** the stable identity `(dataset, tile row, tile column,
  LOD)` used by scheduling and caching.
- **Resident slot:** one GPU atlas allocation containing a native tile payload
  and its runtime spatial metadata.
- **Routing region:** lightweight geographic metadata used to choose a dataset
  and native tile without opening its elevation payload.
- **Transition tile:** an optional, narrowly scoped composite used to reconcile
  two source surfaces near a coverage boundary.

`TerrainSource` currently means one file in one catalogue. Renaming it during
the migration will reduce ambiguity between a dataset and an individual tile.

## Authoritative ray geometry

### Observer and world coordinates

Application-facing observer positions should become geographic, with an
explicit vertical reference:

```text
latitude
longitude
elevation
vertical datum
```

Projected eastings and northings may remain accepted convenience inputs, but
they must be converted before constructing the trace session rather than
defining its world frame.

Use WGS 84 Earth-centred, Earth-fixed (ECEF) geometry as the authoritative
source-independent spatial representation. Store the observer ECEF position in
host `double`. GPU positions are rebased around the observer before conversion
to `float`, so their magnitude is at most approximately the configured range,
not the Earth's 6,400 km radius.

One conceptual ray contains:

```text
origin relative to observer ECEF
direction or propagation tangent
slant-distance parameter
maximum distance
```

Primary rays begin at the observer-relative origin. Shadow rays begin at their
primary collision. This representation does not change when a ray crosses a
dataset boundary.

At 600 km, a `float` distance has centimetre-scale granularity, which is
adequate for distant DEM cells. Catalogue positions, CRS transformations,
transformation fitting, observer rebasing, and validation should remain in
host `double`. Absolute ECEF coordinates must not be uploaded directly as
`float`.

### Curvature and refraction

With terrain located relative to an ellipsoid, terrestrial curvature is part
of the geometry rather than solely a quadratic elevation correction. The
initial native-grid implementation should preserve the current effective-Earth
result for parity, but isolate it behind a propagation model instead of
embedding `coefficient * distance * distance` throughout traversal.

The propagation API should provide a ray position and tangent at distance `t`.
An unrefracted ray is a straight ECEF line. An effective-radius or other
refraction model may be a slightly curved path, divided into locally linear
segments before native-grid traversal.

The existing quadratic correction differs from exact spherical geometry by
tens of metres at 600 km. Before claiming long-range geometric accuracy, add an
exact spherical or ellipsoidal reference test and define how the effective
refraction radius modifies it. Atmospheric uncertainty is unavoidable, but
numerical approximation error should be separately measured.

### Local tile frame

Give each native tile a stable tile-centred east/north/up frame derived from its
geographic centre. Its static metadata describes the mapping:

```text
native raster index (u, v), corrected elevation
    <-> tile-local east, north, up
```

When a tile becomes resident, the host combines its static frame with the
current observer ECEF origin. The GPU can then transform a primary or shadow
ray into the tile-local frame using a rotation and rebased translation. A
straight ECEF ray remains straight under this affine transform.

The raster-to-local mapping is not exactly affine over a finite tile. Store an
adaptive control mesh or low-order patch model and a measured horizontal and
vertical error bound. This metadata transforms only grid geometry; it does not
resample or duplicate the elevation array.

## Native tile preparation and file format

### Source ingestion

`panorama-tile-gen` should gain a native preparation mode. It may rechunk an
input raster along its existing aligned sample grid, but must not horizontally
resample elevation values. Rechunking remains valuable because supplier files
such as one-degree SRTM blocks are too large for fine-grained residency and
their projection mapping may not be locally linear enough.

For each dataset, preparation should:

1. Read and retain its complete horizontal CRS definition, not only the three
   currently enumerated EPSG codes.
2. Record raster registration and the complete native affine transform.
3. Convert elevation units to metres, or record an exact conversion applied by
   the loader.
4. Retain the native no-data/validity mask.
5. Rechunk only at source-grid-aligned sample boundaries.
6. Compute per-tile minimum and maximum elevation bounds.
7. Compute a WGS 84 footprint and conservative geographic bounding region.
8. Construct and validate the native-grid-to-tile-frame control model.
9. Preserve shared boundary samples deterministically.
10. Write dataset and tile manifests without repeating large CRS strings in
    every tile payload.

This work transforms only metadata and a small number of control positions.
It avoids evaluating a horizontal reprojection for every DEM sample and avoids
storing a second elevation raster.

### Dataset manifest

A versioned dataset manifest should include at least:

```text
stable dataset identifier
human-readable name and attribution
horizontal CRS WKT and optional EPSG identifier
vertical datum identifier
elevation unit conversion
native affine raster transform
raster registration
tile cell count and naming scheme
coverage and validity summary
nominal horizontal resolution or accuracy
source priority
available LODs
hierarchical conservative elevation bounds
```

Do not infer priority or vertical compatibility from filenames. A render
configuration should define the ordered dataset stack explicitly, while the
manifest supplies default quality metadata.

### Tile header

The Metal tile header should identify its dataset and native tile key and
describe its payload independently of metre-based square cells. Relevant
fields include:

```text
dataset identifier
tile row and column
LOD
cell count in each axis
vertex representation and offsets
validity-mask representation and offset
minimum and maximum elevation
native-grid extent
control-model reference or coefficients
```

The full CRS belongs in the dataset manifest. A tile may contain or reference
the compact local geometry model needed without GDAL/PROJ at trace time.

Initially require every atlas-compatible native tile to use the same
power-of-two cell count, even though its angular or physical extent differs.
For example, 512 cells may cover about 1 km in a fine national DEM and about
15 km in one-arcsecond SRTM. This retains fixed-stride atlases and a common
maximum-mipmap layout while other assumptions are removed. Atlas pools for
different cell counts can be added later.

## Error-bounded native-grid mapping

### Why an approximation is necessary

Metal cannot apply arbitrary PROJ pipelines, and a geodesic or ECEF ray is not
generally a straight line in a native projected or latitude/longitude raster.
The mapping must therefore be approximated locally without permitting the ray
to skip a possible collision.

For each control patch, fit a mapping between native grid coordinates and the
tile-local frame. Validate it at additional points not used by the fit. Record:

```text
maximum horizontal position error
maximum local-up/base-surface error
maximum Jacobian or direction error
valid native-coordinate domain
```

Tile preparation should subdivide the control mesh until these limits are met
or reject the tile with a useful diagnostic. Thresholds must be expressed
relative to the native cell size as well as in metres.

### Ray segmentation

Within one control patch, approximate the transformed ray as:

```text
u(t) = u0 + du_dt * (t - t0)
v(t) = v0 + dv_dt * (t - t0)
```

Limit the segment to the control patch and to a distance for which the stored
mapping error remains valid. If the propagation model bends the ray, its own
linearisation error contributes to the same bound.

The traversal kernel may process several such segments before leaving a
resident tile. If an implementation instead returns a continuation after every
segment, measure the extra frontier traffic before retaining that design.

### Conservative cell visitation

An approximate centre line is insufficient by itself: however small the error,
it can lie on the wrong side of a cell boundary. Traversal must conservatively
cover a tube whose radius is the transformation and propagation error expressed
in cell coordinates.

Possible implementations, in increasing order of generality, are:

1. refine control patches until the error is below the existing boundary
   tolerance, then explicitly test the neighbouring cell at an ambiguous edge;
2. use a supercover DDA which visits all cells touched by the error tube; or
3. carry lower and upper coordinate bounds through a hierarchical traversal
   and descend where ownership is ambiguous.

The first option is a practical prototype, but correctness tests must include
rays following cell and tile boundaries. A production implementation should
not silently assume that a sampled fit is exact.

### Bilinear collision and normals

Perform the DDA in native cell-index coordinates. The existing bilinear root
solver can be generalised from metre coordinates to the linear segment
`(u(t), v(t), height(t))`; it does not require cells to be physically square.

Maximum-mipmap values must be expressed in the same corrected vertical frame
used for collision testing. Conservative bounds must include:

- vertical-datum correction across the cell or block;
- tile-base ellipsoid or geoid geometry;
- native-grid transform error; and
- ray-propagation linearisation error.

Native elevation differences cannot be used directly as metre-space surface
gradients when the horizontal axes are degrees or have projection scale and
convergence. At the final collision, use the local mapping Jacobian to build
two physical surface tangents and take their cross product. Return the normal
in the common observer/render frame so lighting remains source-independent.

## Multi-source catalogue and routing

### Dataset catalogue

Replace the single-directory `TerrainCatalogue` with two layers:

- `TerrainDatasetCatalogue` owns immutable dataset manifests, priorities,
  coverage, native tile indexes, and hierarchical elevation bounds.
- `TerrainRoute` or an equivalent session object resolves a geographic point
  and requested accuracy to a `TerrainAddress`.

Discovery must be lazy. A 600 km radius covers roughly 1.1 million square
kilometres, so the renderer must not open or flatten every fine source tile
into one vector before tracing. Dataset manifests and spatial bound trees
should answer routing and horizon-culling queries without opening elevation
payloads.

### Priority and validity

Routing chooses the first valid dataset in an explicit ordered stack. Coverage
must distinguish:

- outside the dataset footprint;
- a missing native tile;
- a tile present with internal no-data cells; and
- data present but marked unsuitable by a quality mask.

A lower-priority source should remain available beneath higher-priority data so
it can fill both outer coverage and internal holes. The route decision must be
deterministic at shared boundaries and use the same rule for camera rays,
shadow rays, terrain inspection, and minimap queries.

### Same-dataset continuations

Most boundary crossings remain within one regular native dataset. A resident
tile can carry compact neighbour addresses for its four sides, allowing the GPU
continuation kernel to emit the next `TerrainAddress` without a general CRS
lookup. Corner exits need a deterministic ownership rule matching the current
cell-boundary handling.

### Cross-dataset continuations

When a native neighbour is absent, invalid, or lower priority than a dataset
which begins at the exit position, emit a routing request containing:

```text
ray index
current terrain address
global ray distance at exit
approximate geographic or observer-relative exit position
required accuracy/LOD
```

Initially resolve these uncommon requests on the host using cached PROJ and
coverage indexes, then feed the resulting terrain address into the existing
deferred-work mechanism. If profiling shows source-boundary routing to be
significant, upload a compact geographic routing structure to the GPU.

The global distance remains authoritative. The successor tile derives a fresh
local position and direction from the global ray, preventing transform error
from accumulating across hundreds of tiles.

### Missing coverage

If no dataset covers the exit position, mark the ray as outside known terrain,
not as a sky result immediately. The routing hierarchy should determine
whether the ray can encounter another disjoint coverage region farther away.
Only terminate after proving that no reachable region within maximum distance
can intersect the ray.

## Frontier and cache changes

### Terrain address

Replace finite `source_index` continuations with a stable address. A compact
representation may use a session-local dataset index plus signed row, column,
and LOD. Avoid relying on an index into an eagerly populated vector.

`DeferredRayWork` should contain the address and global entry distance. Host
frontier buckets become an associative collection created lazily per address.
The preparer request queue continues to prioritise the nearest unresolved work.

### Asynchronous preparation

`AsyncTilePreparer` should:

1. resolve a terrain address through its dataset manifest;
2. read the native payload and validity mask;
3. apply or attach vertical-unit and datum corrections;
4. validate the native grid and local geometry model;
5. prepare the maximum hierarchy in the corrected vertical frame; and
6. publish an atlas-compatible tile plus resident metadata.

It must no longer compare every tile's CRS and physical cell size with the
observer tile. Compatibility is limited to the selected atlas pool's payload
shape and sample representation.

### Resident cache

Key residency by `TerrainAddress`. Extend `ResidentTile` with the compact
information needed by the GPU, potentially including:

```text
dataset and native tile identity
tile-local frame relative to the current observer
native-grid/control-patch mapping
mapping error bounds
corrected minimum and maximum elevation
validity-mask location
tile neighbour addresses or routing flags
```

Large or variable control meshes should occupy a separate metadata atlas rather
than making every `ResidentTile` record worst-case sized.

Moving the observer invalidates rebased tile-frame metadata and primary trace
results, but not native elevation payloads or static control models. Refactor
`TerrainTraceSession` so a move can update these small records while retaining
compatible atlas slots, rather than necessarily destroying the full session.

### Atlas layout

Keep one cell count and mipmap shape for the first multi-source prototype.
Separate pools may still be required for Float32 and retained-quantized
payloads. Later, introduce an atlas-pool key such as:

```text
(cell count, sample representation, LOD payload layout)
```

Frontier work then identifies both pool and slot. Do not introduce a general
variable-size GPU allocator until fixed pools have been measured and found
insufficient.

## Metal ABI and kernel changes

### Ray and work records

Generalise `RayDirection` and the shadow-ray equivalent into a common physical
ray representation where practical. Primary rays can still exploit their
shared origin as a specialization, while collision and continuation helpers
consume source-independent position/tangent information.

`RayWorkItem` continues to identify a resident slot, ray index, hierarchy start
level, and entry distance. It may additionally need a control-patch or segment
index. Avoid storing local coordinates and derivatives if they are cheap to
rederive and would become stale at every segment boundary.

### Dispatch parameters

Remove physical cell size and grid origin from dispatch-wide
`RaytraceParameters`. Retain values which genuinely apply to the render, such
as ray count, maximum distance, and propagation-model parameters. Move native
geometry and level layout into resident metadata or atlas-pool constants.

### Trace kernel

Refactor `trace_tile_frontier_impl` in this order:

1. Preserve the existing single-grid path as a reference specialization.
2. Express DDA state in cell-index rather than projected-metre coordinates.
3. Obtain the ray segment from resident tile geometry metadata.
4. Traverse conservatively within one valid control patch.
5. Evaluate corrected ray and terrain heights for maximum rejection.
6. Run the generalised bilinear collision solver at the finest required level.
7. Transform the resulting normal to the observer/render frame.
8. Continue into another patch or emit a tile/routing continuation.

The camera and shadow kernels should share the native-grid traversal and
collision implementation. They may retain different entry construction,
termination, and output policies through compile-time specialization.

### Continuation kernels

Replace `lookup_catalogue_tile` against one global row/column hash with:

- direct same-dataset neighbour lookup for the common path;
- a lazy address-to-resident-slot table;
- deferred address emission when a known neighbour is not resident; and
- explicit cross-dataset routing requests where no direct neighbour applies.

Keep per-tile conservative maximum-elevation rejection before requesting its
payload. Add hierarchical region rejection on the host or a later routing pass
so 600 km traces do not load terrain already below the effective horizon.

## Vertical references and source seams

### Common vertical frame

Native horizontal grids may remain unchanged, but elevations used in one trace
must share a vertical reference. Dataset manifests should identify whether
heights are ellipsoidal or tied to a named geoid or national datum.

A vertical transformation may be:

- a constant dataset or tile offset;
- a bilinear correction over a tile;
- a sampled geoid-correction texture; or
- unavailable, in which case the source must be rejected or explicitly marked
  approximate.

Apply the correction during sampling or tile preparation without horizontally
resampling elevations. Conservative mipmap maxima must include the greatest
possible correction within each block.

### Hard handover

The first end-to-end implementation may use a hard priority boundary. It must
still define the mathematical surface at the edge. Simply stopping one
heightfield and beginning another can leave a gap, overlap, or implied vertical
wall when elevations disagree.

Record both boundary profiles and make the ownership rule deterministic. Add a
debug output which colours collisions by dataset and highlights height jumps,
making routing and datum problems distinguishable from ray-transform errors.

### Transition tiles

The production route should support a narrow transition region represented as
a special terrain address. A transition preparer can sample both native
surfaces into a small temporary local grid, estimate a smooth low-frequency
offset for the fallback source, and feather between them. Only source
boundaries are resampled, rather than every tile in the render.

Transition payloads use the normal heightfield tracer and are cached like other
tiles. Their conservative maxima cover both contributing sources and the
blended result. No-data holes and coastlines need policies distinct from an
ordinary overlapping land boundary.

## Shadows

Native-grid support must be implemented for shadows before the feature is
considered complete. Otherwise enabling raytraced shadows either fails at a
source boundary or forces a second common-grid terrain representation.

The observer-relative world representation accommodates shadow origins without
changing terrain storage:

1. Reconstruct the primary collision in the common world frame.
2. Construct the directional sun ray at that location.
3. Transform it into each resident tile's local frame.
4. Run the same segmented native-grid DDA with any-hit termination.
5. Route continuations through the same dataset priority stack.

The tile-frame mapping is shared; only the ray origin and direction differ.
If atmospheric propagation is applied to shadow rays, its approximation and
error bound enter the same conservative traversal calculation.

Source transition surfaces must cast and receive shadows consistently. A
camera collision on a transition tile should begin its shadow ray from that
same blended surface, not from either unblended source.

## Application and command-line changes

Replace the single `--tile-dir` concept with an ordered dataset configuration,
while retaining the old flag as shorthand for one dataset. A future interface
might accept repeated arguments or a manifest:

```text
--terrain swiss.dataset
--terrain france.dataset
--terrain srtm.dataset
```

The configuration needs explicit priority, optional datum overrides, and a way
to disable approximate sources. Observer input should support latitude,
longitude, and elevation; projected input requires an accompanying CRS.

`panorama-app` should display the active terrain source for an inspected point
in diagnostic builds. User-facing attribution should list every dataset which
contributed to the current render or visible coverage.

## Instrumentation

Add counters before optimising:

- rays and shadow rays entering each dataset;
- same-source and cross-source continuations;
- host routing requests and routing time;
- transform segments evaluated and subdivisions required;
- maximum observed transform error by dataset;
- conservative neighbour-cell tests caused by mapping uncertainty;
- native payload bytes loaded by dataset and LOD;
- transition tiles generated and reused;
- atlas installations and evictions per pool;
- terrain regions rejected without payload I/O; and
- collisions by dataset, transition tile, and no-data fallback.

Diagnostic render modes should visualise dataset identity, native tile
boundaries, control-patch boundaries, transform error, selected LOD, and
source-transition bands.

## Implementation stages

### Stage 1: Reference geometry and transform prototype

1. Add a CPU reference which converts a ray position into an arbitrary native
   raster coordinate using GDAL/PROJ and exact host `double` geometry.
2. Fit tile-local affine and adaptive control-patch mappings for the current
   Swiss dataset without changing GPU traversal.
3. Measure mapping error across tiles at several distances and bearings,
   including close to the 600 km limit.
4. Establish subdivision thresholds in cell units.
5. Add exact or independently calculated curvature reference cases.

This stage decides whether the chosen control model is adequate before any
frontier or file-format migration.

### Stage 2: Per-tile geometry in the existing single source

1. Extend resident metadata with the tile-local frame and mapping.
2. Move DDA state into native cell-index coordinates.
3. Move cell geometry out of global `RaytraceParameters`.
4. Generalise bilinear collision and physical-normal calculation.
5. Render the existing prepared Swiss and British datasets through both paths.
6. Require distance, collision, normal, and image parity within defined
   tolerances.

Keep the old kernel specialization available as a benchmark until the native
path is validated.

### Stage 3: Native geographic dataset

1. Generalise CRS storage beyond the `CrsId` enumeration.
2. Add native, non-resampling tile generation for EPSG:4326 SRTM.
3. Rechunk SRTM into the common power-of-two cell count.
4. Validate latitude/longitude cell traversal at different latitudes.
5. Trace SRTM as the only dataset from an observer inside its coverage.
6. Compare collision positions against the CPU PROJ reference.

This proves that native-grid traversal is independent of a metre-based
projected CRS before introducing source handover.

### Stage 4: Lazy multi-source routing

1. Introduce dataset manifests, stable terrain addresses, and the ordered
   dataset stack.
2. Replace the eager single-grid catalogue with lazy geographic indexes.
3. Route direct same-source neighbours without host transformation.
4. Resolve cross-source continuations on the host.
5. Generalise frontier buckets, preparer state, and cache lookup to terrain
   addresses.
6. Add debug colouring by contributing dataset.
7. Test Swiss-to-SRTM and national-to-national boundaries with hard handover.

### Stage 5: Vertical normalization and seams

1. Record and validate source vertical datums.
2. Apply constant and spatially varying vertical corrections.
3. Include correction bounds in maximum mipmaps and catalogue maxima.
4. Detect and report boundary height disagreement.
5. Add cached transition tiles for overlapping sources.
6. Extend transitions to internal no-data holes where appropriate.

### Stage 6: Shadows and presentation consumers

1. Route hard-shadow rays through native grids and source boundaries.
2. Validate off-screen cross-source shadow casters.
3. Return source-independent collision positions and normals to rendering.
4. Update minimap visibility, point inspection, and locked points.
5. Ensure feature outlines do not interpret source or transition artifacts as
   real ridges.
6. Add contributor attribution and diagnostics to both executables.

### Stage 7: Long-range optimisation

1. Add hierarchical geographic elevation bounds and horizon rejection.
2. Avoid enumerating fine tiles across the complete 600 km radius.
3. Benchmark routing, Metal dispatches, cache pressure, and disk I/O at several
   observer elevations and panorama resolutions.
4. Move cross-source routing to the GPU only if host routing is material.
5. Revisit propagation segmentation and exact curvature after measuring their
   device cost.

## Validation

### Geometry tests

- flat synthetic grids in projected and geographic CRSs;
- the same analytic surface sampled into two different native projections;
- rays along cells, tile sides, corners, and control-patch boundaries;
- high-latitude longitude scaling;
- projection convergence across long rays;
- crossings close to a CRS's declared area of use;
- observer-relative precision at 1 m, 1 km, 100 km, and 600 km;
- effective-Earth and unrefracted reference profiles; and
- gradients and normals transformed from degree-based grids.

### Routing and surface tests

- one complete national source over a complete fallback;
- irregular outer coverage and internal no-data holes;
- three overlapping sources with different priorities;
- absent terrain followed by disjoint distant coverage;
- source changes at a native tile side and through a tile interior;
- different vertical offsets with and without correction;
- transition-band continuity and maximum bounds; and
- deterministic corner and boundary ownership.

### Runtime tests

- cache eviction while rays are pending in several datasets;
- concurrent preparation of different source representations;
- moved observers reusing elevation residency;
- Float32 and retained-quantized native tiles;
- camera turns which reuse the native cache;
- low-angle shadow rays crossing many tiles and sources;
- missing, malformed, and unsupported dataset manifests; and
- cancellation or teardown with outstanding routing and I/O.

### Acceptance criteria

The feature is ready when:

- a ray produces the same physical path regardless of the active dataset;
- transform error is bounded and reported rather than assumed negligible;
- SRTM can be traced directly from its latitude/longitude samples;
- source handover cannot lose or duplicate frontier work;
- all elevation comparisons use a compatible vertical frame;
- primary and shadow rays follow identical source-priority rules;
- a 600 km render does not require eager fine-tile enumeration;
- camera movement does not resample resident elevation payloads; and
- profiling shows that tile-local transformation is small relative to terrain
  traversal and I/O.

## Interaction with level of detail

Native-grid multi-source tracing and the design in
[raytracing-lod.md](raytracing-lod.md) should share one addressing, routing, and
cache model. Implementing them independently would otherwise create two ways
to select terrain and two incompatible continuation formats.

### Addressing and source selection

Use the complete key:

```text
(dataset, native tile row, native tile column, LOD)
```

Dataset priority is resolved before or alongside LOD selection:

1. find the highest-priority valid dataset at the ray position;
2. determine the coarsest level from that dataset which satisfies the requested
   screen-space and geometric error;
3. fall back to a lower-priority dataset only when the preferred dataset lacks
   valid coverage, not merely because one of its LOD payloads is absent; and
4. defer or refine when the required level exists but is not resident.

This distinguishes missing coverage from temporarily unloaded detail.

### Native resolution

LOD numbers are dataset-relative unless manifests also publish their physical
cell footprint. `LOD 0` in a 2 m national DEM and `LOD 0` in approximately 30 m
SRTM are not equivalent. Selection must use projected physical footprint and
error, not the integer level alone.

For a native cell, derive its local metre-space dimensions from the tile
mapping Jacobian. The pixel-footprint rule then compares the source cell to the
rendered angular footprint:

```text
cell_footprint_metres <= quality * slant_range * pixel_angle
```

Latitude/longitude cells may have different east-west and north-south sizes;
use the more restrictive projected dimension or a directional footprint along
the ray and image axes.

### LOD payloads

Each dataset should store independently loadable LOD payloads on its own native
grid, as proposed in the LOD plan. Coarse vertices and residual bounds are
generated from that dataset's finer native samples without horizontal
reprojection. The tile's CRS mapping metadata may be shared between levels or
provided at a coarser control resolution with its own error bound.

Do not upsample SRTM into the fine national DEM's LOD hierarchy. Near a source
boundary, a ray may legitimately move from a fine national level to SRTM's
native LOD 0, which is already coarse in physical terms.

### Combined error budget

LOD traversal needs one combined conservative error containing:

```text
terrain approximation residual
+ native-grid transformation error
+ ray-propagation linearisation error
+ vertical-datum correction error
```

The renderer may accept a coarse collision only when the combined error
projects below the selected screen-space tolerance. If source routing,
transform ownership, or terrain separation remains ambiguous, refine either
the control patch, the terrain LOD, or both.

The maximum hierarchy remains a conservative rejection structure, not a coarse
surface. Every LOD still requires actual collision vertices and residual
bounds, exactly as described in the LOD plan.

### Frontier and cache integration

The host frontier should treat a request for a finer level like any other
terrain address. A resident coarse tile may emit:

- a same-LOD continuation into its neighbour;
- a different-LOD continuation selected by distance;
- a refinement request for the same native region;
- a cross-dataset continuation; or
- a transition-tile request near a source seam.

All use the same deferred-work and residency mechanism. Cache pools are keyed
by payload layout, while logical residency and scheduling are keyed by the full
terrain address.

### Source transitions at different LODs

Transition tiles must account for both source and resolution differences. The
blend width should be defined in physical metres, not cells, and its prepared
resolution should meet the current screen-error requirement without inventing
fine detail in the coarser source.

Shared source boundaries and LOD boundaries must not create cracks or false
feature outlines. Transition or stitch metadata should report its geometric
error so the outline pass can distinguish a representation change from a real
depth discontinuity.

### Shadows and inspection

Shadow rays use the same native LOD hierarchy, but ambiguous potential blockers
must refine conservatively as described in the LOD plan. Otherwise distant
coarse terrain may incorrectly change a binary visibility result.

Interactive inspection should request a defined physical accuracy rather than
blindly selecting dataset `LOD 0`. SRTM cannot provide national-DEM accuracy,
and reporting extra decimal places after refinement would imply detail which is
not present in the source.

### Recommended sequencing

Implement the native-grid single-source geometry work before independently
loadable LOD payloads, because LOD selection needs the physical native-grid
Jacobian. Introduce the shared `TerrainAddress` before multi-source routing or
LOD residency. A practical combined order is:

1. native cell-index traversal with per-tile geometry;
2. geographic SRTM at its native resolution;
3. full terrain addresses and lazy source routing;
4. coarse in-memory LOD experiments;
5. independently loadable native LOD payloads;
6. cross-source transitions with physical error metadata; and
7. conservative LOD for cross-source shadow rays.

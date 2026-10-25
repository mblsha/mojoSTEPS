# Multi-source regions and hyperlocal products

The native core computes one regional ensemble. The optional Python adapters
prepare observational fields and derive local products; they do not run forecast
algorithms or pass Python callbacks into the Mojo core. Install with `.[radar]`
(or use bootstrap, which installs all research/demo/radar extras).

## Geometry and units

`Grid(shape, crs, x0, y0, dx, dy)` describes a regular projected Cartesian grid.
Coordinates/spacing are metres; x0/y0 locate the first pixel centre, not a corner.
Axes may descend (negative dy is common). Geographic longitude/latitude grids
must be reprojected. `Grid.window(row,column,size)` constructs a square tile.
A forecast target grid must match `ForecastConfig.size` and its isotropic pixel
scale. Rectangular and differently spaced source grids are supported.

`RadarFrame(source_id, valid_time, grid, dbz, metadata={})` snapshots one field.
Times must be timezone-aware and are normalized to UTC. Masks, NaNs and infinities
become missing; the field is copied and read-only. Source ids identify products,
not independent sensors: two zoom products of one radar are still one radar.

Adapters in `mojosteps.radar`:

| Adapter | Input and explicit contract |
| --- | --- |
| `from_dbz` | dBZ array plus optional boolean valid mask |
| `read_cf_netcdf` | CF `(y,x)` reflectivity, dBZ units, regular m/km axes, grid mapping and one valid time; supports BOM RF3 ground reflectivity schema |
| `from_rain_rate` | mm/h and required `zr_a`, `zr_b`; converts Z=aR^b; zero maps to configured no-rain dBZ; negative/missing remains missing |
| `from_palette` | uint8 RGBA and an explicit colour→proxy-dBZ dictionary; unknown colours remain missing |

The NetCDF reader honours CF packed scaling/fill values and records a file hash.
Quality/error flags require a source-specific boolean valid mask: the adapter
never guesses whether a numerical error/quality value is good. Rain-rate Z–R
conversion and palette proxy conversion do not make different products equivalent
or establish quantitative input qualification. Acquisition, authentication,
archive retention, source licences and official outages belong to the caller.
No live radar feed, credentials or network downloader is included here.

## Alignment, timestamps and coverage

`prepare_input(frames, grid, config, origin, priority=[...])` constructs exactly
three inputs at origin−2*step, origin−step and origin. It does not snap timestamps
or substitute stale frames. Duplicate source/time pairs and unknown source ids
raise errors. Missing higher-priority sources can fall back to lower-priority
sources at the same valid time.

Sources are reprojected/resampled onto the target pixel centres with bilinear
interpolation in dBZ. A sample is valid only if every contributing pixel is
valid; interpolation does not bridge missing coverage. At each pixel the first
valid source in the explicit priority list wins. There is no averaging or
probability-independence assumption. Returned `provenance[history,y,x]` contains
source-list indices, preserving which product supplied each observation.

Any remaining uncovered pixel raises `CoverageError`: the forecast for that tile
is unavailable. Missing observations are never filled with dry values. Select
covered tiles or supply another valid source. This deliberately does not offer
partial-domain forecasts or infer rain through a radar's blind sectors.

## One forecast, many locations

```python
from mojosteps import Forecast, ForecastConfig
from mojosteps.radar import Grid, read_cf_netcdf
from mojosteps.regional import AreaTarget, RegionalForecaster

config = ForecastConfig(size=256, kmperpixel=0.5, timestep=5)
worker = RegionalForecaster(Forecast("gpu", config=config))
# frames: three or more timestamped RadarFrame objects from your adapters.
# grid: a covered 256-square projected target grid at 500 m.
# origin: timezone-aware observation time. velocity: x/y pixels per 5-min scan.
result = worker.forecast(frames, grid, origin, velocity, 14,
                         priority=["terrey-hills", "appin"], members=96)
for name, longitude, latitude in locations:
    target = AreaTarget.lonlat_circle(name, grid, longitude, latitude, radius_m=1000)
    product = result.probabilities(target, thresholds=(15,20), minimum_members=90)
```

`AreaTarget(name, weights)` accepts nonnegative per-cell area weights. `circle`
and `lonlat_circle` calculate pixel intersection areas using a 256-edge circle
in the projected grid CRS; these are projected distances, not geodesic circles.
A circle extending outside the grid is rejected. One ensemble serves all targets
in that region; queries do not run additional forecasts.

A member is wet when at least `wet_fraction` (default 0.2) of the target's weighted
area exceeds the threshold. Every positive-weight cell must be finite for that
member/lead to be eligible. `probabilities` returns probabilities and valid-member
counts per lead/threshold; insufficient members produce NaN, not zero risk.
`window_risk(target,start_minutes,duration_minutes,...)` uses complete member
trajectories for the inclusive, scan-aligned window, including the current
observation when start=0. It returns `None` when support is insufficient. Do not
multiply marginal lead probabilities to infer window risk.

These products are radar-echo events. Region/source-specific calibration,
forecast-skill assessment and surface-rain validation remain separate work. A
longer supported numerical horizon does not establish longer useful lead time.
Optical-flow estimation remains external; use a motion estimator on the aligned
history and convert its velocity to target-grid pixels per configured scan.

## Larger and rectangular coverage

`tiles(source_grid, size, halo=32)` yields overlapping `Tile` objects with source
row/column offsets, target grid and an interior `query_slice`. Forecast complete
square tiles and serve targets in their interiors; the surrounding halo supplies
motion context. Choose a halo adequate for expected motion and horizon. This
helper does not stitch or declare the outer observation boundary forecastable.

Ensembles generated separately for different tiles are not coherent joint
member trajectories. Do not compute a route-wide or multi-region joint risk by
pairing their member indices. Within one regional ensemble, joint trajectories
are retained. Network scheduling, incremental publication and production service
migration are outside this library change; burakku collection is unchanged.

## Synthetic consumer and checks

```sh
.venv/bin/python examples/multi_source.py --backend gpu --size 256 --kmperpixel 1
MOJOSTEPS_TEST_GPU=1 .venv/bin/python -m pytest -q
```

The example combines synthetic dBZ and rain-rate sources, retains provenance and
answers three circular targets from one forecast. Tests cover CF packing/fill,
reprojection, descending axes, unknown colours, missing scans, priority fallback,
area weighting, member completeness and whole-window dependence. Runtime parity
tests cover all grid sizes, both motion dtypes, changed scales/thresholds and an
18-step horizon. Hosted CI validates CPU; Metal evidence requires real hardware.


For 256/512-square GPU forecasts, model fitting uses native CPU float64 with
pairwise correlation sums. This avoids rare large-grid CDF rank changes; the
ensemble stages remain on Metal. Smaller GPU grids retain on-device fitting.
These paths are explicitly tested rather than silently falling back after an
execution error.


## Expanded workload observations

For 96 members × 14 leads × 512² cells (500 m pixels), the final matrix observed
these warm native-call times with 16 requested workers:

| Host | CPU | GPU |
| --- | ---: | ---: |
| M5 Ultra / koubou5 | 7.25 s | 3.47 s |
| M1 Ultra / koubou1 | 4.61 s | 10.82 s |

These are one warm call after an initial call, under ordinary host load, rather
than statistical architecture comparisons. CPU is the faster observed choice
for this M1 workload. Native allocation, fitting, all forecast stages and readback
are included; source alignment and local-product queries are outside timing.
Both backends pass the unchanged 1e-5 dBZ tolerance and threshold contract; maximum
error across this matrix is below 3.3e-9 dBZ. Original 192² output hashes are
unchanged. See [M5 receipts](../benchmarks/regional/m5-ultra.json) and
[M1 receipts](../benchmarks/regional/m1-ultra.json). The 512² output alone occupies
2.625 GiB; these times do not establish a minimum memory requirement.

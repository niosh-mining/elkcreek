# EPISODES Platform Export Plan

Goal: package the Elk Creek data in this repository as an Anthropogenic Hazards
episode on the [EPISODES Platform](https://episodesplatform.eu) (EPOS TCS AH).

## Target formats

| Data | EPISODES format |
|---|---|
| Seismic catalog | Catalog `.mat`: vector of structs with `field, type, val, unit, description, fieldType`. Required: `ID` (text), `Time` (MATLAB datenum). |
| Waveforms | miniSEED, linked to catalog event `ID`s |
| Seismic network | SC3ML (also GDF "Seismic Stations") |
| Everything else | GDF `.mat`: `FormatName, FormatVersion, CRS, TimeZone, Description, FieldDescription, FieldType, FieldUnit, d`. Named `GDF_<EPISODE>_<description>.mat`; WGS84 (EPSG:4326). |

Note: the catalog format is a flat table with one row per event. It can't hold
multiple origins or magnitudes per event, so the IMS and BEMIS catalogs go in as
separate catalogs, or as source-tagged columns if the maintainer allows it.

References:
- [Seismic catalog format](https://episodesplatform.eu/docs/data-formats/catalogs/seismic-catalog/catalog-description)
- [GDF format](https://episodesplatform.eu/docs/data-formats/gdf/gdf-description)
- [GDF types](https://episodesplatform.eu/docs/data-formats/gdf/gdf-types)
- [Seismic network](https://episodesplatform.eu/docs/data-formats/seismic-network)
- [Waveforms](https://episodesplatform.eu/docs/data-formats/waveforms)
- [Orlecka-Sikora et al. 2020, Sci. Data](https://pmc.ncbi.nlm.nih.gov/articles/PMC7066136/)

## Checklist

### Questions for the episode maintainer
- [ ] Episode code name (e.g. `ELK_CREEK`) and the template/form to fill in
- [ ] Delivery method: their eNode, a shared drive, or a zip?
- [ ] Are non-standard catalog fields (e.g. `ML_IMS`, `ML_BEMIS`) allowed? If not, send separate catalogs.
- [ ] Is QuakeML accepted alongside the `.mat` catalog (to carry multiple origins/magnitudes per event)?
- [ ] StationXML accepted, or must it be SC3ML?
- [ ] Does the network need an FDSN-registered code, or is `EC` fine?
- [ ] Can GDF types be added for pressure cells, sonic probes, ground-condition surveys, and support cans?
- [ ] Request sample files (`Catalog_create.7z`, GDF examples) for validation

### Episode metadata
- [ ] Full name and code name
- [ ] Description: site, mining method, monitoring history, the five bursts
- [ ] Impacting factor: underground mining
- [ ] Data owner/contact: NIOSH
- [ ] Citation: Boltz MS, Chambers DJA, Larson MK, Lawson HE, Törnman W [2026]. Coal Bursting at the Elk Creek Mine: A Western United States Case Study. Mining, Metallurgy & Exploration, https://doi.org/10.1007/s42461-026-01497-0
- [ ] License: public domain / CC-BY 4.0
- [ ] Representative picture (e.g. `p010_burst_events.png`)
- [ ] Attached documents and references (incl. `Notes on Observations of Conditions and Caving.pdf`)

### Setup
- [x] Create branch `episodes`
- [ ] `src/elkcreek/episodes.py`: datenum conversion, catalog `.mat` writer, GDF `.mat` writer (`scipy.io.savemat`), mine coords → WGS84, m → km
- [ ] Tests: reload each `.mat` with `scipy.io.loadmat` and compare against the maintainer's examples

### Seismic catalogs
- [ ] Verify units in the raw IMS and BEMIS catalogs (stress drops differ ~100×, possibly Pa vs MPa; also apparent stress and energy). Check against the paper.
- [ ] Confirm BEMIS moment tensor axes; rotate to `MTrr, MTss, MTee, MTrs, MTre, MTse`
- [ ] Compute `Depth` (km below ground) per event from `topo.dxf`; also include `Elevation`, `X`, `Y`, `Z`
- [ ] Map columns to standard fields: `ID, Time, Lat, Long, ML, Mw, M0, E, Ep, Es, R, sigma_a, delta_sigma, EPI_err, Nl`, ...
- [ ] `e010`: cleaned combined catalog
- [ ] `e011`: raw IMS catalog
- [ ] `e012`: raw BEMIS catalog
- [ ] `e013`: combined QuakeML with both origins/magnitudes per event (if accepted)

### Waveforms (bursts only for now)
- [ ] Rename channels to SEED codes (e.g. `GPZ/GPN/GPE` for 5 kHz geophones)
- [ ] Make catalog event IDs match waveform files
- [ ] `e020`: five burst miniSEED files + `burst_events.qml` picks
- [ ] Later: full waveforms from [data.cdc.gov](https://data.cdc.gov/browse?q=elk+creek+seismic+waveforms&sortBy=alpha&pageSize=20)

### Stations
- [ ] StationXML from `station_list.csv`: lat/lon, elevation, start/end dates, orientation (underground: azimuth 285, dip 90)
- [ ] Unity-gain response (data already in m/s); note sensor types 123/23/131 in descriptions
- [ ] Convert to SC3ML with SeisComP `fdsnxml2inv`
- [ ] `e030`: also write GDF "Seismic Stations"
- [ ] Resolve missing start/end dates for station FNE

### Mining and geology GDFs
- [ ] `e040`: Mining Front Advance from `face_positions.csv`
- [ ] `e050`: Mine Area / Cartesian Mine Area from workings DXFs
- [ ] `e051`: Cartesian Weakzone Area from `anomalous_zones.dxf`
- [ ] `e052`: faults, overburden, topo (type to confirm with maintainer)
- [ ] `e060`: Velocity Model from `homogeneous.csv`

### Instrumentation GDFs
- [ ] `e070`: instrument locations from `instrumentation.csv`
- [ ] `e071`: extensometers (MPBX, string pots) as Insitu Deformation / Strain Gauge Measurement
- [ ] `e072`: closure stations
- [ ] `e073`: borehole pressure cells (type depends on the maintainer)
- [ ] `e074`: support cans, sonic probes, ground-condition surveys (type depends on the maintainer)
- [ ] Fill in missing units/sensor specs/install dates from the paper or notes

### Package and submit
- [ ] `e080`: assemble `data/episodes/ELK_CREEK/` with README, then zip
- [ ] Load every file in Octave/MATLAB as a final check
- [ ] Send an early sample (catalog + stations + one GDF) to the maintainer
- [ ] Full submission → test-portal review → publish → add DOI to `readme.md`

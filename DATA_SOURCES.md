# 📊 Data Sources Documentation

This document provides comprehensive information about all data sources used in the Australian Bushfire ML Analysis project.

---

## 🛰️ NASA FIRMS - Fire Detection Data

### Overview
**Source:** Fire Information for Resource Management System (FIRMS)  
**Provider:** NASA/LANCE  
**Website:** https://firms.modaps.eosdis.nasa.gov/

### Data Description
FIRMS provides near real-time active fire data from MODIS and VIIRS instruments aboard NASA satellites. The system detects thermal anomalies (fires) by measuring brightness temperature in specific infrared bands.

### Datasets Used

#### 1. MODIS Archive Data
- **Satellites:** Terra (10:30am) & Aqua (1:30pm) overpasses
- **Spatial Resolution:** 1km at nadir
- **Temporal Resolution:** Daily (2 passes)
- **Records:** ~160,000 fires
- **Confidence Levels:** Low, Nominal, High

#### 2. VIIRS Archive Data
- **Satellite:** Suomi-NPP
- **Spatial Resolution:** 375m at nadir (higher than MODIS)
- **Temporal Resolution:** Daily
- **Records:** ~90,000 fires
- **Confidence Levels:** Low, Nominal, High

#### 3. MODIS Near Real-Time (NRT)
- **Latency:** 3-hour delay
- **Use:** Rapid response to recent fires
- **Records:** ~20,000 fires

#### 4. VIIRS Near Real-Time (NRT)
- **Latency:** 3-hour delay
- **Records:** ~20,000 fires

### Key Variables

| Variable | Type | Description | Units |
|----------|------|-------------|-------|
| `latitude` | Numeric | Fire location latitude | Decimal degrees |
| `longitude` | Numeric | Fire location longitude | Decimal degrees |
| `brightness` | Numeric | Brightness temperature | Kelvin |
| `bright_ti4` | Numeric | MODIS Band 21/22 brightness | Kelvin |
| `bright_ti5` | Numeric | MODIS Band 31 brightness | Kelvin |
| `scan` | Numeric | Pixel size along scan direction | Kilometers |
| `track` | Numeric | Pixel size along track direction | Kilometers |
| `acq_date` | Date | Acquisition date | YYYY-MM-DD |
| `acq_time` | Time | Acquisition time | HHMM |
| `satellite` | Categorical | Satellite name | Aqua/Terra/N |
| `confidence` | Mixed | Detection confidence | h/n/l or 0-100 |
| `version` | String | Product version | e.g., "6.1NRT" |
| `bright_t31` | Numeric | MODIS Band 31 temperature | Kelvin |
| `frp` | Numeric | Fire Radiative Power | Megawatts (MW) |
| `daynight` | Categorical | Day or night detection | D/N |

### Data Quality Notes

**Strengths:**
- High temporal coverage (multiple daily overpasses)
- Global coverage
- Free and publicly available
- Well-documented and validated

**Limitations:**
- Cloud cover can obscure fires
- Some fires too small to detect (< 1000m² for VIIRS)
- Confidence metric not standardized across products
- Different pixel sizes affect detection sensitivity

### Download Instructions

1. Visit https://firms.modaps.eosdis.nasa.gov/
2. Select region: Australia
3. Date range: 2019-09-01 to 2020-01-31
4. Select products: MODIS Archive, VIIRS Archive, MODIS NRT, VIIRS NRT
5. Download CSV files

### Citation

```
FIRMS: LANCE MODIS Active Fire Product. NASA's Earth Observing System Data and 
Information System (EOSDIS) with funding provided by NASA Headquarters. 
https://earthdata.nasa.gov/firms.
```

---

## 🌤️ Australian Bureau of Meteorology (BOM) - Weather Data

### Overview
**Source:** Australian Bureau of Meteorology  
**Website:** http://www.bom.gov.au/climate/data/

### Data Description
Daily weather observations from high-quality stations across Australia. We selected 5 major city stations representing different climate zones.

### Stations Used

| Station | Name | ID | Latitude | Longitude | State |
|---------|------|----|---------:|---------:|-------|
| Adelaide | Adelaide (West Terrace) | 023000 | -34.9285 | 138.6007 | SA |
| Brisbane | Brisbane | 040842 | -27.3817 | 153.1150 | QLD |
| Sydney | Sydney Observatory Hill | 066062 | -33.8607 | 151.2055 | NSW |
| Canberra | Canberra Airport | 070351 | -35.3088 | 149.2004 | ACT |
| Melbourne | Melbourne (Olympic Park) | 086282 | -37.6690 | 144.8320 | VIC |

### Variables Collected

#### Rainfall Data (IDCJAC0009)
- **Filename Pattern:** `IDCJAC0009_[STATION]_[DATES].csv`
- **Variable:** Daily rainfall amount
- **Units:** Millimeters (mm)
- **Precision:** 0.1mm
- **Missing Values:** Blank cells indicate no observation

#### Maximum Temperature (IDCJAC0010)
- **Filename Pattern:** `IDCJAC0010_[STATION]_[DATES].csv`
- **Variable:** Daily maximum temperature
- **Units:** Degrees Celsius (°C)
- **Precision:** 0.1°C
- **Quality Codes:** Y (verified), N (not verified)

#### Minimum Temperature (IDCJAC0011)
- **Filename Pattern:** `IDCJAC0011_[STATION]_[DATES].csv`
- **Variable:** Daily minimum temperature
- **Units:** Degrees Celsius (°C)
- **Precision:** 0.1°C

### Data Processing

**Files Downloaded:**
- 5 stations × 3 variables = 15 CSV files
- Period: September 2019 - January 2020 (153 days)
- Total: 610 station-days (some missing data)

**Loading Function:**
```r
load_rainfall <- function(filepath) {
  read_csv(filepath) %>%
    rename(
      station_number = `Bureau of Meteorology station number`,
      year = Year,
      month = Month,
      day = Day,
      rainfall_mm = `Rainfall amount (millimetres)`
    ) %>%
    mutate(date = make_date(year, month, day))
}
```

### Station Selection Rationale

1. **Sydney (066062):** Largest fire concentration, NSW fires
2. **Canberra (070351):** Central to major fire activity
3. **Melbourne (086282):** Victoria fires
4. **Brisbane (040842):** Queensland fires
5. **Adelaide (023000):** South Australia fires

### Geographic Matching

Each fire assigned to nearest station using coordinate boundaries:

```r
nearest_station = case_when(
  longitude > 150 & latitude > -35 & latitude < -32 ~ "066062",  # Sydney
  longitude < 146 & latitude < -36 ~ "086282",                   # Melbourne
  longitude > 152 & latitude > -28 ~ "040842",                   # Brisbane
  longitude < 140 & latitude < -33 ~ "023000",                   # Adelaide
  longitude > 148 & longitude < 150 & latitude < -34 ~ "070351", # Canberra
  TRUE ~ "066062"  # Default to Sydney
)
```

### Data Quality

**Strengths:**
- High-quality, professionally maintained stations
- Long historical records
- Verified observations
- Free and public access

**Limitations:**
- Only 5 stations for entire continent
- Some fires >100km from nearest station
- Missing data on some days
- No wind speed data available

### Download Instructions

1. Visit http://www.bom.gov.au/climate/data/
2. Select "Daily rainfall" or "Daily maximum temperature"
3. Select station by name or number
4. Choose date range: 2019-09-01 to 2020-01-31
5. Download CSV
6. Repeat for all 5 stations and 3 variables

### Citation

```
Australian Bureau of Meteorology (2020). Climate Data Online. 
Commonwealth of Australia. http://www.bom.gov.au/climate/data/
```

---

## 👥 WorldPop - Population Density Data

### Overview
**Source:** WorldPop Project  
**Provider:** University of Southampton  
**Website:** https://www.worldpop.org/

### Data Description
High-resolution gridded population estimates for Australia. Uses census data, satellite imagery, and machine learning to estimate population at 1km resolution.

### Dataset Used
- **Filename:** `aus_pd_2019_1km_UNadj.tif`
- **Format:** GeoTIFF raster
- **Year:** 2019
- **Resolution:** 1km × 1km (~100 hectares per pixel)
- **CRS:** WGS84 (EPSG:4326)
- **Adjustment:** UN-adjusted (original model estimates)

### Technical Specifications

| Property | Value |
|----------|-------|
| Pixel Size | 0.00833333° (~1km at equator) |
| Dimensions | ~4000 × 3500 pixels |
| Data Type | Float32 |
| No Data Value | -99999 |
| Units | People per pixel |

### Variables

- **Population Density:** Number of people per km² grid cell
- **Range:** 0 (uninhabited) to 10,000+ (dense urban)

### Extraction Method

```r
# Load raster
pop_raster <- raster("aus_pd_2019_1km_UNadj.tif")

# Convert fires to spatial points
fires_sp <- fires %>%
  st_as_sf(coords = c("longitude", "latitude"), crs = 4326) %>%
  st_transform(crs = crs(pop_raster))

# Extract population at each fire location
population_at_fires <- raster::extract(pop_raster, as(fires_sp, "Spatial"))
```

### Derived Variables

We created categorical bins:

```r
pop_exposure = cut(population_density,
                   breaks = c(-Inf, 0.1, 1, 10, 100, 1000, Inf),
                   labels = c("Uninhabited", "Very Rural", "Rural", 
                              "Suburban", "Urban", "Dense Urban"))
```

| Category | Range (people/km²) | Description |
|----------|-------------------:|-------------|
| Uninhabited | 0 - 0.1 | No permanent population |
| Very Rural | 0.1 - 1 | Sparse rural |
| Rural | 1 - 10 | Rural areas |
| Suburban | 10 - 100 | Suburban areas |
| Urban | 100 - 1000 | Urban areas |
| Dense Urban | >1000 | City centers |

### Data Quality

**Strengths:**
- High spatial resolution (1km)
- Global coverage
- Validated against census
- Free for academic use

**Limitations:**
- Estimates, not exact counts
- 2019 data (may not reflect 2025 population)
- Assumes uniform distribution within pixels
- Less accurate in remote areas

### Download Instructions

1. Visit https://www.worldpop.org/geodata/listing?id=75
2. Select "Australia"
3. Year: 2019
4. Type: "Population Density"
5. Resolution: "1km"
6. Adjustment: "UN-adjusted"
7. Download GeoTIFF file (~50MB)

### Citation

```
WorldPop (2020). Global 1km Population. University of Southampton. 
DOI: 10.5258/SOTON/WP00647.
Available at: https://www.worldpop.org/
```

---

## 📁 Data Files Structure

### Raw Data Directory
```
data/raw/
├── fires/
│   ├── MODIS_Archive_2019-2020.csv
│   ├── VIIRS_Archive_2019-2020.csv
│   ├── MODIS_NRT_2019-2020.csv
│   └── VIIRS_NRT_2019-2020.csv
├── weather/
│   ├── Rainfall/
│   │   ├── IDCJAC0009_023000_2019-2020.csv  (Adelaide)
│   │   ├── IDCJAC0009_040842_2019-2020.csv  (Brisbane)
│   │   ├── IDCJAC0009_066062_2019-2020.csv  (Sydney)
│   │   ├── IDCJAC0009_070351_2019-2020.csv  (Canberra)
│   │   └── IDCJAC0009_086282_2019-2020.csv  (Melbourne)
│   ├── MaxTemp/
│   │   └── [5 files similar to above]
│   └── MinTemp/
│       └── [5 files similar to above]
└── population/
    └── aus_pd_2019_1km_UNadj.tif
```

### Processed Data
```
data/processed/
├── fires_complete_CLEAN.csv        (288,876 × 54 features)
├── fires_for_xgboost.csv          (254,012 × 25 features)
├── high_risk_events.csv           (219 × 54 features)
└── bom_weather_complete.csv       (610 station-days)
```

---

## 📊 Data Statistics

### Fire Data
- **Total Detections:** 288,876
- **Date Range:** 2019-09-01 to 2020-01-31 (153 days)
- **Geographic Extent:** 
  - Latitude: -43.5° to -10.5°
  - Longitude: 113.5° to 154.0°
- **FRP Range:** 0.1 MW to 7,401 MW
- **Average FRP:** 55.4 MW

### Weather Data
- **Stations:** 5
- **Days:** 153
- **Total Station-Days:** 610 (some missing)
- **Temperature Range:** 3.2°C to 45.3°C
- **Rainfall Range:** 0mm to 97mm (daily)

### Population Data
- **Pixels Sampled:** 288,876 (one per fire)
- **Population Range:** 0 to 2,847 people/km²
- **Median:** 0.15 people/km²
- **Fires in Uninhabited Areas:** 236,788 (82%)

---

## ⚖️ Data Usage and Licensing

### NASA FIRMS
- **License:** Public Domain (US Government Work)
- **Attribution:** Required
- **Commercial Use:** Allowed

### BOM Weather Data
- **License:** Creative Commons Attribution 4.0
- **Attribution:** Required
- **Commercial Use:** Allowed with attribution

### WorldPop
- **License:** Creative Commons Attribution 4.0 International
- **Attribution:** Required
- **Commercial Use:** Allowed with attribution
- **Academic Use:** Encouraged

### Our Project
- **License:** MIT
- **Data:** Not redistributed (links provided for download)
- **Code:** Open source

---

## 🔄 Data Updates

### Frequency
- **NASA FIRMS:** Daily updates available
- **BOM:** Daily updates, historical data added monthly
- **WorldPop:** Annual updates (latest: 2020)

### Historical Data
- FIRMS archives available from 2000 (MODIS) and 2012 (VIIRS)
- BOM historical data available from 1800s for some stations
- WorldPop available from 2000-present

---

## 📞 Contact for Data Issues

### NASA FIRMS
- Email: support@earthdata.nasa.gov
- Documentation: https://www.earthdata.nasa.gov/faq/firms-faq

### Australian BOM
- Email: climatedata@bom.gov.au
- Phone: +61 3 9669 4000

### WorldPop
- Email: info@worldpop.org
- Support: https://www.worldpop.org/about

---

## 📚 Additional Resources

### Tutorials
- [FIRMS User Guide](https://www.earthdata.nasa.gov/learn/find-data/near-real-time/firms)
- [BOM Climate Data Guide](http://www.bom.gov.au/climate/data-services/)
- [WorldPop Methods](https://www.worldpop.org/methods/top_down_constrained_vs_unconstrained)

### Research Papers
1. Giglio, L., et al. (2016). "The Collection 6 MODIS active fire detection algorithm and fire products." *Remote Sensing of Environment*.
2. Schroeder, W., et al. (2014). "The New VIIRS 375m active fire detection data product." *Remote Sensing of Environment*.
3. Stevens, F.R., et al. (2015). "Disaggregating census data for population mapping using Random Forests with remotely-sensed and ancillary data." *PLOS ONE*.

---

**Last Updated:** December 2024  
**Maintained By:** Vinh [Last Name]  
**Questions?** Open an issue or contact: your.email@example.com

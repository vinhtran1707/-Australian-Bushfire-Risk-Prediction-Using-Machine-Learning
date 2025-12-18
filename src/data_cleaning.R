# ============================================================================
# AUSTRALIAN BUSHFIRE DATA ANALYSIS - DATA CLEANING & INTEGRATION
# ============================================================================
# Author: Vinh
# Course: MGSC 7310 - Tulane University
# Description: Integrates NASA FIRMS fire data, BOM weather data, and 
#              WorldPop population data for bushfire risk analysis
# ============================================================================

# Load Required Libraries ----
library(tidyverse)
library(lubridate)
library(sf)
library(raster)
library(slider)
library(scales)

# Set Working Directory ----
# Update this path to match your local setup
# setwd("C:/Projects/australian-bushfire-ml-analysis")

# ============================================================================
# PART 1: LOAD AND STANDARDIZE FIRE DATA
# ============================================================================

# Function to standardize fire data formats
standardize_fire_data <- function(df) {
  df %>%
    mutate(
      # Standardize data types
      across(any_of(c("confidence", "version", "track")), as.character),
      across(any_of(c("latitude", "longitude", "brightness", "bright_ti4", 
                      "bright_ti5", "frp", "scan", "track")), as.numeric)
    )
}

# Load Fire Datasets ----
cat("Loading fire datasets...\n")

# Update these paths to your data location
dataroot <- "data/raw/fires"
datafiles <- list.files(path = dataroot, pattern = "\\.csv$", full.names = TRUE)
datafiles_ls <- lapply(datafiles, function(x) read_csv(file = x, show_col_types = FALSE))

# Assign to variables (adjust indices based on your file order)
modis_archive <- datafiles_ls[[1]] %>% standardize_fire_data()
viirs_archive <- datafiles_ls[[2]] %>% standardize_fire_data()
modis_nrt <- datafiles_ls[[3]] %>% standardize_fire_data()
viirs_nrt <- datafiles_ls[[4]] %>% standardize_fire_data()

# Combine all fire datasets
all_fires <- bind_rows(
  modis_archive %>% mutate(source = "MODIS_Archive"),
  viirs_archive %>% mutate(source = "VIIRS_Archive"),
  modis_nrt %>% mutate(source = "MODIS_NRT"),
  viirs_nrt %>% mutate(source = "VIIRS_NRT")
)

cat(sprintf("✓ Loaded %s total fire detections\n", format(nrow(all_fires), big.mark = ",")))

# ============================================================================
# PART 2: DATA CLEANING AND FEATURE ENGINEERING
# ============================================================================

cat("Cleaning and engineering features...\n")

all_fires_clean <- all_fires %>%
  mutate(
    # Unify brightness columns (different satellites use different names)
    brightness = case_when(
      !is.na(brightness) ~ brightness,
      !is.na(bright_ti4) ~ bright_ti4,
      !is.na(bright_ti5) ~ bright_ti5,
      TRUE ~ NA_real_
    ),
    
    # Parse dates and create temporal features
    acq_date = ymd(acq_date),
    month = month(acq_date, label = TRUE),
    month_num = month(acq_date),
    week = week(acq_date),
    year = year(acq_date),
    
    # Convert confidence to numeric
    confidence_numeric = case_when(
      confidence == "l" ~ 15,   # Low confidence
      confidence == "n" ~ 55,   # Nominal confidence
      confidence == "h" ~ 90,   # High confidence
      TRUE ~ as.numeric(confidence)
    ),
    
    # Create confidence flag
    high_confidence = confidence_numeric >= 80,
    
    # Convert acquisition time to numeric
    acq_time = as.numeric(acq_time)
  ) %>%
  # Filter to study period (Black Summer: Sep 2019 - Jan 2020)
  filter(acq_date >= "2019-09-01" & acq_date <= "2020-01-31") %>%
  # Remove records with missing critical data
  filter(!is.na(latitude), !is.na(longitude), !is.na(frp))

cat(sprintf("✓ Cleaned dataset: %s fires\n", format(nrow(all_fires_clean), big.mark = ",")))

# ============================================================================
# PART 3: LOAD AND PROCESS WEATHER DATA
# ============================================================================

cat("Loading weather data...\n")

# Function to load rainfall data
load_rainfall <- function(filepath) {
  read_csv(filepath, show_col_types = FALSE) %>%
    rename(
      station_number = `Bureau of Meteorology station number`,
      year = Year,
      month = Month,
      day = Day,
      rainfall_mm = `Rainfall amount (millimetres)`
    ) %>%
    mutate(date = make_date(year, month, day)) %>%
    select(station_number, date, rainfall_mm)
}

# Function to load max temperature data
load_max_temp <- function(filepath) {
  read_csv(filepath, show_col_types = FALSE) %>%
    rename(
      station_number = `Bureau of Meteorology station number`,
      year = Year,
      month = Month,
      day = Day,
      max_temp = `Maximum temperature (Degree C)`
    ) %>%
    mutate(date = make_date(year, month, day)) %>%
    select(station_number, date, max_temp)
}

# Function to load min temperature data
load_min_temp <- function(filepath) {
  read_csv(filepath, show_col_types = FALSE) %>%
    rename(
      station_number = `Bureau of Meteorology station number`,
      year = Year,
      month = Month,
      day = Day,
      min_temp = `Minimum temperature (Degree C)`
    ) %>%
    mutate(date = make_date(year, month, day)) %>%
    select(station_number, date, min_temp)
}

# Load all weather files
weather_root <- "data/raw/weather"

bom_rainfall_files <- list.files(
  path = file.path(weather_root, "Rainfall"),
  pattern = "IDCJAC0009.*\\.csv$",
  full.names = TRUE
)

bom_maxtemp_files <- list.files(
  path = file.path(weather_root, "MaxTemp"),
  pattern = "IDCJAC0010.*\\.csv$",
  full.names = TRUE
)

bom_mintemp_files <- list.files(
  path = file.path(weather_root, "MinTemp"),
  pattern = "IDCJAC0011.*\\.csv$",
  full.names = TRUE
)

# Load and combine weather data
bom_rainfall <- map_dfr(bom_rainfall_files, load_rainfall)
bom_max_temp <- map_dfr(bom_maxtemp_files, load_max_temp)
bom_min_temp <- map_dfr(bom_mintemp_files, load_min_temp)

# Combine all weather variables
bom_weather_complete <- bom_rainfall %>%
  full_join(bom_max_temp, by = c("date", "station_number")) %>%
  full_join(bom_min_temp, by = c("date", "station_number")) %>%
  # Create weather features
  mutate(
    # Temperature metrics
    temp_range = max_temp - min_temp,
    avg_temp = (max_temp + min_temp) / 2,
    
    # Heat classifications
    extreme_heat = max_temp > 40,
    very_hot = max_temp > 38,
    heat_stress = max_temp > 35,
    
    # Rainfall classifications
    dry_day = rainfall_mm < 1,
    light_rain = rainfall_mm >= 1 & rainfall_mm < 5,
    moderate_rain = rainfall_mm >= 5 & rainfall_mm < 20,
    heavy_rain = rainfall_mm >= 20
  ) %>%
  # Calculate rolling statistics by station
  group_by(station_number) %>%
  arrange(date) %>%
  mutate(
    # Rolling rainfall sums
    rain_sum_7day = slide_dbl(rainfall_mm, sum, .before = 6, .complete = TRUE),
    rain_sum_30day = slide_dbl(rainfall_mm, sum, .before = 29, .complete = TRUE),
    
    # Lag features
    max_temp_lag1 = lag(max_temp, 1),
    max_temp_lag7 = lag(max_temp, 7),
    rain_lag1 = lag(rainfall_mm, 1),
    rain_lag7 = lag(rainfall_mm, 7)
  ) %>%
  ungroup()

# Calculate days since rain (drought tracking)
bom_weather_complete <- bom_weather_complete %>%
  group_by(station_number) %>%
  arrange(date) %>%
  mutate(
    days_since_rain = {
      rain_days <- which(rainfall_mm > 1)
      sapply(seq_along(rainfall_mm), function(i) {
        if (i %in% rain_days) return(0)
        prev_rain <- max(rain_days[rain_days < i], 0)
        return(i - prev_rain)
      })
    }
  ) %>%
  ungroup()

cat(sprintf("✓ Processed weather data: %s station-days\n", 
            format(nrow(bom_weather_complete), big.mark = ",")))

# ============================================================================
# PART 4: MATCH FIRES TO NEAREST WEATHER STATION
# ============================================================================

cat("Matching fires to weather stations...\n")

# Assign each fire to nearest weather station based on geographic location
fires_with_weather <- all_fires_clean %>%
  mutate(
    nearest_station = case_when(
      # Sydney region
      longitude > 150 & latitude > -35 & latitude < -32 ~ "066062",
      # Melbourne region
      longitude < 146 & latitude < -36 ~ "086282",
      # Brisbane region
      longitude > 152 & latitude > -28 ~ "040842",
      # Adelaide region
      longitude < 140 & latitude < -33 ~ "023000",
      # Canberra region
      longitude > 148 & longitude < 150 & latitude < -34 ~ "070351",
      # Default to Sydney
      TRUE ~ "066062"
    )
  ) %>%
  # Join weather data
  left_join(
    bom_weather_complete,
    by = c("acq_date" = "date", "nearest_station" = "station_number")
  )

cat(sprintf("✓ Matched %s fires to weather stations\n", 
            format(nrow(fires_with_weather), big.mark = ",")))

# ============================================================================
# PART 5: EXTRACT POPULATION DENSITY
# ============================================================================

cat("Extracting population density...\n")

# Load population raster
pop_raster <- raster("data/raw/population/aus_pd_2019_1km_UNadj.tif")

# Convert fires to spatial points
fires_sp <- fires_with_weather %>%
  st_as_sf(coords = c("longitude", "latitude"), crs = 4326) %>%
  st_transform(crs = crs(pop_raster))

# Extract population density at each fire location
population_at_fires <- raster::extract(pop_raster, as(fires_sp, "Spatial"))

# Add population data back to fires dataframe
fires_with_population <- fires_with_weather %>%
  mutate(population_density = population_at_fires)

cat("✓ Extracted population density for all fires\n")

# ============================================================================
# PART 6: CREATE FINAL FEATURES AND CLASSIFICATIONS
# ============================================================================

cat("Creating final features and classifications...\n")

fires_complete <- fires_with_population %>%
  mutate(
    # Population exposure categories
    pop_exposure = cut(
      population_density,
      breaks = c(-Inf, 0.1, 1, 10, 100, 1000, Inf),
      labels = c("Uninhabited", "Very Rural", "Rural", 
                 "Suburban", "Urban", "Dense Urban")
    ),
    
    # Population flags
    in_populated_area = population_density > 1,
    near_urban = population_density > 100,
    
    # HIGH-RISK EVENT CLASSIFICATION (Critical!)
    # Definition: Intense fire (>100 MW) + Populated area (>10 people/km²)
    high_risk_event = frp > 100 & population_density > 10
  )

# Count high-risk events
n_high_risk <- sum(fires_complete$high_risk_event, na.rm = TRUE)
cat(sprintf("✓ Identified %s high-risk events (%.2f%% of total)\n", 
            format(n_high_risk, big.mark = ","),
            100 * n_high_risk / nrow(fires_complete)))

# ============================================================================
# PART 7: SAVE PROCESSED DATA
# ============================================================================

cat("Saving processed datasets...\n")

# Create output directory if it doesn't exist
dir.create("data/processed", recursive = TRUE, showWarnings = FALSE)

# Save complete dataset
write_csv(fires_complete, "data/processed/fires_complete_CLEAN.csv")
cat("✓ Saved: fires_complete_CLEAN.csv\n")

# Create ML-ready dataset (remove NAs, select features)
fires_for_ml <- fires_complete %>%
  filter(
    # Remove any remaining NAs in critical columns
    !is.na(max_temp),
    !is.na(rainfall_mm),
    !is.na(population_density),
    !is.na(days_since_rain)
  ) %>%
  select(
    # Fire characteristics
    frp, brightness, scan, track, confidence_numeric,
    
    # Weather features
    max_temp, min_temp, temp_range, avg_temp,
    rainfall_mm, days_since_rain, rain_sum_7day, rain_sum_30day,
    extreme_heat, dry_day,
    
    # Population features
    population_density, in_populated_area, near_urban,
    
    # Temporal features
    month_num, week, acq_time,
    
    # Location
    latitude, longitude,
    
    # Source information
    source, satellite, daynight,
    
    # Target variables
    high_risk_event
  )

# Save ML-ready dataset
write_csv(fires_for_ml, "data/processed/fires_for_xgboost.csv")
cat(sprintf("✓ Saved: fires_for_xgboost.csv (%s complete cases)\n", 
            format(nrow(fires_for_ml), big.mark = ",")))

# Save high-risk events separately
high_risk_events <- fires_complete %>%
  filter(high_risk_event == TRUE)

write_csv(high_risk_events, "data/processed/high_risk_events_complete.csv")
cat(sprintf("✓ Saved: high_risk_events_complete.csv (%s events)\n", 
            format(nrow(high_risk_events), big.mark = ",")))

# Save weather data
write_csv(bom_weather_complete, "data/processed/bom_weather_complete.csv")
cat("✓ Saved: bom_weather_complete.csv\n")

# ============================================================================
# PART 8: SUMMARY STATISTICS
# ============================================================================

cat("\n" %+% strrep("=", 70) %+% "\n")
cat("DATA PROCESSING COMPLETE - SUMMARY STATISTICS\n")
cat(strrep("=", 70) %+% "\n\n")

cat("FIRE DATA:\n")
cat(sprintf("  Total fires: %s\n", format(nrow(fires_complete), big.mark = ",")))
cat(sprintf("  High-risk events: %s (%.2f%%)\n", 
            format(n_high_risk, big.mark = ","),
            100 * n_high_risk / nrow(fires_complete)))
cat(sprintf("  Date range: %s to %s\n", 
            min(fires_complete$acq_date), 
            max(fires_complete$acq_date)))

cat("\nWEATHER DATA:\n")
cat(sprintf("  Station-days: %s\n", format(nrow(bom_weather_complete), big.mark = ",")))
cat(sprintf("  Temperature range: %.1f°C to %.1f°C\n", 
            min(fires_complete$max_temp, na.rm = TRUE),
            max(fires_complete$max_temp, na.rm = TRUE)))

cat("\nPOPULATION EXPOSURE:\n")
pop_summary <- fires_complete %>%
  count(pop_exposure) %>%
  mutate(pct = 100 * n / sum(n))
for(i in 1:nrow(pop_summary)) {
  cat(sprintf("  %s: %s (%.1f%%)\n", 
              pop_summary$pop_exposure[i],
              format(pop_summary$n[i], big.mark = ","),
              pop_summary$pct[i]))
}

cat("\nML-READY DATASET:\n")
cat(sprintf("  Complete cases: %s\n", format(nrow(fires_for_ml), big.mark = ",")))
cat(sprintf("  Features: %s\n", ncol(fires_for_ml) - 1))  # -1 for target variable

cat("\n" %+% strrep("=", 70) %+% "\n")
cat("✓ All data ready for Python XGBoost modeling!\n")
cat(strrep("=", 70) %+% "\n")

# ============================================================================
# END OF SCRIPT
# ============================================================================

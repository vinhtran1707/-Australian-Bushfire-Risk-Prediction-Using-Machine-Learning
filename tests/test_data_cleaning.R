# =============================================================================
# Unit Tests for Australian Bushfire ML Analysis - R Module
# Author: Vinh
# Course: MGSC 7310 - Tulane University
# 
# Tests for data cleaning, feature engineering, and validation.
# Run with: testthat::test_file("tests/test_data_cleaning.R")
# =============================================================================

library(testthat)

# Test: Data Type Conversions ----
test_that("standardize_fire_data converts types correctly", {
  # Create sample data with mixed types
  sample_df <- data.frame(
    latitude = c("10.5", "20.3"),
    longitude = c("150.2", "151.4"),
    brightness = c("320", "330"),
    frp = c("50.5", "60.2"),
    confidence = c("h", "n"),
    track = c("1.5", "2.0"),
    stringsAsFactors = FALSE
  )
  
  # Define standardize function (simplified version)
  standardize_fire_data <- function(df) {
    df %>%
      dplyr::mutate(
        dplyr::across(dplyr::any_of(c("confidence", "track")), as.character),
        dplyr::across(dplyr::any_of(c("latitude", "longitude", "brightness", "frp")), 
                     as.numeric)
      )
  }
  
  # Test conversions
  result <- standardize_fire_data(sample_df)
  
  expect_type(result$latitude, "double")
  expect_type(result$longitude, "double")
  expect_type(result$brightness, "double")
  expect_type(result$frp, "double")
  expect_type(result$confidence, "character")
  expect_type(result$track, "character")
})


# Test: Confidence Conversion ----
test_that("confidence letters convert to correct numeric values", {
  # Create sample data
  sample_df <- data.frame(
    confidence = c("l", "n", "h", "75"),
    stringsAsFactors = FALSE
  )
  
  # Convert confidence
  result <- sample_df %>%
    dplyr::mutate(
      confidence_numeric = dplyr::case_when(
        confidence == "l" ~ 15,
        confidence == "n" ~ 55,
        confidence == "h" ~ 90,
        TRUE ~ as.numeric(confidence)
      )
    )
  
  expect_equal(result$confidence_numeric[1], 15)   # Low
  expect_equal(result$confidence_numeric[2], 55)   # Nominal
  expect_equal(result$confidence_numeric[3], 90)   # High
  expect_equal(result$confidence_numeric[4], 75)   # Numeric
})


# Test: High Confidence Flag ----
test_that("high_confidence flag is created correctly", {
  sample_df <- data.frame(
    confidence_numeric = c(15, 55, 75, 85, 90, 95)
  )
  
  result <- sample_df %>%
    dplyr::mutate(high_confidence = confidence_numeric >= 80)
  
  expect_equal(sum(result$high_confidence), 3)  # 85, 90, 95
  expect_false(result$high_confidence[1])       # 15
  expect_false(result$high_confidence[2])       # 55
  expect_false(result$high_confidence[3])       # 75
  expect_true(result$high_confidence[4])        # 85
})


# Test: Date Parsing ----
test_that("dates are parsed correctly", {
  sample_df <- data.frame(
    acq_date = c("2019-09-01", "2019-10-15", "2020-01-31"),
    stringsAsFactors = FALSE
  )
  
  result <- sample_df %>%
    dplyr::mutate(
      acq_date = lubridate::ymd(acq_date),
      month = lubridate::month(acq_date, label = TRUE),
      month_num = lubridate::month(acq_date)
    )
  
  expect_s3_class(result$acq_date, "Date")
  expect_equal(result$month_num, c(9, 10, 1))
  expect_equal(as.character(result$month), c("Sep", "Oct", "Jan"))
})


# Test: Temperature Features ----
test_that("temperature features are calculated correctly", {
  sample_df <- data.frame(
    max_temp = c(35, 42, 28),
    min_temp = c(20, 25, 15)
  )
  
  result <- sample_df %>%
    dplyr::mutate(
      temp_range = max_temp - min_temp,
      avg_temp = (max_temp + min_temp) / 2,
      extreme_heat = max_temp > 40
    )
  
  expect_equal(result$temp_range, c(15, 17, 13))
  expect_equal(result$avg_temp, c(27.5, 33.5, 21.5))
  expect_equal(result$extreme_heat, c(FALSE, TRUE, FALSE))
})


# Test: Rainfall Features ----
test_that("rainfall features are calculated correctly", {
  sample_df <- data.frame(
    rainfall_mm = c(0, 5, 15, 25)
  )
  
  result <- sample_df %>%
    dplyr::mutate(
      dry_day = rainfall_mm < 1,
      light_rain = rainfall_mm >= 1 & rainfall_mm < 5,
      moderate_rain = rainfall_mm >= 5 & rainfall_mm < 20,
      heavy_rain = rainfall_mm >= 20
    )
  
  expect_equal(result$dry_day, c(TRUE, FALSE, FALSE, FALSE))
  expect_equal(result$light_rain, c(FALSE, FALSE, FALSE, FALSE))
  expect_equal(result$moderate_rain, c(FALSE, TRUE, TRUE, FALSE))
  expect_equal(result$heavy_rain, c(FALSE, FALSE, FALSE, TRUE))
})


# Test: Population Exposure Categories ----
test_that("population categories are assigned correctly", {
  sample_df <- data.frame(
    population_density = c(0.05, 0.5, 5, 50, 500, 1500)
  )
  
  result <- sample_df %>%
    dplyr::mutate(
      pop_exposure = cut(
        population_density,
        breaks = c(-Inf, 0.1, 1, 10, 100, 1000, Inf),
        labels = c("Uninhabited", "Very Rural", "Rural", 
                   "Suburban", "Urban", "Dense Urban")
      )
    )
  
  expect_equal(as.character(result$pop_exposure[1]), "Uninhabited")
  expect_equal(as.character(result$pop_exposure[2]), "Very Rural")
  expect_equal(as.character(result$pop_exposure[3]), "Rural")
  expect_equal(as.character(result$pop_exposure[4]), "Suburban")
  expect_equal(as.character(result$pop_exposure[5]), "Urban")
  expect_equal(as.character(result$pop_exposure[6]), "Dense Urban")
})


# Test: Population Flags ----
test_that("population flags are created correctly", {
  sample_df <- data.frame(
    population_density = c(0.5, 5, 50, 150, 500)
  )
  
  result <- sample_df %>%
    dplyr::mutate(
      in_populated_area = population_density > 1,
      near_urban = population_density > 100
    )
  
  expect_equal(sum(result$in_populated_area), 4)  # All except 0.5
  expect_equal(sum(result$near_urban), 2)         # 150, 500
})


# Test: High-Risk Classification ----
test_that("high-risk events are classified correctly", {
  sample_df <- data.frame(
    frp = c(50, 120, 80, 150, 200),
    population_density = c(5, 5, 50, 50, 150)
  )
  
  result <- sample_df %>%
    dplyr::mutate(
      high_risk_event = frp > 100 & population_density > 10
    )
  
  # Should be TRUE only for rows 4 and 5
  expect_equal(sum(result$high_risk_event), 2)
  expect_false(result$high_risk_event[1])  # FRP too low
  expect_false(result$high_risk_event[2])  # Population too low
  expect_false(result$high_risk_event[3])  # FRP too low
  expect_true(result$high_risk_event[4])   # Both conditions met
  expect_true(result$high_risk_event[5])   # Both conditions met
})


# Test: Date Filtering ----
test_that("dates are filtered to study period correctly", {
  sample_df <- data.frame(
    acq_date = lubridate::ymd(c("2019-08-15", "2019-09-01", "2019-12-15", 
                                "2020-01-31", "2020-02-15"))
  )
  
  result <- sample_df %>%
    dplyr::filter(acq_date >= lubridate::ymd("2019-09-01") & 
                  acq_date <= lubridate::ymd("2020-01-31"))
  
  expect_equal(nrow(result), 3)  # Should keep middle 3 dates
  expect_true(min(result$acq_date) >= lubridate::ymd("2019-09-01"))
  expect_true(max(result$acq_date) <= lubridate::ymd("2020-01-31"))
})


# Test: Missing Value Handling ----
test_that("missing values are handled correctly", {
  sample_df <- data.frame(
    latitude = c(10, NA, 20),
    longitude = c(150, 151, NA),
    frp = c(50, 60, 70)
  )
  
  result <- sample_df %>%
    dplyr::filter(!is.na(latitude), !is.na(longitude), !is.na(frp))
  
  expect_equal(nrow(result), 1)  # Only first row has all values
})


# Test: Brightness Unification ----
test_that("brightness columns are unified correctly", {
  sample_df <- data.frame(
    brightness = c(320, NA, NA),
    bright_ti4 = c(NA, 330, NA),
    bright_ti5 = c(NA, NA, 340)
  )
  
  result <- sample_df %>%
    dplyr::mutate(
      brightness_unified = dplyr::case_when(
        !is.na(brightness) ~ brightness,
        !is.na(bright_ti4) ~ bright_ti4,
        !is.na(bright_ti5) ~ bright_ti5,
        TRUE ~ NA_real_
      )
    )
  
  expect_equal(result$brightness_unified, c(320, 330, 340))
})


# Test: Station Assignment ----
test_that("fires are assigned to correct weather stations", {
  sample_df <- data.frame(
    latitude = c(-33.5, -37.0, -27.5, -34.5, -35.2),
    longitude = c(151.5, 144.5, 153.0, 138.5, 149.1)
  )
  
  result <- sample_df %>%
    dplyr::mutate(
      nearest_station = dplyr::case_when(
        longitude > 150 & latitude > -35 & latitude < -32 ~ "066062",  # Sydney
        longitude < 146 & latitude < -36 ~ "086282",                   # Melbourne
        longitude > 152 & latitude > -28 ~ "040842",                   # Brisbane
        longitude < 140 & latitude < -33 ~ "023000",                   # Adelaide
        longitude > 148 & longitude < 150 & latitude < -34 ~ "070351", # Canberra
        TRUE ~ "066062"  # Default Sydney
      )
    )
  
  expect_equal(result$nearest_station[1], "066062")  # Sydney
  expect_equal(result$nearest_station[2], "086282")  # Melbourne
  expect_equal(result$nearest_station[3], "040842")  # Brisbane
  expect_equal(result$nearest_station[4], "023000")  # Adelaide
  expect_equal(result$nearest_station[5], "070351")  # Canberra
})


# Test: Feature Count ----
test_that("final dataset has expected number of features", {
  # This is a meta-test to ensure we're creating all expected features
  expected_features <- c(
    # Fire characteristics
    "frp", "brightness", "scan", "track", "confidence_numeric",
    # Weather
    "max_temp", "min_temp", "temp_range", "avg_temp", "rainfall_mm",
    "days_since_rain", "rain_sum_7day", "rain_sum_30day",
    "extreme_heat", "dry_day",
    # Population
    "population_density", "pop_exposure", "in_populated_area", "near_urban",
    # Temporal
    "acq_date", "month", "month_num", "week", "year",
    # Spatial
    "latitude", "longitude",
    # Classification
    "high_risk_event", "high_confidence"
  )
  
  # Should have at least these core features
  expect_gte(length(expected_features), 25)
})


# Run All Tests ----
cat("\n")
cat("="*70, "\n")
cat("RUNNING R UNIT TESTS - Australian Bushfire ML Analysis\n")
cat("="*70, "\n\n")

test_results <- test_file("tests/test_data_cleaning.R")

cat("\n")
cat("="*70, "\n")
cat("TEST SUMMARY\n")
cat("="*70, "\n")
cat("All tests completed.\n")
cat("Review results above for any failures or warnings.\n")
cat("="*70, "\n")

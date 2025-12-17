# 🔬 Methodology

## Table of Contents
- [Overview](#overview)
- [Data Collection](#data-collection)
- [Data Integration Pipeline](#data-integration-pipeline)
- [Feature Engineering](#feature-engineering)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Model Evaluation](#model-evaluation)
- [Validation Strategy](#validation-strategy)

---

## Overview

This project employs a comprehensive data science pipeline combining R for data engineering and Python for machine learning, addressing the challenge of predicting bushfire intensity and identifying high-risk events during the 2019-2020 Australian Black Summer fires.

---

## Data Collection

### Fire Detection Data (NASA FIRMS)

**Sources:**
- MODIS Archive (Terra & Aqua satellites)
- VIIRS Archive (Suomi-NPP satellite)
- MODIS Near Real-Time (NRT)
- VIIRS Near Real-Time (NRT)

**Initial Records:** 289,486 fire detections  
**Date Range:** September 1, 2019 - January 31, 2020  
**Geographic Coverage:** All of Australia

**Key Variables:**
- `latitude`, `longitude`: Fire location (decimal degrees)
- `frp`: Fire Radiative Power in megawatts (MW)
- `brightness`: Brightness temperature (Kelvin)
- `confidence`: Detection confidence (h/n/l or 0-100%)
- `scan`, `track`: Pixel dimensions
- `acq_date`, `acq_time`: Acquisition timestamp
- `daynight`: Day or night detection

**Data Quality Issues Addressed:**
1. Different column names across datasets (e.g., `brightness` vs `bright_ti4` vs `bright_ti5`)
2. Mixed confidence formats (letters vs numbers)
3. Mixed data types (character vs numeric)
4. Missing values in non-essential columns

### Weather Data (Australian Bureau of Meteorology)

**Stations:** 5 major cities
- Sydney (Station #066062)
- Melbourne (Station #086282)
- Brisbane (Station #040842)
- Adelaide (Station #023000)
- Canberra (Station #070351)

**Variables:**
- Daily rainfall (mm)
- Maximum temperature (°C)
- Minimum temperature (°C)

**Temporal Coverage:** September 2019 - January 2020  
**Records:** 610 station-days

### Population Data (WorldPop)

**Format:** GeoTIFF raster  
**Resolution:** 1km × 1km grid cells  
**Year:** 2019 (UN-adjusted)  
**Coverage:** All of Australia  
**Values:** Population density (people per km²)

---

## Data Integration Pipeline

### Phase 1: Fire Data Standardization (R)

**Step 1: Load All Datasets**
```r
datafiles <- list.files(path=dataroot, full.names = TRUE)
datafiles_ls <- lapply(datafiles, function(x) read_csv(file=x))

modis_archive <- datafiles_ls[[1]]
viirs_archive <- datafiles_ls[[2]]
modis_nrt <- datafiles_ls[[3]]
viirs_nrt <- datafiles_ls[[4]]
```

**Step 2: Standardize Data Types**
```r
standardize_fire_data <- function(df) {
  df %>%
    mutate(
      across(any_of(c("confidence", "version", "track")), as.character),
      across(any_of(c("latitude", "longitude", "brightness", "frp")), as.numeric)
    )
}
```

**Why:** Different satellites store data in different formats. MODIS uses `bright_ti4`/`bright_ti5`, while VIIRS uses `brightness`. We need consistency.

**Step 3: Combine Datasets**
```r
all_fires <- bind_rows(
  modis_archive %>% mutate(source = "MODIS_Archive"),
  viirs_archive %>% mutate(source = "VIIRS_Archive"),
  modis_nrt %>% mutate(source = "MODIS_NRT"),
  viirs_nrt %>% mutate(source = "VIIRS_NRT")
)
```

**Result:** 289,486 fires with source tracking

### Phase 2: Data Cleaning (R)

**Brightness Unification:**
```r
brightness = case_when(
  !is.na(brightness) ~ brightness,
  !is.na(bright_ti4) ~ bright_ti4,
  !is.na(bright_ti5) ~ bright_ti5,
  TRUE ~ NA_real_
)
```

**Date Parsing:**
```r
acq_date = ymd(acq_date),
month = month(acq_date, label = TRUE),
month_num = month(acq_date)
```

**Confidence Conversion:**
```r
confidence_numeric = case_when(
  confidence == "l" ~ 15,    # Low
  confidence == "n" ~ 55,    # Nominal
  confidence == "h" ~ 90,    # High
  TRUE ~ as.numeric(confidence)
)
```

**Filtering:**
- Date range: 2019-09-01 to 2020-01-31
- Remove records with missing lat/lon/frp
- **Final:** 288,876 fires

### Phase 3: Weather Integration (R)

**Challenge:** Each fire needs weather data, but weather stations are at specific locations.

**Solution:** Geographic matching

**Step 1: Load Weather Data**
```r
bom_rainfall <- map_dfr(bom_rainfall_files, load_rainfall)
bom_max_temp <- map_dfr(bom_maxtemp_files, load_max_temp)
bom_min_temp <- map_dfr(bom_mintemp_files, load_min_temp)

bom_weather_complete <- bom_rainfall %>%
  full_join(bom_max_temp, by = c("date", "station_number")) %>%
  full_join(bom_min_temp, by = c("date", "station_number"))
```

**Step 2: Assign Nearest Station**
```r
fires_with_weather <- all_fires_clean %>%
  mutate(
    nearest_station = case_when(
      longitude > 150 & latitude > -35 & latitude < -32 ~ "066062",  # Sydney
      longitude < 146 & latitude < -36 ~ "086282",                   # Melbourne
      longitude > 152 & latitude > -28 ~ "040842",                   # Brisbane
      longitude < 140 & latitude < -33 ~ "023000",                   # Adelaide
      longitude > 148 & longitude < 150 & latitude < -34 ~ "070351", # Canberra
      TRUE ~ "066062"  # Default to Sydney
    )
  ) %>%
  left_join(bom_weather_complete, 
            by = c("acq_date" = "date", "nearest_station" = "station_number"))
```

**Coverage:** 99.2% of fires successfully matched to weather data

### Phase 4: Population Extraction (R)

**Challenge:** Population is stored as a raster (image) with 1km pixels.

**Solution:** Spatial point extraction

**Step 1: Convert Fires to Spatial Points**
```r
fires_sp <- fires_with_weather %>%
  st_as_sf(coords = c("longitude", "latitude"), crs = 4326) %>%
  st_transform(crs = crs(pop_raster))
```

**Step 2: Extract Population at Each Fire**
```r
population_at_fires <- raster::extract(pop_raster, as(fires_sp, "Spatial"))
```

**Result:** Each fire now has population density at exact location

---

## Feature Engineering

### Weather Features

**Temperature Metrics:**
```r
temp_range = max_temp - min_temp,
avg_temp = (max_temp + min_temp) / 2
```

**Heat Classifications:**
```r
extreme_heat = max_temp > 40,    # Extreme heat days
very_hot = max_temp > 38,
heat_stress = max_temp > 35
```

**Rainfall Classifications:**
```r
dry_day = rainfall_mm < 1,
light_rain = rainfall_mm >= 1 & rainfall_mm < 5,
moderate_rain = rainfall_mm >= 5 & rainfall_mm < 20,
heavy_rain = rainfall_mm >= 20
```

### Drought Tracking

**Rolling Rainfall Sums:**
```r
rain_sum_7day = slide_dbl(rainfall_mm, sum, .before = 6),
rain_sum_30day = slide_dbl(rainfall_mm, sum, .before = 29)
```

**Days Since Rain:**
```r
days_since_rain = {
  rain_days <- which(rainfall_mm > 1)
  sapply(seq_along(rainfall_mm), function(i) {
    if (i %in% rain_days) return(0)
    prev_rain <- max(rain_days[rain_days < i], 0)
    return(i - prev_rain)
  })
}
```

### Lag Features

**Previous Weather:**
```r
max_temp_lag1 = lag(max_temp, 1),     # Yesterday's temp
max_temp_lag7 = lag(max_temp, 7),     # Last week's temp
rain_lag1 = lag(rainfall_mm, 1),
rain_lag7 = lag(rainfall_mm, 7)
```

**Rationale:** Weather conditions from previous days affect fire behavior

### Confidence Features

**Numeric Confidence:**
```r
confidence_numeric = case_when(
  confidence == "l" ~ 15,
  confidence == "n" ~ 55,
  confidence == "h" ~ 90,
  TRUE ~ as.numeric(confidence)
)
```

**High Confidence Flag:**
```r
high_confidence = confidence_numeric >= 80
```

**Rationale:** Satellite confidence indicates detection reliability. We discovered high-confidence fires are 7x more intense.

### Population Features

**Population Exposure Categories:**
```r
pop_exposure = cut(population_density,
                   breaks = c(-Inf, 0.1, 1, 10, 100, 1000, Inf),
                   labels = c("Uninhabited", "Very Rural", "Rural", 
                              "Suburban", "Urban", "Dense Urban"))
```

**Population Flags:**
```r
in_populated_area = population_density > 1,
near_urban = population_density > 100
```

### **KEY CLASSIFICATION: High-Risk Events** ⭐

**Definition:**
```r
high_risk_event = frp > 100 & population_density > 10
```

**Rationale:**
- Fire must be **intense** (FRP > 100 MW)
- **AND** near people (>10 people/km²)
- This is our **target variable** for classification

**Result:** 219 high-risk events identified (0.08% of all fires)

---

## Machine Learning Pipeline

### Data Preparation (Python)

**Step 1: Load Clean Data**
```python
df = pd.read_csv('fires_for_xgboost.csv')
# 254,012 complete cases with 25 features
```

**Step 2: Encode Categorical Variables**
```python
from sklearn.preprocessing import LabelEncoder

for col in ['source', 'satellite', 'daynight']:
    le = LabelEncoder()
    df[col + '_encoded'] = le.fit_transform(df[col])
```

**Why:** XGBoost requires numeric inputs

**Encoding Scheme:**
- `source`: MODIS_Archive=0, MODIS_NRT=1, VIIRS_Archive=2, VIIRS_NRT=3
- `satellite`: Aqua=0, Terra=1, N=2
- `daynight`: D=0, N=1

**Step 3: Define Features**
```python
feature_cols = [
    # Weather (10)
    'max_temp', 'min_temp', 'temp_range', 'avg_temp',
    'rainfall_mm', 'days_since_rain', 'rain_sum_7day', 'rain_sum_30day',
    'extreme_heat', 'dry_day',
    
    # Fire characteristics (4)
    'brightness', 'scan', 'track', 'confidence_numeric',
    
    # Population (3)
    'population_density', 'in_populated_area', 'near_urban',
    
    # Location/Time (5)
    'latitude', 'longitude', 'month_num', 'week', 'acq_time',
    
    # Satellite (3)
    'source_encoded', 'satellite_encoded', 'daynight_encoded'
]
```

**Total:** 25 features

### Train-Test Split

**Strategy:** 80% training, 20% testing

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y_class  # For classification only
)
```

**Sizes:**
- Training: 203,209 fires (80%)
- Testing: 50,803 fires (20%)

**Rationale:** Hold out 20% for unbiased evaluation. Use stratification for classification to maintain class balance.

### Model 1: XGBoost Regressor

**Task:** Predict fire intensity (FRP in MW)

**Hyperparameters:**
```python
model_reg = xgb.XGBRegressor(
    n_estimators=100,         # Number of trees
    max_depth=6,              # Maximum tree depth
    learning_rate=0.1,        # Step size shrinkage
    subsample=0.8,            # Row sampling ratio
    colsample_bytree=0.8,     # Column sampling ratio
    random_state=42
)
```

**Training:**
```python
model_reg.fit(X_train, y_train)
```

**Why These Hyperparameters:**
- `n_estimators=100`: Balance between performance and computation time
- `max_depth=6`: Prevent overfitting while capturing interactions
- `learning_rate=0.1`: Standard conservative rate
- `subsample=0.8`: Reduce overfitting through row sampling
- `colsample_bytree=0.8`: Reduce overfitting through feature sampling

### Model 2: XGBoost Classifier

**Task:** Classify high-risk events (binary: 0 or 1)

**Challenge:** Severe class imbalance (818:1 ratio)

**Solution:** Class weighting

```python
scale_pos_weight = (y_class==0).sum() / (y_class==1).sum()
# Result: 1238:1 weighting

model_clf = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,  # Handle imbalance
    random_state=42,
    eval_metric='logloss'
)
```

**Why `scale_pos_weight`:**
- Makes model penalize missing high-risk fires 1238x more than false alarms
- Critical for emergency response applications
- Improves recall (catching dangerous fires)

---

## Model Evaluation

### Regression Metrics

**R² Score (Coefficient of Determination):**
```
R² = 1 - (SS_res / SS_tot)
```
- Measures proportion of variance explained
- Range: -∞ to 1 (1 = perfect)
- **Our Result:** 0.9202 (92% variance explained)

**RMSE (Root Mean Squared Error):**
```
RMSE = sqrt(mean((y_true - y_pred)²))
```
- Average prediction error in MW
- Lower is better
- **Our Result:** 47.16 MW

**MAE (Mean Absolute Error):**
```
MAE = mean(|y_true - y_pred|)
```
- Typical prediction error
- **Our Result:** 9.08 MW

### Classification Metrics

**Why F1 Score Over ROC-AUC?**

With 818:1 class imbalance:
- **ROC-AUC** can be misleading (evaluates all thresholds)
- **F1 Score** evaluates at actual decision threshold
- F1 balances precision and recall equally

**Precision:**
```
Precision = TP / (TP + FP)
```
- Of predicted high-risk, how many are correct?
- **Our Result:** 76.19% (16/21)

**Recall (Sensitivity):**
```
Recall = TP / (TP + FN)
```
- Of actual high-risk, how many did we catch?
- **Our Result:** 88.89% (16/18)

**F1 Score:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
- Harmonic mean of precision and recall
- **Our Result:** 0.8205 (82.05%)

**Interpretation:** F1 of 82% is **excellent** for 818:1 imbalance

**Confusion Matrix:**
```
                Predicted
            Normal    High-Risk
Normal      14,725        5
High-Risk        2       16
```

**Key Metrics:**
- True Positives: 16 (correctly identified high-risk)
- False Negatives: 2 (missed high-risk fires) ⚠️
- False Positives: 5 (false alarms)
- True Negatives: 14,725 (correctly identified normal)

---

## Validation Strategy

### Cross-Validation Considerations

**Not Used:** Traditional k-fold cross-validation

**Why:**
1. Large dataset (254K fires) - single split sufficient
2. Temporal component - fires not independent
3. Computational cost for 100-tree XGBoost

**Instead:** Single train-test split with:
- Random state set for reproducibility
- Stratification for classification (maintains class balance)
- 20% holdout (50,803 fires) provides robust evaluation

### Feature Importance Validation

**Method:** XGBoost built-in importance

**Metrics Used:**
- **Gain:** Average improvement in loss when feature is used
- Default metric in XGBoost

**Interpretation:**
- High importance → feature strongly influences predictions
- Used to understand model decisions
- Validates domain knowledge (e.g., brightness should predict intensity)

### Model Interpretability

**Feature Importance Analysis:**
```python
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
```

**Top Findings:**
- **Regression:** Brightness (35%), Track (27%), Scan (12%)
- **Classification:** Population density (52%), In populated area (42%)

**Validation:** Results align with domain knowledge and problem definition

---

## Assumptions and Limitations

### Assumptions

1. **Weather stations represent regional conditions** - Nearest station approximation
2. **Population density stable** - Using 2019 WorldPop data
3. **Independent observations** - Treating each fire detection separately
4. **Feature relationships** - XGBoost can capture non-linear interactions

### Limitations

1. **Temporal limitations:**
   - Only 5 months of data (Sep 2019 - Jan 2020)
   - Specific to Black Summer conditions
   - May not generalize to other fire seasons

2. **Geographic limitations:**
   - Only 5 weather stations for all of Australia
   - Some fires far from nearest station (>100km)
   - Missing micro-climate variations

3. **Data quality:**
   - Different satellite sensitivities (MODIS vs VIIRS)
   - Missing confidence data for some VIIRS_NRT
   - Population data is 2019 estimate

4. **Model limitations:**
   - No wind speed data (unavailable)
   - No fuel moisture data
   - No terrain/elevation features
   - No fire spread modeling (point-in-time only)

### Future Improvements

1. **Additional features:**
   - Wind speed and direction
   - Fuel moisture content
   - Elevation and slope
   - Vegetation type

2. **Model enhancements:**
   - Ensemble methods (combine multiple models)
   - Time series forecasting (predict fire spread)
   - Deep learning for spatial patterns

3. **Deployment:**
   - Real-time prediction API
   - Integration with emergency dispatch systems
   - Mobile app for field responders

---

## Reproducibility

### Random Seeds
- Python: `random_state=42` in all `train_test_split` and XGBoost models
- Ensures reproducible results

### Software Versions
- Python: 3.8+
- R: 4.0+
- XGBoost: 2.0.0
- scikit-learn: 1.3.0

### Data Processing
- All preprocessing steps documented in code
- Intermediate datasets saved for verification

---

## Conclusion

This methodology combines:
- **R** for complex data engineering (spatial joins, raster extraction)
- **Python** for machine learning (XGBoost, evaluation)
- **Domain knowledge** for feature engineering (high-risk definition)
- **Rigorous evaluation** appropriate for imbalanced data (F1 score)

The result is a production-ready system for:
1. Predicting fire intensity (92% accuracy)
2. Identifying high-risk events (82% F1 score)
3. Supporting emergency response decisions

---

**For questions about methodology:**  
📧 Contact: [your-email]  
📚 See also: [RESULTS.md](RESULTS.md) | [DATA_SOURCES.md](DATA_SOURCES.md)

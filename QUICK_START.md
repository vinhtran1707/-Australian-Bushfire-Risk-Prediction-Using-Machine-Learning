# ⚡ Quick Start Guide

Get the Australian Bushfire ML Analysis project running in **under 15 minutes**.

---

## 🎯 For Employers / Recruiters

**Want to see what this project does quickly?**

1. **View Results First:** Check the [results/figures/](results/figures/) directory for visualizations
2. **Read Key Files:**
   - [README.md](README.md) - Project overview
   - [RESULTS.md](docs/RESULTS.md) - Detailed findings
3. **Review Code:**
   - [src/data_cleaning.R](src/data_cleaning.R) - R data engineering
   - [notebooks/02_xgboost_modeling.ipynb](notebooks/02_xgboost_modeling.ipynb) - Python ML
4. **See Models:** [results/models/](results/models/) - Trained XGBoost models

**Time Investment:** 10-15 minutes to review

---

## 💻 For Developers - Full Setup

### Prerequisites Check

```bash
# Check Python version (need 3.8+)
python --version

# Check R version (need 4.0+)
R --version

# Check Git
git --version
```

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/australian-bushfire-ml-analysis.git
cd australian-bushfire-ml-analysis
```

### 2. Python Environment Setup

**Option A: Using Conda (Recommended)**
```bash
conda env create -f environment.yml
conda activate bushfire-analysis
```

**Option B: Using pip**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. R Packages Setup

```r
# Open R or RStudio and run:
install.packages(c(
  "tidyverse", "lubridate", "sf", "raster",
  "leaflet", "plotly", "viridis", "corrplot",
  "scales", "patchwork", "slider"
))
```

### 4. Verify Installation

**Python:**
```python
import pandas as pd
import xgboost as xgb
print("✅ All Python packages loaded!")
```

**R:**
```r
library(tidyverse)
library(sf)
library(raster)
cat("✅ All R packages loaded!\n")
```

---

## 📊 Running the Analysis

### Option 1: Quick Demo (Recommended First)

**Use Pre-processed Data:**
```python
# In Python/Jupyter
import pandas as pd

# Load clean data
df = pd.read_csv('data/processed/fires_for_xgboost.csv')
print(f"Loaded {len(df):,} fires")

# Load pre-trained models
import xgboost as xgb
model_reg = xgb.XGBRegressor()
model_reg.load_model('results/models/xgboost_frp_regression.json')

# Make prediction
sample_fire = df.iloc[0:1]
predicted_frp = model_reg.predict(sample_fire[feature_cols])[0]
print(f"Predicted FRP: {predicted_frp:.1f} MW")
```

**Time:** 2 minutes

### Option 2: View Jupyter Notebooks

```bash
jupyter notebook

# Open and run:
# 1. notebooks/01_data_exploration.ipynb
# 2. notebooks/02_xgboost_modeling.ipynb
# 3. notebooks/03_results_analysis.ipynb
```

**Time:** 10-30 minutes (depending on exploration)

### Option 3: Full Pipeline (Requires Raw Data)

**Step 1: Download Data**
- NASA FIRMS: https://firms.modaps.eosdis.nasa.gov/
- BOM Weather: http://www.bom.gov.au/climate/data/
- WorldPop: https://www.worldpop.org/

Place in `data/raw/`

**Step 2: Run R Data Cleaning**
```r
# In RStudio
source("src/data_cleaning.R")
```

**Step 3: Run Python ML Pipeline**
```bash
python src/python/train_model.py
python src/python/evaluate_model.py
```

**Time:** 45-60 minutes (including data download)

---

## 🎓 Understanding the Code

### R Code Structure

```r
# data_cleaning.R
1. Load fire datasets (MODIS/VIIRS) ────► bind_rows()
2. Standardize formats ─────────────────► mutate(), across()
3. Load weather (BOM) ──────────────────► read_csv(), full_join()
4. Match fires to stations ─────────────► case_when(), left_join()
5. Extract population ──────────────────► raster::extract()
6. Create features ─────────────────────► Feature engineering
7. Output clean data ───────────────────► write_csv()
```

**Key Techniques:**
- `tidyverse` for data manipulation
- `sf` and `raster` for spatial operations
- `slider` for rolling calculations

### Python Code Structure

```python
# train_model.py
1. Load clean data ─────────────────────► pd.read_csv()
2. Encode categories ───────────────────► LabelEncoder()
3. Train/test split ────────────────────► train_test_split()
4. Train XGBoost ───────────────────────► XGBRegressor/Classifier()
5. Evaluate ────────────────────────────► metrics from sklearn
6. Save models ─────────────────────────► model.save_model()
```

**Key Techniques:**
- `pandas` for data manipulation
- `xgboost` for ML models
- `scikit-learn` for evaluation

---

## 📈 Key Results to Show

### Model Performance

**Regression (Predict Fire Intensity):**
- R² = 0.92 (92% variance explained)
- RMSE = 47.16 MW
- Top feature: Brightness (35%)

**Classification (Identify High-Risk):**
- F1 Score = 0.82 (excellent for imbalance)
- Recall = 88.9% (caught 16/18)
- Precision = 76.2%

### Important Findings

1. **High-confidence fires are 7x more intense** (139 MW vs 19.8 MW)
2. **VIIRS is 6x more reliable** than MODIS
3. **219 high-risk events** identified from 288,876 fires
4. **Population density is #1 predictor** of high-risk (52%)

---

## 🖼️ Visualizations to Highlight

**Must-See Figures:**
1. `xgb_actual_vs_predicted.png` - Model accuracy visualization
2. `xgb_confusion_matrix.png` - Classification performance
3. `fire_density_heatmap.png` - Geographic distribution
4. `high_risk_weather_intensity.png` - Risk factors

**Location:** `results/figures/`

---

## 🛠️ Troubleshooting

### Python Issues

**"Module not found"**
```bash
pip install <missing-package>
```

**"XGBoost not working"**
```bash
# Reinstall with:
pip uninstall xgboost
pip install xgboost==2.0.0
```

### R Issues

**"Package not available"**
```r
install.packages("<package-name>")
```

**"rgdal/sf errors"**
```r
# On Ubuntu/Linux:
# sudo apt-get install libgdal-dev libproj-dev

# Then install in R:
install.packages("sf")
```

### Data Issues

**"File not found"**
- Check file paths in scripts
- Ensure data is in `data/raw/` or `data/processed/`
- Update paths to match your system

---

## 📚 Learning Resources

### For R Beginners
- [R for Data Science](https://r4ds.had.co.nz/) - Free online book
- [RStudio Cheatsheets](https://www.rstudio.com/resources/cheatsheets/)

### For Python/ML Beginners
- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

### For Spatial Data
- [Geocomputation with R](https://geocompr.robinlovelace.net/)
- [Introduction to GIS in R](https://www.jessesadler.com/post/gis-with-r-intro/)

---

## 🎯 Next Steps After Setup

1. **Explore the data:**
   - Run `notebooks/01_data_exploration.ipynb`
   - Look at summary statistics
   - Visualize distributions

2. **Review the models:**
   - Check feature importance
   - Understand predictions
   - Evaluate metrics

3. **Experiment:**
   - Try different hyperparameters
   - Add new features
   - Test on different subsets

4. **Extend the project:**
   - Add wind data
   - Implement fire spread prediction
   - Create real-time dashboard

---

## 💡 Tips for Employers

**Assessing This Project:**

✅ **Strong Points to Note:**
- End-to-end pipeline (R → Python)
- Real-world data integration (4 sources)
- Proper handling of imbalanced data
- Production-ready models (JSON format)
- Comprehensive documentation
- Clean, reproducible code

**Questions to Ask Candidate:**
1. "Why did you use F1 score instead of ROC-AUC?"
2. "How did you handle the class imbalance?"
3. "What would you add with more time/data?"
4. "How would you deploy this to production?"

---

## 📞 Need Help?

**Issues with setup:**
- Check [GitHub Issues](https://github.com/yourusername/australian-bushfire-ml-analysis/issues)
- Review [METHODOLOGY.md](docs/METHODOLOGY.md)

**Have questions about the project:**
- Email: your.email@example.com
- LinkedIn: [your-profile](https://linkedin.com/in/yourprofile)

**Want to contribute:**
- Fork the repository
- Make improvements
- Submit a pull request

---

## ⏱️ Time Estimates

| Task | Time | For Whom |
|------|------|----------|
| Review README + results | 10-15 min | Recruiters |
| Setup environment | 5-10 min | Developers |
| Run pre-trained models | 2 min | Quick demo |
| Full Jupyter walkthrough | 20-30 min | Technical review |
| Complete pipeline from scratch | 45-60 min | Deep dive |

---

<div align="center">

**Ready to explore bushfire prediction with machine learning!** 🔥📊

[⬆️ Back to Top](#quick-start-guide)

</div>

# Tests Directory

Unit tests for the Australian Bushfire ML Analysis project.

## 📁 Test Files

| File | Language | Tests | Description |
|------|----------|-------|-------------|
| `test_data_cleaning.R` | R | 15 tests | Data cleaning, feature engineering, validation |
| `test_predictions.py` | Python | 18+ tests | Model loading, predictions, data validation |

---

## 🧪 Running Tests

### Python Tests

**Requirements:**
```bash
pip install pytest unittest
```

**Run all tests:**
```bash
# From project root
pytest tests/test_predictions.py -v

# Or using unittest
python tests/test_predictions.py
```

**Run specific test class:**
```bash
pytest tests/test_predictions.py::TestModelLoading -v
```

**Expected output:**
```
==================== RUNNING UNIT TESTS ====================
test_regression_model_exists ... ok
test_classification_model_exists ... ok
test_load_regression_model ... ok
...
==================== TEST SUMMARY ====================
Tests run: 18
Successes: 16
Failures: 0
Errors: 0
Skipped: 2
```

### R Tests

**Requirements:**
```r
install.packages("testthat")
```

**Run all tests:**
```r
# From R console
library(testthat)
test_file("tests/test_data_cleaning.R")

# Or from command line
Rscript -e "testthat::test_file('tests/test_data_cleaning.R')"
```

**Expected output:**
```
==================== RUNNING R UNIT TESTS ====================
✓ | OK F W S | Context
✓ | 15       | test_data_cleaning
```

---

## 📊 Test Coverage

### Python Tests Cover:

**Model Loading (4 tests):**
- ✓ Regression model file exists
- ✓ Classification model file exists
- ✓ Models load correctly
- ✓ Models have correct hyperparameters

**Data Validation (3 tests):**
- ✓ All 25 required features defined
- ✓ Feature data types are correct
- ✓ Feature values in reasonable ranges

**Predictions (5 tests):**
- ✓ Regression predictions return correct shape
- ✓ Classification predictions return correct shape
- ✓ Predictions are within reasonable ranges
- ✓ Probabilities sum to 1
- ✓ Binary predictions are 0 or 1

**Predict Function (2 tests):**
- ✓ Function can be imported
- ✓ Function works with valid input

**File Structure (2 tests):**
- ✓ Required directories exist
- ✓ Model files exist

### R Tests Cover:

**Data Type Conversions (4 tests):**
- ✓ Numeric conversions work correctly
- ✓ Character conversions work correctly
- ✓ Confidence letters → numbers
- ✓ High confidence flags created

**Feature Engineering (6 tests):**
- ✓ Date parsing works correctly
- ✓ Temperature features calculated
- ✓ Rainfall features calculated
- ✓ Population categories assigned
- ✓ Population flags created
- ✓ Brightness unification works

**Classification Logic (2 tests):**
- ✓ High-risk events classified correctly
- ✓ Study period filtering works

**Data Quality (3 tests):**
- ✓ Missing values handled
- ✓ Station assignment works
- ✓ Expected features present

---

## ✅ Test Checklist

Before pushing to GitHub, ensure:

- [ ] Python tests pass
- [ ] R tests pass
- [ ] No skipped tests (or skips are documented)
- [ ] Test coverage > 80%
- [ ] All critical functions tested
- [ ] Edge cases covered

---

## 🐛 Troubleshooting

### Python Tests

**"ModuleNotFoundError: No module named 'xgboost'"**
```bash
pip install xgboost
```

**"FileNotFoundError: Model file not found"**
- Ensure models are in `results/models/`
- Train models first with `python src/python/train_model.py`

**Tests are skipped**
- Some tests skip if models aren't available
- This is normal if models haven't been trained yet

### R Tests

**"Error: could not find function 'test_that'"**
```r
install.packages("testthat")
library(testthat)
```

**"Error: object 'dplyr' not found"**
```r
install.packages("tidyverse")
library(dplyr)
```

---

## 🔧 Continuous Integration

These tests can be integrated with CI/CD pipelines:

### GitHub Actions Example:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.8
      
      - name: Install Python dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest
      
      - name: Run Python tests
        run: pytest tests/test_predictions.py
      
      - name: Set up R
        uses: r-lib/actions/setup-r@v2
      
      - name: Install R dependencies
        run: Rscript -e "install.packages('testthat')"
      
      - name: Run R tests
        run: Rscript -e "testthat::test_file('tests/test_data_cleaning.R')"
```

---

## 📝 Adding New Tests

### Python Test Template:

```python
class TestNewFeature(unittest.TestCase):
    """Test description"""
    
    def test_something(self):
        """Test that something works"""
        result = my_function(input_data)
        self.assertEqual(result, expected_value)
```

### R Test Template:

```r
test_that("description of what you're testing", {
  # Arrange
  input_data <- data.frame(...)
  
  # Act
  result <- my_function(input_data)
  
  # Assert
  expect_equal(result$value, expected_value)
})
```

---

## 📚 Resources

**Python Testing:**
- [pytest documentation](https://docs.pytest.org/)
- [unittest documentation](https://docs.python.org/3/library/unittest.html)

**R Testing:**
- [testthat documentation](https://testthat.r-lib.org/)
- [R Packages Testing chapter](https://r-pkgs.org/testing-basics.html)

---

## 💡 Best Practices

1. **Write tests first** (Test-Driven Development)
2. **Test edge cases** (empty data, null values, extreme values)
3. **Keep tests independent** (each test should run alone)
4. **Use descriptive names** (test_that_function_does_X)
5. **Test one thing per test** (don't combine multiple assertions)
6. **Mock external dependencies** (don't rely on network/files)
7. **Keep tests fast** (< 1 second per test)

---

**Last Updated:** December 2024  
**Test Framework:** pytest (Python), testthat (R)  
**Total Tests:** 33+

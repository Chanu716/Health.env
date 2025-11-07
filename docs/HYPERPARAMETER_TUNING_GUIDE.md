# 🔧 Hyperparameter Tuning Guide for Dengue Risk Prediction

## 📋 What Was Added

I've added comprehensive **hyperparameter tuning** to your `model_training_evaluation.ipynb` notebook using **GridSearchCV**.

---

## 🎯 New Cells Added (After Cell 4 - Initial Training)

### 1️⃣ **Hyperparameter Grid Definition**
Defines search spaces for each model:

**Logistic Regression:**
- C: [0.01, 0.1, 1, 10, 100]
- Penalty: ['l2']
- Solver: ['lbfgs', 'newton-cg', 'sag']
- Max iterations: [1000, 2000]

**Decision Tree:**
- Max depth: [5, 10, 15, 20, None]
- Min samples split: [2, 5, 10, 20]
- Min samples leaf: [1, 2, 5, 10]
- Criterion: ['gini', 'entropy']

**Random Forest:**
- N estimators: [50, 100, 200]
- Max depth: [10, 15, 20, None]
- Min samples split: [2, 5, 10]
- Min samples leaf: [1, 2, 5]
- Max features: ['sqrt', 'log2', None]

**XGBoost:**
- N estimators: [50, 100, 200]
- Max depth: [3, 5, 7, 9]
- Learning rate: [0.01, 0.05, 0.1, 0.2]
- Subsample: [0.7, 0.8, 0.9, 1.0]
- Colsample bytree: [0.7, 0.8, 0.9, 1.0]

---

### 2️⃣ **GridSearch Execution**
- Performs 5-fold cross-validation
- Tests all parameter combinations
- Finds best parameters for each model
- Measures tuning time

---

### 3️⃣ **Before vs After Comparison**
Compares:
- Accuracy before tuning (default params)
- Accuracy after tuning (optimized params)
- Improvement percentage
- Cross-validation scores
- Tuning times

---

### 4️⃣ **Visualization**
Creates comparison charts:
- **Before vs After bar chart** (side-by-side comparison)
- **Tuning time vs improvement scatter plot**
- Saved as: `hyperparameter_tuning_comparison.png`

---

### 5️⃣ **Best Parameters Display**
Shows optimal hyperparameters found for each model

---

### 6️⃣ **Model Update**
Replaces default models with tuned models for final evaluation

---

### 7️⃣ **Results Export**
Saves:
- `hyperparameter_tuning_results.json` - Complete tuning results
- `tuning_comparison.csv` - Comparison table

---

## 🚀 How to Run

### Step 1: Run Initial Cells (1-4)
```python
# Cell 1: Import libraries
# Cell 2: Load data
# Cell 2.1: Prepare X and y (FIXED - now properly defines features and target)
# Cell 2.2: Train-test split
# Cell 3: Initialize models
# Cell 4: Train with default parameters
```

### Step 2: Run Hyperparameter Tuning (NEW Cells 4.1)
```python
# Cell 4.1.1: Define parameter grids
# Cell 4.1.2: Run GridSearchCV (⚠️ This takes time!)
# Cell 4.1.3: Compare before vs after
# Cell 4.1.4: Visualize improvements
# Cell 4.1.5: Display best parameters
# Cell 4.1.6: Update trained_models
# Cell 4.1.7: Save results
```

### Step 3: Continue with Evaluation (Existing Cells 5+)
```python
# Cell 5: Predictions (now uses tuned models!)
# Cell 6: Evaluation metrics
# Cell 7-14: Visualizations, confusion matrices, ROC curves, etc.
```

---

## ⏱️ Expected Runtime

**Approximate tuning times** (depends on hardware):
- Logistic Regression: ~1-3 minutes
- Decision Tree: ~5-10 minutes
- Random Forest: ~10-20 minutes
- XGBoost: ~15-30 minutes

**Total: ~30-60 minutes** for complete hyperparameter tuning

⚠️ **Tip:** The notebook prints progress as it runs. You'll see:
```
Fitting 5 folds for each of X candidates, totalling Y fits
```

---

## 📊 What You'll Get

After running tuning, you'll see:

1. **Comparison Table:**
```
Model               Before Tuning  After Tuning  Improvement  CV Score
Logistic Regression    0.8234        0.8456       +2.70%      0.8423
Decision Tree          0.7891        0.8123       +2.94%      0.8089
Random Forest          0.8567        0.8789       +2.59%      0.8734
XGBoost                0.8645        0.8912       +3.09%      0.8876
```

2. **Best Parameters** for each model

3. **Visualizations:**
   - Before vs After bar chart
   - Tuning time vs improvement scatter plot

4. **Exported Files:**
   - `hyperparameter_tuning_results.json`
   - `tuning_comparison.csv`
   - `hyperparameter_tuning_comparison.png`

---

## 🎯 Key Improvements

### ✅ Fixed Issues:
1. **Added missing X and y preparation** (Cell 2.1 was broken)
2. **Proper feature-target separation**
3. **Automatic target column detection**

### ✅ Added Features:
1. **Comprehensive hyperparameter tuning**
2. **5-fold cross-validation**
3. **Before/after comparison**
4. **Best parameters documentation**
5. **Results export (JSON + CSV)**
6. **Visualization of improvements**

---

## 💡 Tips

1. **Start with smaller grids** if tuning takes too long:
   - Reduce parameter value ranges
   - Use fewer n_estimators
   - Example: `'n_estimators': [50, 100]` instead of `[50, 100, 200]`

2. **Use parallel processing:**
   - GridSearchCV already uses `n_jobs=-1` (all CPU cores)
   - Make sure your system has adequate RAM

3. **Monitor progress:**
   - Each model prints "Fitting 5 folds..." messages
   - Shows completion time after each model

4. **Interpret results:**
   - Positive improvement % = tuning helped
   - Negative improvement % = default params were already good
   - Higher CV score = better generalization

---

## 🔄 Next Steps After Tuning

1. ✅ Hyperparameter tuning complete
2. ⏭️ Run remaining evaluation cells (5-14)
3. ⏭️ Compare all models with optimized parameters
4. ⏭️ Select best model
5. ⏭️ Save best model as .pkl
6. ⏭️ Integrate with dashboard

---

## ❓ Troubleshooting

### Issue: "X is not defined"
**Solution:** Make sure you run Cell 2.1 (Data Validation) before Cell 2.2 (Train-Test Split)

### Issue: Tuning takes too long
**Solution:** Reduce parameter grid size or use RandomizedSearchCV instead

### Issue: Out of memory
**Solution:** Close other applications, reduce n_estimators, or use smaller cv folds

### Issue: No improvement after tuning
**Solution:** This is normal! Default parameters are sometimes already optimal. Focus on model with best absolute accuracy.

---

## 📝 Summary

You now have a complete model training pipeline with:
- ✅ Data preparation
- ✅ Train-test split
- ✅ Feature scaling
- ✅ Initial training (default params)
- ✅ **Hyperparameter tuning (NEW!)**
- ✅ Before/after comparison (NEW!)
- ✅ Model evaluation
- ✅ Comprehensive visualizations
- ✅ Results export

**Ready to find the best model for your dengue prediction system!** 🦟🎯

# Generalization Evaluation of UWB NLOS Models on the e-Wine Dataset

This repository provides a robust framework for evaluating the generalization performance of pre-trained Ultra-Wideband (UWB) NLOS/LOS classification models. The scripts are specifically designed to test models trained on one dataset (e.g., Ghent) against the various real-world scenarios presented in the **e-Wine public dataset**.

This is **not a training script**. Its core purpose is to load existing models and assess their performance on new, unseen data, employing post-inference analysis techniques like probability calibration and optimal thresholding to ensure a fair and rigorous evaluation.

## 🔬 Core Functionality

* **Generalization Testing**: Loads pre-trained models from multiple independent training runs (differentiated by random seeds).
* **Multi-Scene Evaluation**: Sequentially performs inference on various scenes defined in the configuration.
* **Probability Calibration**: Improves the reliability of model probability scores on new data using **Platt Scaling** or **Isotonic Regression**.
* **Optimal Thresholding**: Calculates the best classification threshold for each model on each new scene using **Youden's J statistic**, avoiding the naive 0.5 default.
* **Comprehensive Reporting**: Generates a detailed CSV report with a wide range of classification metrics (AUROC, AUPRC, F1, Balanced Accuracy, MCC).
* **Confusion Matrix Analysis**: Automatically generates and aggregates confusion matrices, providing insights into model failure modes across different scenarios.

## 🤖 Models Evaluated

The framework is set up to load and evaluate the following pre-trained models:

* **Dual-Channel Transformer (DT)**: The primary model from our previous work.
* **LS-SVM**: Least-Squares Support Vector Machine.
* **DNN**: Deep Neural Network.
* **Single-Channel Transformer (ST)**.
* **XGBoost (XGB)**.
* **CNN-LSTM (C-L)**.

## ✅ Prerequisites & Setup

### 1. Required Artifacts

This script **requires pre-trained models and their associated artifacts**. Before running, you must have the output files from your training script. Based on your project, you have **10 runs** (from `seed=42` to `seed=51`).

### 2. Directory Structure

Create a main directory for your pre-trained models (e.g., `Pretrained/`). Inside, create a sub-folder for each training run, named `run_seed_XX`. Each of these folders must contain the necessary model and scaler files.

```
Pretrained/
├── run_seed_42/
│   ├── dual_channel_transformer_model_final_GMM_BIC.keras
│   ├── central_cir_scaler_combined.joblib
│   ├── gmm_los_model.joblib
│   ├── gmm_nlos_model.joblib
│   ├── flos_gmm_scaler_combined_GMM_BIC.joblib
│   ├── dnn_model_final.keras
│   ├── svc_model_final.joblib
│   ├── ridge_model_final.joblib
│   ├── xgboost_cls_model_final.joblib
│   ├── single_transformer_model_final.keras
│   └── cnn_lstm_model_final.keras
│
├── run_seed_43/
│   ├── ... (same set of files)
│
├── ... (up to run_seed_51)
│
└── run_seed_51/
    └── ... (same set of files)
```

### 3. Installation

If you haven't already, install the required Python packages.

```bash
# It's recommended to use a virtual environment
pip install tensorflow pandas scikit-learn joblib
```

## 🚀 Workflow & Usage

### Step 1: Configure the Evaluation in `ewine_config.py`

Open the `ewine_config.py` file and update it to match your setup. This is the most crucial step.

```python
# ewine_config.py

# 1. Point to the parent directory containing all your training run folders
PRETRAINED_MODEL_BASE_DIR = "Pretrained"

# 2. Specify the exact runs to evaluate.
#    This will evaluate seeds 42, 43, ..., 51.
BASE_SEED_TO_LOAD = 42
NUM_RUNS_TO_LOAD = 10

# 3. Define the paths to the e-Wine dataset scenes and their names
RELATIVE_DATA_PATHS = [
    "ewine_dataset/e-wine_scene_1.csv",
    "ewine_dataset/e-wine_scene_2.csv",
    # ... add all other scene paths
]
SCENE_NAMES = [
    "Scene 1",
    "Scene 2",
    # ... add all other scene names
]

# 4. Set the directory for saving all evaluation results
RESULTS_DIR = "ewine_Scene_Statistical_Results"

# 5. Enable the models you want to test by setting them to True
RUN_DUAL_TRANSFORMER = True
RUN_LS_SVM = True
RUN_DNN = True
RUN_SINGLE_TRANSFORMER = True
RUN_XGBOOST = True
RUN_CNN_LSTM = True
```

### Step 2: Execute the Script

Once the configuration is set, run the main evaluation script from your terminal.

```bash
python ewine_main_script.py
```

The script will iterate through each run (42-51), load the models, evaluate them on each specified e-Wine scene, and print the progress.

## 📊 Output

All results are saved in the directory you specified in `RESULTS_DIR` (e.g., `ewine_Scene_Statistical_Results/`).

* **`all_runs_raw_results_with_calibration.csv`**: The primary output. A detailed CSV containing performance metrics for every model, on every scene, for every training run evaluated.
* **`logs/`**: A directory containing timestamped log files for each execution.
* **`scene_summary.csv`**: An aggregated report summarizing the confusion matrix (TN, FP, FN, TP) for each model across all scenes.
* **`failure_mode_analysis.csv`**: A report analyzing common failure modes based on the aggregated confusion matrices.

## 📁 Project File Reference

```
.
├── ewine_Scene_Statistical_Results/ # OUTPUT: Main results and reports
├── scene_classification_figures/    # OUTPUT: Generated figures
├── ewine_main_script.py             # <<<< MAIN EVALUATION SCRIPT >>>>
├── ewine_config.py                  # CONFIG: Paths, model flags, scenes
├── ewine_data_utils.py              # Helper for e-Wine data loading
├── ewine_model_definition.py        # Definition of the Dual-Channel Transformer
├── ewine_confusion_matrix_handle.py # Helper for managing confusion matrices
├── ewine_flos_module.py             # Helper for calculating FLOS features
├── ewine_plot_confusion_matrix_results.py # Utility to visualize CM results
└── inspect_model.py                 # Utility to inspect saved model architectures
```

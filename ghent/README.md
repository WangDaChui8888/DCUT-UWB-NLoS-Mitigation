# UWB NLOS Identification and Error Mitigation with Deep Learning (Ghent Dataset)

This repository contains the complete Python source code for the research project on Ultra-Wideband (UWB) Non-Line-of-Sight (NLOS) identification and ranging error mitigation, evaluated on the Ghent public UWB dataset. The core of this project is a novel **Dual-Channel UWB Transformer (DCUT)** model, which is benchmarked against several standard machine learning and deep learning models.

## ✨ Features

* **Novel Model**: Implementation of a **Dual-Channel Transformer** that uniquely processes raw Channel Impulse Response (CIR) features and statistical Fading Likelihood of Sight (FLOS) features in parallel channels.
* **Comprehensive Benchmarking**: Includes implementations and fair comparisons with six baseline models:
    * Support Vector Machine (SVM) with Ridge Regression
    * Deep Neural Network (DNN)
    * Single-Channel Transformer
    * XGBoost
    * CNN-LSTM
* **End-to-End Workflow**: The main script (`main_script.py`) handles the entire pipeline from data loading and cleaning to cross-validation, final model training, evaluation, and saving artifacts.
* **Reproducibility**: Uses global random seeds and structured logging to ensure that experiments can be reproduced.
* **Modular & Extensible**: The code is organized into logical modules for data processing, evaluation, and individual model experiments, making it easy to extend or modify.
* **Advanced Experimentation**: Contains scripts and a structured framework for conducting hyperparameter tuning (`Tuning_Runs`) and ablation studies (`Ablation_Runs`).

## 📁 Project Structure

The repository is organized to separate configuration, data utilities, model definitions, experiments, and analysis scripts.

```
.
├── Ablation_Runs/            # Output directory for ablation study results
├── Tuning_Runs/              # Output directory for hyperparameter tuning runs
├── 0_analyze_results.py      # Script to analyze and aggregate results from runs
├── 3_plot_heatmap.py         # Script to generate heatmaps for analysis
├── 3_run_ablation_study.py   # Main script to orchestrate an ablation study
├── 4_run_hyper_tuning_baselines.py # Script to run hyperparameter tuning for baseline models
├── config.py                 # Main configuration file (paths, hyperparameters)
├── data_utils.py             # Functions for loading, cleaning, and splitting data
├── evaluation_utils.py       # Functions for calculating classification/regression metrics
├── flos_module.py            # Module for calculating FLOS features using GMM
├── main_script.py            # <<<< MAIN ENTRY POINT FOR EXPERIMENTS >>>>
├── model_definition.py       # Definition of the Dual-Channel Transformer model
├── svm_experiment.py         # Experiment script for the SVM+Ridge model
├── dnn_experiment.py         # Experiment script for the DNN model
├── xgboost_experiment.py     # Experiment script for the XGBoost model
├── cnn_lstm_experiment.py    # Experiment script for the CNN-LSTM model
└── single_transformer_experiment.py # Experiment script for the Single-Channel Transformer
```

## ⚙️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/Comparision_transformer_prj.git](https://github.com/your-username/Comparision_transformer_prj.git)
    cd Comparision_transformer_prj
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    A `requirements.txt` file should be created with the following contents.
    ```
    tensorflow==2.x.x
    pandas
    numpy
    scikit-learn
    xgboost
    joblib
    scipy
    ```
    Install them using pip:
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage

The primary script for running experiments is `main_script.py`.

### 1. Configuration

Before running, configure your experiment in `config.py` (or `config_v7.py`). This is where you set:
* Paths to the Ghent dataset (`RELATIVE_DATA_PATHS`).
* The main directory for saving results (`SAVE_DIRECTORY`).
* Hyperparameters for all models.
* Training settings like epochs, batch size, and K-folds for cross-validation.

### 2. Running the Main Experiment

To run the full experiment (cross-validation and final testing for all enabled models), execute the main script.

```bash
python main_script.py --seed 42 --save_dir ./Ablation_Runs/My_First_Run
```

#### Command-Line Arguments

You can override the settings from `config.py` using command-line arguments.

* `--seed`: Set the global random seed for reproducibility.
    * Example: `--seed 42`
* `--save_dir`: Specify the directory to save all outputs (logs, models, results).
    * Example: `--save_dir ./Tuning_Runs/XGBoost_Tuning`
* `--run_only`: Run the experiment for only one specified model. This is useful for debugging or tuning a single model.
    * Choices: `DualTransformer`, `SVM`, `DNN`, `SingleTransformer`, `XGBoost`, `CNNLSTM`
    * Example: `python main_script.py --run_only XGBoost`
* **Hyperparameter Overrides**: Directly set key hyperparameters for quick tests.
    * Example: `python main_script.py --run_only DualTransformer --learning_rate 0.0005 --transformer_layers 3`

### 3. Output

All results and artifacts are saved in the directory specified by `--save_dir`. This includes:
* **`execution_main.log`**: A detailed log of the entire run.
* **`summary.csv`**: A summary table comparing the cross-validation and final test metrics for all models.
* **Final Models**: Saved models for each algorithm (`.h5` for TF models, `.joblib` for others).
* **Scalers**: `StandardScaler` and FLOS GMM models are saved as `.joblib` files.
* **Raw Predictions**: Test set predictions for each model are saved as `.mat` files for further analysis in MATLAB or Python.

---
**Author**: WANGSHOUDE  
**Institution**: Universiti Sains Malaysia (USM)

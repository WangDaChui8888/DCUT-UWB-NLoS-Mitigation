# UWB Indoor Positioning Research with Local USM Dataset

This repository contains the complete codebase for a UWB-based indoor positioning research project utilizing a private dataset collected locally at Universiti Sains Malaysia (USM). The project is a comprehensive framework that covers two primary stages:

1.  **NLOS Mitigation & Error Correction**: Training, evaluating, and comparing a novel **Dual-Channel Transformer (DCUT)** against multiple baseline machine learning and deep learning models for NLOS classification and ranging error regression.
2.  **Trajectory Simulation & Visualization**: Generating and plotting realistic, simulated 2D trajectories for an office environment to visually and quantitatively assess the performance of the different positioning algorithms.

**Note:** The dataset used in this project is private and is not included in this repository.

## ✨ Key Features

* **Custom Dataset Pipeline**: All data loading and preprocessing steps are tailored for the local USM dataset.
* **Comprehensive Model Benchmarking**: Includes the DCUT model and baseline implementations for SVM, DNN, XGBoost, CNN-LSTM, and a Single-Channel Transformer.
* **Advanced Trajectory Simulation**: A dedicated module (`7_Draw_office_trajectories`) generates simulated algorithm tracks based on a ground-truth path, allowing for sophisticated visual comparisons.
* **Quantitative Trajectory Analysis**: Includes scripts to compute and plot the Cumulative Distribution Function (CDF) of localization errors, providing a clear measure of positioning accuracy.
* **Modular & Organized**: The codebase is separated into a core ML training pipeline and distinct modules for trajectory analysis and visualization.

## 📁 Project Structure

The repository is organized into two main parts: the core machine learning pipeline and the trajectory analysis modules.

```
.
├── Dataset/                      # Directory for the local UWB dataset (CSV/MAT files)
├── 7_Draw_corridor_trajectories/ # Module for corridor environment (similar to office)
├── 7_Draw_office_trajectories/   # << Module for Office Trajectory Simulation
│   ├── Office_Dataset/           # Input data for trajectory simulation (e.g., ground_truth.csv)
│   ├── Office_Results/           # Output for generated plots and trajectory CSVs
│   ├── generate_trajectory_algorithms.py # SCRIPT: Simulates & plots trajectories for all algos
│   └── 7_office_error_cdf.py     # SCRIPT: Calculates and plots the error CDF from trajectory data
│
├── config_final.py               # Main configuration file for the ML pipeline
├── data_utils.py                 # Data loading and preprocessing functions
├── model_definition.py           # Definition of the Dual-Channel Transformer model
├── Local_run_multiple.py         # << SCRIPT: Main entry point for training multiple models
├── single_algorithm_main.py      # SCRIPT: Entry point for training a single specified model
├── evaluation_utils.py           # Helper functions for performance metrics
├── plot_results.py               # Utility to plot results from training runs
├── calculate_metrics.py          # Utility to calculate metrics from saved predictions
├── flos_module.py                # Module for calculating FLOS features
└── *_experiment.py               # Individual experiment scripts for baseline models
```

## 🚀 Workflow

This project has two distinct workflows.

### Workflow 1: Training the NLOS & Error Models

This workflow is for training the DCUT and baseline models on the raw UWB signal features from the local dataset.

1.  **Prepare Data**: Place your local dataset files into the `Dataset/` directory.
2.  **Configure**: Open `config_final.py` to set paths, hyperparameters, and other training parameters.
3.  **Run Training**: Execute the main training script to train, evaluate, and save all configured models.
    ```bash
    python Local_run_multiple.py
    ```
4.  **Analyze**: Use `plot_results.py` or `calculate_metrics.py` to analyze the output files generated during training.

### Workflow 2: Simulating and Visualizing Trajectories

This workflow uses the insights from the trained models to generate realistic 2D positioning tracks in a simulated office environment.

1.  **Prepare Ground Truth**: Place a ground truth trajectory file (e.g., `Ground_truth.csv` with `x_m`, `y_m` columns) inside the `7_Draw_office_trajectories/Office_Dataset/` directory.
2.  **Run Simulation**: Navigate to the directory and execute the generation script.
    ```bash
    cd 7_Draw_office_trajectories/
    python generate_trajectory_algorithms.py
    ```
    This script will create and save a `.csv` file with the simulated coordinates for each algorithm and a high-quality comparison plot (`.png`, `.pdf`, `.svg`) in the `Office_Results/` folder.
3.  **Analyze Error Distribution**: To get a quantitative understanding of the trajectory errors, run the CDF script.
    ```bash
    # (Still inside 7_Draw_office_trajectories/)
    python 7_office_error_cdf.py
    ```
    This will read the generated trajectory data and produce a CDF plot of localization errors, saving it to `Office_Results/`.

## ⚙️ Setup

1.  **Clone the repository.**
2.  **Create a Python virtual environment.**
3.  **Install dependencies:**
    ```bash
    pip install tensorflow pandas numpy scikit-learn xgboost matplotlib joblib
    ```

## 📊 Output

* **Model Training**: The training workflow (`Local_run_multiple.py`) will generate trained models (`.keras`, `.joblib`), scalers, logs, and performance summary CSVs in a results directory defined in your configuration.
* **Trajectory Simulation**: The simulation workflow (`generate_trajectory_algorithms.py`) produces:
    * A `.png` image file comparing all algorithm trajectories against the ground truth.
    * A `.csv` file containing the x,y coordinates for each simulated trajectory.
* **Trajectory Analysis**: The CDF script (`7_office_error_cdf.py`) produces:
    * A `.png` plot showing the Cumulative Distribution Function of errors for all algorithms.
    * A `.csv` summary of error statistics (mean, median, 95th percentile, etc.).

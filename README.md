# UWB NLoS Identification and Error Mitigation – Multi-Dataset Codebase

This repository integrates **three datasets (Ghent, eWine, Local)** with corresponding source code for Ultra-Wideband (UWB) Non-Line-of-Sight (NLoS) classification and ranging error mitigation.  
The goal is to provide a unified framework for evaluating and comparing different datasets under the same experimental pipeline.

---

## 📂 Repository Structure
- `ghent/`  
  Code and configurations for the **Ghent public dataset**.  
  Includes preprocessing, model training, and evaluation scripts.

- `ewine/`  
  Code for the **eWine dataset**, focusing on benchmarking and transfer evaluation.

- `local/`  
  Code for the **locally collected office dataset**, designed to test robustness in real-world indoor environments.

- `datasets/`  
  Placeholder directory for dataset samples or download scripts.  
  (⚠️ Full raw data is not uploaded here; see each subdirectory README for preparation instructions.)

- `common/` *(optional)*  
  Utility functions shared across datasets (metrics, visualization, logging).

---

## 🚀 Usage
### 1. Environment Setup
Install dependencies:
```bash
pip install -r env/requirements.txt

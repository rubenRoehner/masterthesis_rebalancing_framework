# INSTALLATION AND USAGE
## Installation
To set up the environment, please use the provided environment.yaml file with Conda.

``` bash
# Create the conda environment
conda env create -f environment.yaml

# Activate the environment
conda activate e-scooter-hrl
```

## Usage
1. Data Preprocessing (`/data_preprocessing`)
   - Use the files 'extract_demand_bolt.ipynb' and 'extract_demand_voi.ipynb' to extract pick-up and drop-off events from GBFS feed snapshots
   - Use the file `aggregate_demand_data.ipynb` to aggregate the extracted events into dataframes for training the framework.
   - Use the file `generate_zone_neighbor_map.ipynb` to create a zone neighbor map for the demand forecasting model.

2. Model Training
    - First, train the Demand Forecasting model using the script `/rl_framework/demand_forecasting/IrConv_LSTM/training_model.py`
    - Train the Regional Distribution Coordinator (RDC) using the script `/rl_framework/training_loop.py`
    - Train the User Incentive Coordinator (UIC) using the script `/rl_framework/uic_training_loop.py`

3. Hyperparameter Optimization
   - Use the Optuna-based optimization scripts in the `/rl_framework/hyperparameter_optimization/` directory to tune the hyperparameters for both the RDC and UIC.

4. Evaluation
   - Use the evaluation scripts in the `/rl_framework/evaluation/` directory to assess the performance of the trained models and baselines.
   - Use the Jupyter notebook `/rl_framework/evaluation/evaluation_analysis.ipynb` for analysis of the evaluation results and visualization.
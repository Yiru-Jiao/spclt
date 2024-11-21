# Structure-preserving contrastive learning for spatial time series

## Dependencies

To install the required dependencies, run:

```sh
pip install -r requirements.txt
```

## In order to repeat the experiments:
- Step 0: Ensure that all dependencies are installed as mentioned above. Put datasets in the `datasets` directory. The UEA archive datasets are available at https://www.timeseriesclassification.com/dataset.php. The MacroTraffic and MicroTraffic datasets will be provided if requested.

- Step 1: Test the environment setting by running the `environment_test.py` script. This script tests imports and random seeds that will be used to repeat experiments.

- Step 2: Precompute the distance matrices for the datasets using the `precompute_distance.py` script. Computed distance matrices are saved in corresponding folders where the data are.

- Step 3: Grid search for hyperparameters using the `search_hyperparameters.py` script. The hyperparameters are saved in `results/hyper_parameters/` and openly available.

- Step 4: Train and evaluate various models using the `evaluate.py` script. The trained models and evaluation results are saved in the `results/evaluation` directory.

## Visualization
Visualization of the results can be done using the `figures/visual.ipynb` script. The script generates tables and plots for the evaluation results and saves them in the `results/figures` directory.
# Code for "Structure-preserving contrastive learning for spatial time series"

This study was first submitted to ICLR 2025 and got rejected. Its record on OpenReview is at https://openreview.net/forum?id=sz7HdeVVHo 

After revision based on the advices from ICLR reviewers and extension, a new version of the paper is being submitted to a journal. This code repository is provided for repeating the experiments and reusing the proposed methods.

## Abstract


## Dependencies

For an encapsulated environment, you may create a virtual environment with Python 3.12.4 and install the dependencies by running the following command:

```sh
pip install -r requirements.txt
```

## In order to repeat the experiments:

- __Step 0:__ Ensure that all dependencies are installed as mentioned above. Put datasets in the `datasets` directory. The UEA archive datasets are available at https://www.timeseriesclassification.com/dataset.php. The MacroTraffic and MicroTraffic datasets will be provided if requested via email or GitHub Issues.

- __Step 1:__ Test the environment setting by running `environment_test.py`. This script tests imports and random seeds that will be used to repeat experiments.

- __Step 2:__ Precompute the distance matrices for the UEA datasets using `precompute_distmat.py`. Computed distance matrices are saved in corresponding folders where the data are.

- __Step 3:__ Grid search for hyperparameters using `ssrl_paramsearch.py`. The hyperparameters are saved in `results/hyper_parameters/` and we make them openly available for convenience.

- __Step 4:__ Train various encoders using `ssrl_train.py`. The trained encoders are saved in the `results/pretrain` directory. We make the trained encoders openly available.

- __Step 5:__ Apply the trained models for downstream tasks using `tasks/uea_classification.py`, `tasks/macro_progress.py`, and `tasks/micro_prediction.py` for UEA classification, macroTraffic traffic flow prediction, and microTraffic trajectory prediction. The completely trained models are saved in the `results/finetune` directory; the evaluation results are saved in the `results/evaluation` directory. We also make the models and evaluation results openly available.

## Analysis and visualisation
To analyse and visualise the results, use `figures/visual.ipynb`. The notebook generates tables and plots for the evaluation results and saves them in the `results/figures` directory.

## Citation
```bibtex
@article{,
    title = {Structure-preserving contrastive learning for spatial time series},
    author = {Yiru Jiao and Sander {van Cranenburgh} and Simeon C. Calvert and Hans {van Lint}},
    year = {2025},
    journal = {arXiv preprint},
    pages = {arXiv:}
}
```
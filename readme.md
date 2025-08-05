# Structure-preserving contrastive learning for spatial time series

This study was first submitted to ICLR 2025 and got rejected. Its record on OpenReview is at https://openreview.net/forum?id=sz7HdeVVHo 

After revision based on the advices from ICLR reviewers and extension for more scientific insights, we are submitting a new paper (preprinted at [arXiv](https://arxiv.org/abs/2502.06380)) to a journal. This code repository is provided for repeating the experiments and reusing the proposed methods.

## Highlights
- Regularisers at two scales are introduced for contrastive learning of spatial time series.
- The regularisers preserve fine‐grained similarity structures across time or instances.
- A dynamic mechanism balances contrastive learning and structure preservation.
- State-of-the-arts in spatial time series classification and multi-scale traffic prediction.
- Better preservation of similarity structures implies more informative representations.

## In order to repeat the experiments:

### Dependencies

`torch` `numba` `numpy` `scipy` `pandas` `tqdm` `scikit-learn` `tslearn` `h5py` `pytables` `zarr` `scikit-image`

For an encapsulated environment, you may create a virtual environment with Python 3.12.4 and install the dependencies by running the following command:

```sh
pip install -r requirements.txt
```

### Data
Raw data of the UEA archive datasets can be downloaded from https://www.timeseriesclassification.com/dataset.php. The MacroTraffic and MicroTraffic datasets will be provided if requested via email or GitHub Issues.

Resulting data, i.e., a zipped file of the `./results` folder, can be downloaded from https://doi.org/10.4121/3b8cf098-c2ce-49b1-8e36-74b37872aaa6

### Step-by-step instructions
- __Step 1:__ Test the environment setting by running `./environment_test.py`. This script tests imports and random seeds that will be used to repeat experiments.

- __Step 2:__ Precompute the distance matrices for the UEA datasets using `./precompute_distmat.py`. Computed distance matrices are saved in corresponding folders where the data are.

- __Step 3:__ Grid search for hyperparameters using `./ssrl_paramsearch.py`. The hyperparameters are saved in the directory `./results/hyper_parameters/`.

- __Step 4:__ Train various encoders using `./ssrl_train.py`. The trained encoders are saved in the directory `./results/pretrain/`.

- __Step 5:__ Apply the trained models for downstream tasks using `./tasks/uea_classification.py`, `./tasks/macro_progress.py`, and `./tasks/micro_prediction.py` for UEA classification, macroTraffic traffic flow prediction, and microTraffic trajectory prediction. The completely trained models are saved in the `./results/finetune/` directory; the evaluation results are saved in the `./results/evaluation/` directory.

### Analysis and visualisation
To analyse and visualise the results, use `./figures/visual.ipynb`. The notebook generates tables and plots for the evaluation results and saves them in the `./results/figures/` directory.

## Citation
```bibtex
@article{jiao2025structure,
    title = {Structure-preserving contrastive learning for spatial time series},
    author = {Yiru Jiao and Sander {van Cranenburgh} and Simeon C. Calvert and Hans {van Lint}},
    year = {2025},
    journal = {arXiv preprint},
    pages = {arXiv:2502.06380}
}
```

## Repo references
Thanks to GitHub for offering the open environment, from which this work reuses/learns/adapts the following repositories to different extents:
- TS2Vec https://github.com/zhihanyue/ts2vec
- SoftCLT https://github.com/seunghan96/softclt
- TopoAE https://github.com/BorgwardtLab/topological-autoencoders
- GGAE https://github.com/JungbinLim/GGAE-public
- TAM https://github.com/dmfolgado/tam/

We thank the authors for their contributions to open science.

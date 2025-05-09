# Code for "Structure-preserving contrastive learning for spatial time series"

This study was first submitted to ICLR 2025 and got rejected. Its record on OpenReview is at https://openreview.net/forum?id=sz7HdeVVHo 

After revision based on the advices from ICLR reviewers and extension for more scientific insights, we are submitting a new paper (preprinted at [arXiv](https://arxiv.org/abs/2502.06380)) to a journal. This code repository is provided for repeating the experiments and reusing the proposed methods.

## Abstract
Neural network models are increasingly applied in transportation research to tasks such as prediction. The effectiveness of these models largely relies on learning meaningful latent patterns from data, where self-supervised learning of informative representations can enhance model performance and generalisability. However, self-supervised representation learning for spatially characterised time series, which are ubiquitous in transportation domain, poses unique challenges due to the necessity of maintaining fine-grained spatio-temporal similarities in the latent space. In this study, we introduce two structure-preserving regularisers for the contrastive learning of spatial time series: one regulariser preserves the topology of similarities between instances, and the other preserves the graph geometry of similarities across spatial and temporal dimensions. To balance the contrastive learning objective and the need for structure preservation, we propose a dynamic weighting mechanism that adaptively manages this trade-off and stabilises training. We validate the proposed method through extensive experiments, including multivariate time series classification to demonstrate its general applicability, as well as macroscopic and microscopic traffic prediction to highlight its particular usefulness in encoding traffic interactions. Across all tasks, our method preserves the similarity structures more effectively and improves state-of-the-art task performances. This method can be integrated with an arbitrary neural network model and is particularly beneficial for time series data with spatial or geographical features. Furthermore, our findings suggest that well-preserved similarity structures in the latent space indicate more informative and useful representations. This provides insights to design more effective neural networks for data-driven transportation research.

## Dependencies

`torch` `numba` `numpy` `scipy` `pandas` `tqdm` `scikit-learn` `tslearn` `h5py` `pytables` `zarr` `scikit-image`

For an encapsulated environment, you may create a virtual environment with Python 3.12.4 and install the dependencies by running the following command:

```sh
pip install -r requirements.txt
```

## In order to repeat the experiments:

- __Step 0:__ Ensure that all dependencies are installed as mentioned above. Put datasets in the `datasets` directory. The UEA archive datasets are available at https://www.timeseriesclassification.com/dataset.php. The MacroTraffic and MicroTraffic datasets will be provided if requested via email or GitHub Issues.

- __Step 1:__ Test the environment setting by running `environment_test.py`. This script tests imports and random seeds that will be used to repeat experiments.

- __Step 2:__ Precompute the distance matrices for the UEA datasets using `precompute_distmat.py`. Computed distance matrices are saved in corresponding folders where the data are.

- __Step 3:__ Grid search for hyperparameters using `ssrl_paramsearch.py`. The hyperparameters are saved in `results/hyper_parameters/` and we make them openly available for convenience*.

- __Step 4:__ Train various encoders using `ssrl_train.py`. The trained encoders are saved in the `results/pretrain` directory. We make the trained encoders openly available*.

- __Step 5:__ Apply the trained models for downstream tasks using `tasks/uea_classification.py`, `tasks/macro_progress.py`, and `tasks/micro_prediction.py` for UEA classification, macroTraffic traffic flow prediction, and microTraffic trajectory prediction. The completely trained models are saved in the `results/finetune` directory; the evaluation results are saved in the `results/evaluation` directory. We also make the models and evaluation results openly available*.

*Note: The resulting data are too large (21.2 GB) to be provided in this repository. You are welcome to download them from the following link: https://surfdrive.surf.nl/files/index.php/s/2wNdn6MxIAndxrs

## Analysis and visualisation
To analyse and visualise the results, use `figures/visual.ipynb`. The notebook generates tables and plots for the evaluation results and saves them in the `results/figures` directory.

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

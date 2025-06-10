# Resulting dataset README

This dataset deposited at https://doi.org/10.4121/3b8cf098-c2ce-49b1-8e36-74b37872aaa6 contains all the outputs generated while running experiments in this repository. It can be used for reproducing precomputed matrices, model checkpoints, and evaluation metrics from training and downstream tasks.

## Data Organisation
`results.zip` has a structure as follows:
- `results/hyper_parameters/` contains CSV files logging tuned hyperparameters and the corresponding evaluation score from grid searches. Subfolders are named after datasets, including `UEA`, `MacroTraffic`, `MicroTraffic`, `MacroGRU`, and `MacroLSTM`.

- `results/pretrain/` stores trained encoder checkpoints. Subfolders are organised as `<dataset>/<model>/<subdataset>/file`, where `file` can be `.pth` of the trained model, `.npy` of logarithm weights if the model balances between contrastive learning and structure preservation, `.csv` of training logs, and `.npz` of latent representations.

- `results/finetune/` holds a series of checkpoints during downstream tasks (classification, prediction) for training progress check. Subfolders are named after `<datase_progress>/<model>/ckpt_x.pth`, where `ckpt_x.pth` represents the model checkpoint at epoch `x`.

- `results/evaluation/` contains CSV files reporting evaluation metrics for each task and model type. 

## Data Variables and Column Headings
- **Hyperparameters & Parameters**:  
  - `tau_inst`: Temperature parameter associated with instance-level regularisation.  
  - `tau_temp`: Temperature parameter associated with temporal regularisation.  
  - `temporal_hierarchy`: Option to apply hierarchical temporal features.  
  - `bandwidth` and `batch_size`: Affect both training dynamics and computation of similarity matrices.
  - `weight_lr`: Learning rate for weight updates in the experiment.

- **Evaluation Metrics & Logs**:  
  - For classification tasks, key columns include performance measures such as `svm_acc` and `svm_auprc`.  
  - Traffic prediction evaluations include error measures (`mae`, `rmse`, `error_std`, `explained_variance`) along with kNN-based metrics prefixed by `mean_` (e.g., `mean_trustworthiness`).  
  - Training efficiency logs record `training_time`, `training_epochs`, and the derived `training_time_per_epoch`.

## Usage Notes
- Data files are stored in standard formats (HDF5, CSV, NPY) and can be loaded using Python packages such as `pandas.read_hdf`, `pandas.read_csv`, and `numpy.load`. 
- Pytorch models are saved in PTH and can be loaded using `torch.load()`.
- The dataset is structured to facilitate easy access to hyperparameters, model checkpoints, and evaluation results for each dataset and task.
- For replication details, experiment code is open-sourced at https://github.com/Yiru-Jiao/SPCLT


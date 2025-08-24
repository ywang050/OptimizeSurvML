# OptimizeSurvML

This class will automatically perform efficient hyperparameter tuning using nested CV and Optuna to assist with model selection. Following model-selection, more extensive hyperparameter tuning can be performed using non-nested CV.

The following models are supported:
* Random Survival Forest (```run_rsf_nested_cv```)
* Gradient Boosting Machine Survival Analysis (```run_gbm_nested_cv```)
* Support Vector Machine Survival analysis (```run_svm_nested_cv```)
* DeepSurv (```run_deepsurv_nested_cv```)

## Getting started
Installation:
```
pip install git+https://github.com/ywang050/OptimizeSurvML
```
Basic usage:
```
from OptimizeSurvML.selection import SurvivalModelSelection
model_selection = SurvivalModelSelection(x_train, y_train)
rsf_results = model_selection.run_deepsurv_nested_cv()
==================================================
RSF NESTED CROSS-VALIDATION RESULTS
==================================================
                    metric   mean    std    min    max
0                  c_index  0.743  0.027  0.712  0.772
1   integrated_brier_score  0.142  0.013  0.122  0.154
2  mean_time_dependent_auc  0.786  0.033  0.752  0.839
3                  auc_1yr  0.798  0.024  0.775  0.825
4                  auc_2yr  0.779  0.015  0.762  0.795
5                  auc_3yr  0.772  0.047  0.729  0.851
6                  auc_4yr  0.775  0.061  0.691  0.860
7                  auc_5yr  0.791  0.067  0.702  0.861
```

## API reference
```
class OptimizeSurvML.selection.SurvivalModelSelection(x_data, y_data, n_outer_folds=5, n_inner_folds=5,
                  random_state=17, logging = False, logging_loc = "surv_model_selection.db")
```
### Parameters:
* ```x_data```: X data. (pandas.DataFrame)
  * Should be imputed, one-hot-encoded, and normalized
* ```y_data```: Survival labels (pandas.DataFrame)
  * Should contain events in column 1 and time in column 2
* ```n_outer_folds```: Number of outer CV folds (default: 5) (int)
* ```n_inner_folds```: Number of inner CV folds for hyperparameter tuning (default: 5) (int)
* ```random_state```: Random state for CV (int)
* ```logging```: whether to store the results of Optuna optimization studies (bool) - see [Optuna documentation](https://optuna-dashboard.readthedocs.io/en/latest/getting-started.html)
* ```logging_loc```: location for logging database (str)

### Functions:
Shared parameters:
* ```n_trials```: the number of hyperparameter combinations to try within each inner CV fold. (int)
* ```n_jobs```: number of CPU threads to use for model training. Defaults to using all cores (-1)
* The remaining parameters enable customization of the hyperparameter search space. Can usually be left to default values.

All functions return a DataFrame containing the following metrics for each outer CV fold:
* C-index
* Integrated Brier Score
* Mean time-dependant AUC
* 1-, 2-, 3-, 4-, 5- year AUC
* Best hyperparameters
  
```
run_rsf_nested_cv(n_trials=1000, 
                  n_jobs = -1,
                  n_estimators_min = 50,
                  n_estimators_max = 1000,
                  min_samples_leaf_min = 1,
                  min_samples_leaf_max = 10,
                  min_samples_split_min = 2,
                  min_samples_split_max = 20,
                  max_features = ['sqrt', 'log2', 0.1, 0.3, 0.5, 0.7],
                  bootstrap = [True, False])
```
```
run_svm_nested_cv(n_trials=1000, 
                  n_jobs = -1,
                  alpha_min = 1e-4,
                  alpha_max = 10.0,
                  kernel = ['linear', 'rbf'],
                  rank_ratio_min = 0.1,
                  rank_ratio_max = 0.9,
                  max_iter_min = 500,
                  max_iter_max = 5000,
                  tol_min = 1e-5,
                  tol_max = 1e-3,
                  gamma_min = 1e-4, 
                  gamma_max= 1.0):
```
```
run_gbm_nested_cv(n_trials=1000, 
                  n_jobs = -1,
                  n_estimators_min = 50,
                  n_estimators_max = 1000,
                  learning_rate_min = 0.001,
                  learning_rate_max = 0.3,
                  max_depth_min = 3,
                  max_depth_max = 20,
                  subsample_min = 0.3,
                  subsample_max = 1.0):

```
```
run_deepsurv_nested_cv(n_trials=1000, 
                       n_jobs = -1,
                       n_layers_min = 1,
                       n_layers_max = 4,
                       n_nodes = [8, 16, 32, 64, 128],
                       dropout_min = 0.2,
                       dropout_max = 0.8,
                       lr_min = 1e-5,
                       lr_max = 1e-2,
                       epochs_min = 50,
                       epochs_max = 1000,
                       weight_decay_min = 1e-5, 
                       weight_decay_max= 1e-2,
                       batch_size = [128, 256]):
```


Developed by Yifan Wang

If you have any questions or spot errors, feel free to reach out to me at ywang050@uottawa.ca.

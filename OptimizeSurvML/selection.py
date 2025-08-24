import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from statistics import mean
import torch
import torch.nn as nn
import torchtuples as tt

# pycox imports
from pycox.models import CoxPH, DeepHitSingle
from pycox.evaluation import EvalSurv

# sksurv imports
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.svm import FastKernelSurvivalSVM
from sksurv.metrics import integrated_brier_score, cumulative_dynamic_auc
from sksurv.util import Surv

from datetime import datetime

import scipy
scipy.integrate.simps = scipy.integrate.simpson

class SurvivalModelSelection:
    """
    This class will automatically perform efficient hyperparameter tuning using nested CV 
    (stratified by outcome) and Optuna to assist with model selection. Following model-selection,
    more extensive hyperparameter tuning can be performed using non-nested CV.

    The following models are supported:
        Random Survival Forest (run_rsf_nested_cv)
        Gradient Boosting Machine Survival Analysis (run_gbm_nested_cv)
        Support Vector Machine Survival analysis (run_svm_nested_cv)
        DeepSurv (run_deepsurv_nested_cv)

    The following metrics will be returned
        C-index
        Integrated Brier Score
        Mean time-dependant AUC
        1-, 2-, 3-, 4-, 5- year AUC

    By default, 1000 Optuna trials (each one a different hyperparameter combination)
    are used, representing the upper limit of the suggested number of trials when using
    the TPE sampler. This can be customized when calling the method using n_trials.

    Usage example:
    -----
    import *** (see note below under __init__)
    model_selection = SurvivalModelSelection(x_train, y_train)
    rsf_results = model_selection.run_deepsurv_nested_cv(n_trials = 1000)
    rsf_results
    > ==================================================
    > DeepSurv NESTED CROSS-VALIDATION RESULTS
    > ==================================================
    >                     metric   mean    std    min    max
    > 0                  c_index  0.645  0.085  0.508  0.729
    > 1   integrated_brier_score  0.113  0.008  0.101  0.121
    > 2  mean_time_dependent_auc  0.651  0.097  0.494  0.748
    > 3                  auc_1yr  0.662  0.095  0.538  0.785
    > 4                  auc_2yr  0.669  0.073  0.559  0.748
    > 5                  auc_3yr  0.645  0.108  0.486  0.746
    > 6                  auc_4yr  0.619  0.121  0.441  0.746
    > 7                  auc_5yr  0.656  0.121  0.450  0.743
    (details about specific folds are also returned)

    Developed by Yifan Wang
    If you have any questions or spot errors, feel free to reach out to me at ywang050@uottawa.ca.
        
    """
    
    def __init__(self, x_data, y_data, n_outer_folds=5, n_inner_folds=5, random_state=17, 
                 logging = False, logging_loc = "surv_model_selection.db"):
        """
        * Important: Run the following code first *

        -----
        import numpy as np   #
        import pandas as pd
        import optuna
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from statistics import mean
        import torch
        import torch.nn as nn
        import torchtuples as tt
        from pycox.models import CoxPH
        from pycox.evaluation import EvalSurv
        from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
        from sksurv.svm import FastKernelSurvivalSVM
        from sksurv.metrics import integrated_brier_score, cumulative_dynamic_auc
        from sksurv.util import Surv
        import scipy
        scipy.integrate.simps = scipy.integrate.simpson
        -----
        
        Parameters:
        -----
        
        x_data : X data. (pandas.DataFrame)
            * Should be imputed, one-hot-encoded, and normalized
        y_data : Survival labels (pandas.DataFrame)
            * Should contain events in column 1 and time in column 2
        n_outer_folds : Number of outer CV folds (default: 5) (int)
        n_inner_folds : Number of inner CV folds for hyperparameter tuning (default: 5) (int)
        random_state (int)
        """
        if not isinstance(x_data, pd.DataFrame):
            raise TypeError("x_data must be a pandas DataFrame")
        if not isinstance(y_data, pd.DataFrame):
            raise TypeError("y_data must be a pandas DataFrame")
        
        # Check for missing values
        if x_data.isnull().any().any():
            raise ValueError("x_data contains missing values. Please impute missing values before initialization.")
        if y_data.isnull().any().any():
            raise ValueError("y_data contains missing values. Please ensure survival data is complete.")
        
        # Check if data is entirely numeric
        if not x_data.select_dtypes(include=[np.number]).shape[1] == x_data.shape[1]:
            raise ValueError("x_data must contain only numeric values. Please ensure all categorical variables are one-hot encoded.")
        if not y_data.select_dtypes(include=[np.number]).shape[1] == y_data.shape[1]:
            raise ValueError("y_data must contain only numeric values.")
        
        # Check if x_data and y_data have the same length
        if len(x_data) != len(y_data):
            raise ValueError(f"x_data and y_data must have the same length. x_data has {len(x_data)} rows, y_data has {len(y_data)} rows.")
        
        # Check if y_data has exactly 2 columns
        if y_data.shape[1] != 2:
            raise ValueError(f"y_data must have exactly 2 columns (event, time). Found {y_data.shape[1]} columns.")
        
        # Check if first column of y_data contains only 0s and 1s (binary event indicator)
        event_col = y_data.iloc[:, 0]
        if not set(event_col.unique()).issubset({0, 1}):
            raise ValueError("The first column of y_data (event indicator) must contain only 0 and 1 values.")
        
        # Check if second column of y_data contains positive values (survival times)
        time_col = y_data.iloc[:, 1]
        if (time_col <= 0).any():
            raise ValueError("The second column of y_data (time) must contain only positive values.")
        
        # Check if x_data appears to be normalized (rough check)
        # This checks if the mean is close to 0 and std is close to 1 for most columns
        x_means = x_data.mean()
        x_stds = x_data.std()
        
        # Allow some tolerance for normalization check
        mean_threshold = 2
        std_threshold_high = 1.5
        
        non_normalized_cols = []
        for col in x_data.columns:
            if abs(x_means[col]) > mean_threshold or x_stds[col] > std_threshold_high:
                non_normalized_cols.append(col)
        
        if len(non_normalized_cols) > len(x_data.columns) * 0.2:  # If more than 20% of columns appear non-normalized
            print("Warning: x_data may not be properly normalized. Consider standardizing your features to ensure fair model comparison, as SVM and DeepSurv are sensitive to standardization")
            print(f"Columns that may need normalization: {non_normalized_cols[:5]}...")  # Show first 5
            

        # Can change to get more detailed logging per trial of tuning
        optuna.logging.set_verbosity(optuna.logging.CRITICAL)
        
        self.x_data = x_data
        self.y_data = y_data
        self.n_outer_folds = n_outer_folds
        self.n_inner_folds = n_inner_folds
        self.random_state = random_state
        self.logging = logging
        self.logging_loc = logging_loc
        
        # Initialize variables that will be set during CV loops
        self.X_train_outer = None
        self.X_val_outer = None
        self.y_train_outer = None
        self.y_val_outer = None
        
        # Desired time points for evaluation. 
        self.eval_times = np.array([50, 250, 450, 650, 850, 1050, 1250, 1450, 1650, 1850, 2050])
        self.yearly_times = np.array([365, 730, 1095, 1460, 1825])

    # Scorer helper function for c-index evaluation of sk-surv models
    def c_index_scorer(self, model, x, y):
        return model.score(x, y)

    # Formats survival data dataframe into structured array for sk-surv
    def get_surv(self, y_data):
        # y_data should contain events in column 1 and time in column 2
        nicm_surv_y_train = Surv.from_arrays(event = y_data.iloc[:, 0],
                                                time = y_data.iloc[:, 1])                                      
        return nicm_surv_y_train

    # Finds closest time to the desired time points within a dataset
    def find_closest_time(self, available_times, desired_times):
        closest_times = []
        for target_time in desired_times:
            closest_idx = np.argmin(np.abs(available_times - target_time))
            closest_time = available_times[closest_idx]
            closest_times.append(closest_time)
        return np.unique(closest_times)

    # Optuna objective function for GBM
    def objective_gbm(self, trial, x_train_outer, y_train_outer, param_config, n_jobs):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', param_config['n_estimators_min'], param_config['n_estimators_max'], step = 10),
            'learning_rate': trial.suggest_float('learning_rate', param_config['learning_rate_min'], param_config['learning_rate_max'], log=True),
            'max_depth': trial.suggest_int('max_depth', param_config['max_depth_min'], param_config['max_depth_max']),
            'subsample': trial.suggest_float('subsample', param_config['subsample_min'], param_config['subsample_max'], step=0.1)
        }

        model = GradientBoostingSurvivalAnalysis(**params)

        # Perform inner CV loop to optimize hyperparameters 
        inner_cv = StratifiedKFold(n_splits=self.n_inner_folds, shuffle=True, random_state=self.random_state)
        c_indices_inner = cross_val_score(
            model, x_train_outer, y_train_outer, 
            scoring=self.c_index_scorer, n_jobs=n_jobs,
            cv=inner_cv.split(x_train_outer, [item[0] for item in y_train_outer])
        )

        # Optimize by c-index
        return c_indices_inner.mean()

    # Optuna objective function for RSF        
    def objective_rsf(self, trial, x_train_outer, y_train_outer, param_config, n_jobs):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', param_config['n_estimators_min'], param_config['n_estimators_max'], step = 10),
            'min_samples_split': trial.suggest_int('min_samples_split', param_config['min_samples_split_min'], param_config['min_samples_split_max']),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', param_config['min_samples_leaf_min'], param_config['min_samples_leaf_max']),
            'max_features': trial.suggest_categorical('max_features', param_config['max_features']),
            'bootstrap': trial.suggest_categorical('bootstrap', param_config['bootstrap'])
        }
      
        model = RandomSurvivalForest(**params)
        
        # Perform inner CV loop to optimize hyperparameters 
        inner_cv = StratifiedKFold(n_splits=self.n_inner_folds, shuffle=True, random_state=self.random_state)
        c_indices_inner = cross_val_score(
            model, x_train_outer, y_train_outer, 
            scoring=self.c_index_scorer, n_jobs=n_jobs,
            cv=inner_cv.split(x_train_outer, [item[0] for item in y_train_outer])
        )
        
        # Optimize by c-index
        return c_indices_inner.mean()
        

    # Optuna objective function for SVM
    def objective_svm(self, trial, x_train_outer, y_train_outer, param_config, n_jobs):
        params = {
            'alpha': trial.suggest_float('alpha', param_config['alpha_min'], param_config['alpha_max'], log=True),
            'kernel': trial.suggest_categorical('kernel', param_config['kernel']),
            'rank_ratio': trial.suggest_float('rank_ratio', param_config['rank_ratio_min'], param_config['rank_ratio_max'], step=0.1),
            'max_iter': trial.suggest_int('max_iter', param_config['max_iter_min'], param_config['max_iter_max'], step=10),
            'tol': trial.suggest_float('tol', param_config['tol_min'], param_config['tol_max'], log=True)
        }

        # Gamma hyperparameter is specific to rbf kernel
        if params['kernel'] == 'rbf':
            params['gamma'] = trial.suggest_float('gamma', param_config['gamma_min'], param_config['gamma_max'], log=True)

        model = FastKernelSurvivalSVM(**params)
        
        # Perform inner CV loop to optimize hyperparameters 
        inner_cv = StratifiedKFold(n_splits=self.n_inner_folds, shuffle=True, random_state=self.random_state)
        c_indices_inner = cross_val_score(
            model, x_train_outer, y_train_outer, 
            scoring=self.c_index_scorer, n_jobs=-1,
            cv=inner_cv.split(x_train_outer, [item[0] for item in y_train_outer])
        )

        # Optimize by c-index
        return c_indices_inner.mean()

    # Construct a neural net architecture for DeepSurv
    def define_net(self, trial, in_features, param_config):

        n_layers = trial.suggest_int("n_layers", param_config['n_layers_min'], param_config['n_layers_max'])
        layers = []
        current_features = in_features

        # Each loop dynamically constructs a layer with certain number of units and 
        # dropout threshold and appends them to *layers*
        for i in range(n_layers):
            out_features = trial.suggest_categorical(f"n_units_l{i}", param_config['n_nodes'])  ### Change as needed 
            layers.append(nn.Linear(current_features, out_features))
            layers.append(nn.ReLU())
            p = trial.suggest_float(f"dropout_l{i}", param_config['dropout_min'], param_config['dropout_max'], step=0.1)  ### Change as needed
            layers.append(nn.Dropout(p))
            current_features = out_features

        layers.append(nn.Linear(current_features, 1, bias=False))
        return nn.Sequential(*layers)

    # Optuna objective function for DeepSurv
    def objective_nn(self, trial, x_train_outer, y_train_outer, param_config, n_jobs):
        ### Modify parameter search space as needed ###
        lr = trial.suggest_float("lr", param_config['lr_min'], param_config['lr_max'], log=True)
        epochs = trial.suggest_int("epochs", param_config['epochs_min'], param_config['epochs_max'], step=10)
        weight_decay = trial.suggest_float("weight_decay", param_config['weight_decay_min'], param_config['weight_decay_max'], log=True)
        batch_size = trial.suggest_categorical("batch_size", param_config['batch_size'])
        
        inner_cv = StratifiedKFold(n_splits=self.n_inner_folds, shuffle=True, random_state=self.random_state)
        c_indices = []
        
        for fold_inner, (train_idx_inner, val_idx_inner) in enumerate(inner_cv.split(x_train_outer, y_train_outer[1])):
            net = self.define_net(trial, x_train_outer.shape[1], param_config)
            model = CoxPH(net, tt.optim.Adam(lr=lr, weight_decay=weight_decay))

            # Manually define inner CV loop
            x_train_inner = x_train_outer[train_idx_inner]
            x_val_inner = x_train_outer[val_idx_inner]

            # Format y dataset for DeepSurv - tuple(all survival times, all outcomes)
            y_train_inner = tuple(element[train_idx_inner] for element in y_train_outer)
            y_val_inner = tuple(element[val_idx_inner] for element in y_train_outer)
            
            for epoch in range(epochs):
                model.fit(x_train_inner, y_train_inner, batch_size=batch_size, verbose=False)
                model.compute_baseline_hazards()

            # Note: Optuna pruners are not compatible with cross-validation, as it cannot uniquely
            # identify each fold, and will prune based on the first fold alone. As such, this code does 
            # not take use optuna pruners.
            
            surv = model.predict_surv_df(x_val_inner)
            c_index = EvalSurv(surv, y_val_inner[0], y_val_inner[1], censor_surv='km').concordance_td()
            c_indices.append(c_index)
        
        # Optimize by c-index
        return mean(c_indices)

    # Define a neural network based on pre-specified parameters from inner CV loop
    def define_net_prespecified(self, study, in_features):
        params = study.best_trial.params
        n_layers = params['n_layers']
        layers = []
        current_features = in_features
        
        for i in range(n_layers):
            out_features = params[f'n_units_l{i}']
            dropout = params[f'dropout_l{i}']
            layers.append(nn.Linear(current_features, out_features))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_features = out_features

        layers.append(nn.Linear(current_features, 1, bias=False))
        return nn.Sequential(*layers)
        
    def run_gbm_nested_cv(self, n_trials=1000, 
                          n_jobs = -1,
                          n_estimators_min = 50,
                          n_estimators_max = 1000,
                          learning_rate_min = 0.001,
                          learning_rate_max = 0.3,
                          max_depth_min = 3,
                          max_depth_max = 20,
                          subsample_min = 0.3,
                          subsample_max = 1.0):
        
    
        param_config = {
            'n_estimators_min': n_estimators_min,
            'n_estimators_max': n_estimators_max,
            'learning_rate_min': learning_rate_min,
            'learning_rate_max': learning_rate_max,
            'max_depth_min': max_depth_min,
            'max_depth_max': max_depth_max,
            'subsample_min': subsample_min,
            'subsample_max': subsample_max
        }

        results_outer = pd.DataFrame()
        
        outer_cv = StratifiedKFold(n_splits=self.n_outer_folds, shuffle=True, random_state=self.random_state)

        surv_y_data = self.get_surv(self.y_data)
        
        for fold, (train_idx, val_idx) in enumerate(outer_cv.split(self.x_data, [item[0] for item in surv_y_data])):
            x_train_outer = self.x_data.iloc[train_idx]
            x_val_outer = self.x_data.iloc[val_idx]
            
            y_train_outer = surv_y_data[train_idx]
            y_val_outer = surv_y_data[val_idx]
            
            # Hyperparameter optimization
            if self.logging:
                time = datetime.now().strftime("%Y-%m-%d %H:%M")
                study = optuna.create_study(study_name = 'GBM_Fold{}_{}'.format(fold+1, time), direction = 'maximize', storage = "sqlite:///{}".format(self.logging_loc))
            else:
                study = optuna.create_study(direction='maximize')
                
            study.optimize(lambda trial: self.objective_gbm(trial, x_train_outer, y_train_outer, param_config, n_jobs), 
                           n_trials=n_trials, n_jobs=1, show_progress_bar = True)
            
            # Train final model with best parameters
            model = GradientBoostingSurvivalAnalysis(**study.best_trial.params)
            model.fit(x_train_outer, y_train_outer)
            
            # Evaluate model
            c_index = model.score(x_val_outer, y_val_outer)

            # Get evaluation times
            times = self.find_closest_time([item[1] for item in y_val_outer], self.eval_times)
            yearly_times = self.find_closest_time([item[1] for item in y_val_outer], self.yearly_times)
            
            # Additional metrics
            surv_func = model.predict_survival_function(x_val_outer)
            preds = np.asarray([[fn(t) for t in times] for fn in surv_func])
            ibs = integrated_brier_score(y_train_outer, y_val_outer, preds, times)
            
            chf_func = model.predict_cumulative_hazard_function(x_val_outer, return_array=False)
            risk_scores = np.vstack([chf(times) for chf in chf_func])
            auc_scores_comprehensive, mean_auc = cumulative_dynamic_auc(
                y_train_outer, y_val_outer, risk_scores, times
            )
            
            # Yearly AUCs
            risk_scores_yearly = np.vstack([chf(yearly_times) for chf in chf_func])
            auc_scores_yearly, _ = cumulative_dynamic_auc(
                y_train_outer, y_val_outer, risk_scores_yearly, yearly_times
            )
            
            fold_results = {
                'fold': fold + 1,
                'c_index': c_index,
                'integrated_brier_score': ibs,
                'mean_time_dependent_auc': mean_auc,
                'auc_1yr': auc_scores_yearly[0],
                'auc_2yr': auc_scores_yearly[1],
                'auc_3yr': auc_scores_yearly[2],
                'auc_4yr': auc_scores_yearly[3],
                'auc_5yr': auc_scores_yearly[4],
                'best_params': study.best_trial.params
            }
            
            results_outer = pd.concat([results_outer, pd.DataFrame([fold_results])], ignore_index=True)
            print(f"Fold {fold + 1} - C-index: {c_index:.3f}, IBS: {ibs:.3f}, Mean AUC: {mean_auc:.3f}")
        
        return self._format_results(results_outer, 'GBM')
    
    # Run nested CV for RSF
    def run_rsf_nested_cv(self, n_trials=1000, 
                          n_jobs = -1,
                          n_estimators_min = 50,
                          n_estimators_max = 1000,
                          min_samples_leaf_min = 1,
                          min_samples_leaf_max = 10,
                          min_samples_split_min = 2,
                          min_samples_split_max = 20,
                          max_features = ['sqrt', 'log2', 0.1, 0.3, 0.5, 0.7],
                          bootstrap = [True, False]):
        
    
        param_config = {
            'n_estimators_min': n_estimators_min,
            'n_estimators_max': n_estimators_max,
            'min_samples_leaf_min': min_samples_leaf_min,
            'min_samples_leaf_max': min_samples_leaf_max,
            'min_samples_split_min': min_samples_split_min,
            'min_samples_split_max': min_samples_split_max,
            'max_features': max_features,
            'bootstrap': bootstrap
        }
        
        results_outer = pd.DataFrame()
        
        outer_cv = StratifiedKFold(n_splits=self.n_outer_folds, shuffle=True, random_state=self.random_state)

        surv_y_data = self.get_surv(self.y_data)
        
        for fold, (train_idx, val_idx) in enumerate(outer_cv.split(self.x_data, [item[0] for item in surv_y_data])):
            x_train_outer = self.x_data.iloc[train_idx]
            x_val_outer = self.x_data.iloc[val_idx]
            
            y_train_outer = surv_y_data[train_idx]
            y_val_outer = surv_y_data[val_idx]
            
            # Hyperparameter optimization
            if self.logging:
                time = datetime.now().strftime("%Y-%m-%d %H:%M")
                study = optuna.create_study(study_name = 'RSF_Fold{}_{}'.format(fold+1, time), direction = 'maximize', storage = "sqlite:///{}".format(self.logging_loc))
            else:
                study = optuna.create_study(direction='maximize')
                
            study.optimize(lambda trial: self.objective_rsf(trial, x_train_outer, y_train_outer, param_config, n_jobs), 
                           n_trials=n_trials, n_jobs=1, show_progress_bar = True)
            
            # Train final model with best parameters
            model = RandomSurvivalForest(**study.best_trial.params)
            model.fit(x_train_outer, y_train_outer)
            
            # Evaluate model
            c_index = model.score(x_val_outer, y_val_outer)

            # Get evaluation times
            times = self.find_closest_time([item[1] for item in y_val_outer], self.eval_times)
            yearly_times = self.find_closest_time([item[1] for item in y_val_outer], self.yearly_times)
            
            # Additional metrics
            surv_func = model.predict_survival_function(x_val_outer)
            preds = np.asarray([[fn(t) for t in times] for fn in surv_func])
            ibs = integrated_brier_score(y_train_outer, y_val_outer, preds, times)
            
            chf_func = model.predict_cumulative_hazard_function(x_val_outer, return_array=False)
            risk_scores = np.vstack([chf(times) for chf in chf_func])
            auc_scores_comprehensive, mean_auc = cumulative_dynamic_auc(
                y_train_outer, y_val_outer, risk_scores, times
            )
            
            # Yearly AUCs
            risk_scores_yearly = np.vstack([chf(yearly_times) for chf in chf_func])
            auc_scores_yearly, _ = cumulative_dynamic_auc(
                y_train_outer, y_val_outer, risk_scores_yearly, yearly_times
            )
            
            fold_results = {
                'fold': fold + 1,
                'c_index': c_index,
                'integrated_brier_score': ibs,
                'mean_time_dependent_auc': mean_auc,
                'auc_1yr': auc_scores_yearly[0],
                'auc_2yr': auc_scores_yearly[1],
                'auc_3yr': auc_scores_yearly[2],
                'auc_4yr': auc_scores_yearly[3],
                'auc_5yr': auc_scores_yearly[4],
                'best_params': study.best_trial.params
            }
            
            results_outer = pd.concat([results_outer, pd.DataFrame([fold_results])], ignore_index=True)
            print(f"Fold {fold + 1} - C-index: {c_index:.3f}, IBS: {ibs:.3f}, Mean AUC: {mean_auc:.3f}")
        
        return self._format_results(results_outer, 'RSF')
    
    def run_svm_nested_cv(self, n_trials=1000, 
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
        
    
        param_config = {
            'alpha_min': alpha_min,
            'alpha_max': alpha_max,
            'kernel': kernel,
            'rank_ratio_min': rank_ratio_min,
            'rank_ratio_max': rank_ratio_max,
            'max_iter_min': max_iter_min,
            'max_iter_max': max_iter_max,
            'tol_min': tol_min,
            'tol_max': tol_max,
            'gamma_min': gamma_min,
            'gamma_max': gamma_max
        }
        
        results_outer = pd.DataFrame()
        
        outer_cv = StratifiedKFold(n_splits=self.n_outer_folds, shuffle=True, random_state=self.random_state)

        surv_y_data = self.get_surv(self.y_data)
        
        for fold, (train_idx, val_idx) in enumerate(outer_cv.split(self.x_data, [item[0] for item in surv_y_data])):
            x_train_outer = self.x_data.iloc[train_idx]
            x_val_outer = self.x_data.iloc[val_idx]
            
            y_train_outer = surv_y_data[train_idx]
            y_val_outer = surv_y_data[val_idx]
            
            # Hyperparameter optimization
            if self.logging:
                time = datetime.now().strftime("%Y-%m-%d %H:%M")
                study = optuna.create_study(study_name = 'SVM_Fold{}_{}'.format(fold+1, time), direction = 'maximize', storage = "sqlite:///{}".format(self.logging_loc))
            else:
                study = optuna.create_study(direction='maximize')
                
            study.optimize(lambda trial: self.objective_svm(trial, x_train_outer, y_train_outer, param_config, n_jobs), 
                           n_trials=n_trials, n_jobs=1, show_progress_bar = True)
            
            # Train final model with best parameters
            model = FastKernelSurvivalSVM(**study.best_trial.params)
            model.fit(x_train_outer, y_train_outer)
            
            # Evaluate model
            c_index = model.score(x_val_outer, y_val_outer)
            
            fold_results = {
                'fold': fold + 1,
                'c_index': c_index,
                'best_params': study.best_trial.params
            }
            
            results_outer = pd.concat([results_outer, pd.DataFrame([fold_results])], ignore_index=True)
            print(f"Fold {fold + 1} - C-index: {c_index:.3f}")
        
        print("NOTE: IBS and AUC metrics cannot be calculated as Survival SVM does not return a survival or hazard function.")
        return self._format_results(results_outer, 'SVM', c_index_only=True)

    # Run nested CV for DeepSurv
    def run_deepsurv_nested_cv(self, n_trials=1000, 
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
        
    
        param_config = {
            'n_layers_min': n_layers_min,
            'n_layers_max': n_layers_max,
            'n_nodes': n_nodes,
            'dropout_min': dropout_min,
            'dropout_max': dropout_max,
            'lr_min': lr_min,
            'lr_max': lr_max,
            'epochs_min': epochs_min,
            'epochs_max': epochs_max,
            'weight_decay_min': weight_decay_min,
            'weight_decay_max': weight_decay_max,
            'batch_size': batch_size
        }
        if n_jobs < 0:
            threads = torch.get_num_threads()
            torch.set_num_threads(threads-n_jobs+1)
        else:
            torch.set_num_threads(n_jobs)

        results_outer = pd.DataFrame()
        
        # Convert data for DeepSurv if needed
        x_data_array = self.x_data.astype(np.float32).to_numpy()

        get_target = lambda df: (df.iloc[:,1].values.astype(np.float32), 
                         df.iloc[:,0].values.astype(np.float32))
        
        y_data_array = get_target(self.y_data)
        
        outer_cv = StratifiedKFold(n_splits=self.n_outer_folds, shuffle=True, random_state=self.random_state)
        
        for fold, (train_idx, val_idx) in enumerate(outer_cv.split(x_data_array, y_data_array[1])):
            # Set outer fold data
            x_train_outer = x_data_array[train_idx]
            x_val_outer = x_data_array[val_idx]
            # Format y dataset for DeepSurv - tuple(all survival times, all outcomes)
            y_train_outer = tuple(element[train_idx] for element in y_data_array)
            y_val_outer = tuple(element[val_idx] for element in y_data_array)
            
            # Hyperparameter optimization
            if self.logging:
                time = datetime.now().strftime("%Y-%m-%d %H:%M")
                study = optuna.create_study(study_name = 'DeepSurv_Fold{}_{}'.format(fold+1, time), direction = 'maximize', storage = "sqlite:///{}".format(self.logging_loc))
            else:
                study = optuna.create_study(direction='maximize')
                
            study.optimize(lambda trial: self.objective_nn(trial, x_train_outer, y_train_outer, param_config, n_jobs), 
                           n_trials=n_trials, n_jobs=1, show_progress_bar = True)
            
            # Train final model with best parameters
            net = self.define_net_prespecified(study, x_train_outer.shape[1])
            model = CoxPH(net, tt.optim.Adam(
                lr=study.best_trial.params['lr'],
                weight_decay=study.best_trial.params["weight_decay"]
            ))
            
            model.fit(x_train_outer, y_train_outer,
                     epochs=study.best_trial.params["epochs"],
                     batch_size=study.best_trial.params['batch_size'], verbose=False)

            # baseline hazards must be computed before metric calculation
            model.compute_baseline_hazards()
            
            # Evaluate model
            surv = model.predict_surv_df(x_val_outer)
            ev = EvalSurv(surv, y_val_outer[0], y_val_outer[1], censor_surv='km')

            # C index
            c_index = ev.concordance_td()

            # Get evaluation times
            available_times = surv.index.values
            times = self.find_closest_time(available_times, self.eval_times)
            yearly_times = self.find_closest_time(available_times, self.yearly_times)
            
            # IBS (
            ibs = ev.integrated_brier_score(times)

            # Mean AUC - uses chf function
            chf_func = model.predict_cumulative_hazards(x_val_outer)
            risk_scores = chf_func.loc[times].values.transpose()

            y_train_outer_surv = Surv.from_arrays(event = y_train_outer[1], time = y_train_outer[0]) 
            y_val_outer_surv = Surv.from_arrays(event = y_val_outer[1], time = y_val_outer[0])    
            
            auc_scores, mean_auc = cumulative_dynamic_auc(
                y_train_outer_surv, y_val_outer_surv, risk_scores, times)
            
            
            # Yearly AUC
            risk_scores_yearly = chf_func.loc[yearly_times].values.transpose()
            
            auc_scores_yearly, _ = cumulative_dynamic_auc(
                y_train_outer_surv, y_val_outer_surv, risk_scores_yearly, yearly_times)
            
            fold_results = {
                'fold': fold + 1,
                'inner_c_index': study.best_trial.value,
                'c_index': c_index,
                'integrated_brier_score': ibs,
                'mean_time_dependent_auc': mean_auc,
                'auc_1yr': auc_scores_yearly[0],
                'auc_2yr': auc_scores_yearly[1],
                'auc_3yr': auc_scores_yearly[2],
                'auc_4yr': auc_scores_yearly[3],
                'auc_5yr': auc_scores_yearly[4],
                'best_params': study.best_trial.params
            }
            
            results_outer = pd.concat([results_outer, pd.DataFrame([fold_results])], ignore_index=True)
            print(f"Fold {fold + 1} - C-index: {c_index:.3f}")
        
        return self._format_results(results_outer, 'DeepSurv')

    # Format + display results    
    def _format_results(self, results_df, model_name, c_index_only=False):
        print("\n" + "="*50)
        print(f"{model_name} NESTED CROSS-VALIDATION RESULTS")
        print("="*50)
        
        if c_index_only:
            metrics = ['c_index']
        else:
            metrics = ['c_index', 'integrated_brier_score', 'mean_time_dependent_auc', 
                      'auc_1yr', 'auc_2yr', 'auc_3yr', 'auc_4yr', 'auc_5yr']
        
        summary_stats = pd.DataFrame({
            'metric': metrics,
            'mean': [results_df[metric].mean() for metric in metrics],
            'std': [results_df[metric].std() for metric in metrics],
            'min': [results_df[metric].min() for metric in metrics],
            'max': [results_df[metric].max() for metric in metrics]
        })

        print(summary_stats.round(3))
        
        # Detailed results per fold
        detailed_df = results_df[['fold'] + metrics + ['best_params']].copy()
        
        # Compute average and std rows
        avg_row = {'fold': 'Average', **{metric: results_df[metric].mean() for metric in metrics}}
        std_row = {'fold': 'Std Dev', **{metric: results_df[metric].std() for metric in metrics}}
        
        # Append rows
        detailed_df = pd.concat(
            [detailed_df, pd.DataFrame([avg_row, std_row])],
            ignore_index=True
        )
        
        return detailed_df

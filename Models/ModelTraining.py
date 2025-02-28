"""
Tree-based Models training for my mid-freq crpto trading strategies
Author: Haoqing Wu
Date: 2024-12-22

This module demos how to train CatBoost models for a classification task

"""
import pandas as pd
import numpy as np
import datetime as dt
import json, logging
from typing import Optional, List, Dict, Tuple, Union
from pathlib import Path
from collections import defaultdict

from catboost import CatBoostRegressor, CatBoostClassifier, Pool
from sklearn.metrics import ( mean_absolute_error, 
                              mean_squared_error, 
                              r2_score, 
                              accuracy_score, 
                              precision_score, 
                              recall_score )
# from Utils.FeatureEngineering import equipFeaturesAndLabels
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import TimeSeriesSplit

import statsmodels.api as sm


###################################
######## Weight Adjustment ########
def weights_adjustment( y, alpha: float = 0.75 ) -> np.ndarray:
    """
    Calculate sample weights for a 0-1 binary classification target, assigning higher weights
    to the minority class to address class imbalance.

    Parameters:
        y (pandas.Series or array-like): Binary labels where 1 indicates the minority class.
        alpha (float, optional): Multiplier for the minority class weights. Defaults to 0.75.

    Returns:
        numpy.ndarray: An array of sample weights, where the minority class samples have
                       increased weight to alleviate imbalance effects.
    """

    class_counts = y.value_counts()
    majority_class = class_counts.idxmax()
    minority_class = 1 - majority_class
    
    base_weights = {
        majority_class: 1.0,
        minority_class: class_counts[ majority_class ] / class_counts[ minority_class ]
    }
    
    adjusted_weights = {
        majority_class: base_weights[majority_class],
        minority_class: base_weights[minority_class] * alpha
    }
    
    return y.map( adjusted_weights ).values


# Focal loss integrated in Catboost model
class FocalLossObjective:
    def __init__( self, alpha: float = 0.25, gamma: float = 2.0 ):
        """
        Focal Loss is a modification of the standard Cross-Entropy Loss to address class imbalance:
            $$
            FL(p_t) = - alpha * (1 - p_t)^gamma * log(p_t)
            $$
        where:
        p_t: predicted probability;
        alpha: class weights (prioritizes minority class);
        gamma: focusing parameter (higher values prioritize misclassified samples).
        """

        self.alpha = alpha
        self.gamma = gamma

    def calc_ders_range( self, approxes, targets, weights ):
        # approxes: model's raw outputs (logits)
        # targets: true labels (0 or 1)
        # Returns: list of (gradient, hessian) pairs
        ders = []
        for approx, target in zip( approxes, targets ):
            p = 1 / ( 1 + np.exp( -approx ) )  # sigmoid
            y = target

            if y == 1:
                focal_factor = self.alpha * ( 1 - p ) ** self.gamma
            else:
                focal_factor = ( 1 - self.alpha ) * p ** self.gamma

            grad = ( p - y ) * focal_factor
            hess = p * ( 1 - p ) * focal_factor

            ders.append( ( grad, hess ) )
        return ders
###################################


####################################
########## Model Training ##########

# 1. CatBoost Classifier Training
def CatBoostBinaryClassifierTraining( X: pd.DataFrame, 
                                        y: pd.Series, 
                                        model_name: str,
                                        model_params: dict,
                                        tscv_params: dict,
                                        walk_forward_CV: bool = False,
                                        training_config: dict = {},
                                        cat_features: list[ str ] = None,
                                        custom_metrics: list[ str ] = None,
                                        output_dir: str = "./models",
                                        experiment_name: str = "exp1"
                                    )   -> tuple[ list[ CatBoostClassifier ], pd.DataFrame, pd.DataFrame ]:
    """
    Streamlined CatBoost model training with TimeSeriesSplit or Walk-Forward cross-validation.
    Args:
        - X (pd.DataFrame): The input features dataframe.
        - y (pd.Series): The target variable series.
        - model_name (str): The name of the model.
        - model_params (dict): The CatBoost model parameters.
        - tscv_params (dict): The TimeSeriesSplit cross-validation parameters.
        - walk_forward_CV (bool, optional): Whether to use Walk-Forward Cross-Validation. Defaults to False.
        - training_config (dict): The training configuration parameters for model.fit
        - cat_features (List[str], optional): Categorical features to include in the model. Defaults to None.
        - custom_metrics (List[str], optional): Custom metrics to evaluate model performance. Defaults to None.
    Returns:
        - Tuple[List[Union[CatBoostClassifier, CatBoostRegressor]], pd.DataFrame]: A tuple containing:
            - A list of trained CatBoost models.
            - A dataframe containing the model performance metrics.
            - A dataframe containing the feature importances.
    """

    # Default Model Parameters
    default_model_params = {
        'iterations': 1000,
        'learning_rate': 0.05,
        'depth': 6,
        'loss_function': 'Logloss',
        'early_stopping_rounds': 50,
        'od_type': 'Iter',
        'task_type': 'CPU',
        'thread_count': -1,
        'verbose': 100,
        'random_seed': 42
    }

    # Default TimeSeriesSplit Parameters
    default_tscv_params = {
        'n_splits': 5,
        'test_size': None,
        'gap': 0
    }

    # Update default parameters with user-defined values
    model_params = { **default_model_params, **( model_params or {} ) }
    tscv_params = { **default_tscv_params, **( tscv_params or {} ) }
    custom_metrics = custom_metrics or [ "Precision", "Recall" ]

    # Output Paths
    model_dir = Path( output_dir ) / experiment_name
    model_dir.mkdir( parents = True, exist_ok = True )

    tscv = TimeSeriesSplit( **tscv_params )
    models = []
    metrics = []
    feature_importances = []
    
    if walk_forward_CV:
        logging.info( "Using Walk-Forward Cross-Validation...")
        timestamps = list( X.index )
        n_splits = tscv_params[ 'n_splits' ]
        test_size = tscv_params.get( 'test_size', X.shape[ 0 ] // n_splits)
        train_size = X.shape[ 0 ]  - n_splits * test_size

        for fold in range(n_splits):
            print(f"\n{'='*40}")
            print(f"Training Fold {fold+1}/{n_splits}")

            train_start_idx = fold * test_size
            train_end_idx = (fold + 1) * test_size + train_size

            train_index = X.index[ train_start_idx : train_end_idx ]
            test_index = X.index[ train_end_idx : min( train_end_idx + test_size, len( X.index ) ) ]

            if len( test_index ) == 0:
                break  # Stop if no test data left

            print(f"Train period: {train_index[0]} to {train_index[-1]}")
            print(f"Test period: {test_index[0]} to {test_index[-1]}")
            print("="*40)

            # Get train, val data and test data
            train_mask = X.index.isin( train_index )
            test_mask  = X.index.isin( test_index )
            X_train, X_test = X[ train_mask ], X[ test_mask ]
            y_train, y_test = y[ train_mask ], y[ test_mask ]

            train_pool = Pool( X_train, y_train, cat_features = cat_features )
            test_pool   = Pool( X_test, y_test, cat_features = cat_features )
            # test_pool = Pool(X_test, y_test, cat_features=cat_features)

            model = CatBoostClassifier( **model_params )
            model.fit(
                train_pool,
                eval_set = test_pool,
                use_best_model=True,
                plot = False,
                verbose = model_params['verbose'],
                **training_config
            )

            # Model Evaluation
            eval_results = {}
            pred_proba = model.predict_proba( X_test )[ :, 1 ] 
            y_pred = ( pred_proba > 0.5 ).astype( int )
            temp_df = pd.DataFrame( { 'Close': X_test[ "Close" ], 'True': y_test, 'Pred': y_pred, 'Pred_prob': pred_proba }, index = y_test.index )
            pos_cases = ( y_test == 1 ).sum()
            y_pred_pos = ( y_pred == 1 ).sum()
            eval_results[ 'Positive Cases' ] = pos_cases
            eval_results[ 'Predicted Positives' ] = y_pred_pos

            if 'Accuracy' in custom_metrics:
                eval_results[ 'Accuracy' ] = accuracy_score( y_test, y_pred )
            if "Precision" in custom_metrics:
                eval_results[ 'Precision' ] = precision_score( y_test, y_pred )
            if "Recall" in custom_metrics:
                eval_results[ 'Recall' ] = recall_score( y_test, y_pred )
            if 'AUC' in custom_metrics:
                from sklearn.metrics import roc_auc_score
                eval_results[ 'AUC' ] = roc_auc_score(y_test, pred_proba)
            if 'F1' in custom_metrics:
                from sklearn.metrics import f1_score
                eval_results[ 'F1' ] = f1_score(y_test, (pred_proba > 0.5).astype(int))

            metrics.append({
                'fold': fold+1,
                'train_start': train_index[ 0 ],
                'train_end':   train_index[ -1 ],
                'test_start':  test_index[ 0 ],
                'test_end':    test_index[ -1 ],
                **eval_results
            })


            # Feature Importances
            fi = model.get_feature_importance()
            feature_names = model.feature_names_
            feature_importances_df = pd.DataFrame( {
                                                    "feature": feature_names,
                                                    "Importance": fi,            
                                                }).sort_values('Importance', ascending = False )
            feature_importances.append( feature_importances_df )

            # Save Model
            model.save_model( model_dir / f"{model_name}_model_{ fold + 1 }_WFCV.cbm" )
            models.append( model )

            # Save Model Metrics
            pd.DataFrame(metrics).to_csv(model_dir / "model_metrics.csv", index=False)

    
    else: # TimeSeriesSplit
        logging.info( "Using TimeSeriesSplit..." )
        for fold, ( train_index, test_index ) in enumerate( tscv.split( X ) ):
            print( f"\n{'='*40}" )
            print( f"Training Fold {fold+1}/{tscv.get_n_splits()}" )
            print( f"Train period: {X.index[train_index[0]]} to {X.index[train_index[-1]]}" )
            print( f"Test period:  {X.index[test_index[0]]} to {X.index[test_index[-1]]}" )
            print( '='*40 )

            X_train, X_test = X.iloc[ train_index ], X.iloc[ test_index ]
            y_train, y_test = y.iloc[ train_index ], y.iloc[ test_index ]

            train_pool = Pool( X_train, y_train, cat_features = cat_features )
            test_pool  = Pool( X_test, y_test, cat_features = cat_features )

            model = CatBoostClassifier( **model_params )
            # training
            model.fit(
                train_pool,
                eval_set = test_pool,
                use_best_model = True,
                plot = False,
                verbose = model_params[ 'verbose' ],
                **training_config
            )

            # Model Evaluation
            eval_results = {}
            pred_proba = model.predict_proba( X_test )[ :, 1 ]
            y_pred = ( pred_proba > 0.5 ).astype( int )
            temp_df = pd.DataFrame( { 'Close': X_test[ "Close" ], 'True': y_test, 'Pred': y_pred, 'Pred_prob': pred_proba }, index = y_test.index )
            pos_cases = ( y_test == 1 ).sum()
            y_pred_pos = ( y_pred == 1 ).sum()
            eval_results[ 'Positive Cases' ] = pos_cases
            eval_results[ 'Predicted Positives' ] = y_pred_pos

            if 'Accuracy' in custom_metrics:
                eval_results[ 'Accuracy' ] = accuracy_score( y_test, y_pred )
            if "Precision" in custom_metrics:
                eval_results[ 'Precision' ] = precision_score( y_test, y_pred )
            if "Recall" in custom_metrics:
                eval_results[ 'Recall' ] = recall_score( y_test, y_pred )
            if 'AUC' in custom_metrics:
                from sklearn.metrics import roc_auc_score
                eval_results[ 'AUC' ] = roc_auc_score(y_test, pred_proba)
            if 'F1' in custom_metrics:
                from sklearn.metrics import f1_score
                eval_results[ 'F1' ] = f1_score(y_test, (pred_proba > 0.5).astype(int))

            metrics.append({
                'fold': fold+1,
                'train_start': X.index[ train_index[0] ],
                'train_end':   X.index[ train_index[-1] ],
                'test_start':  X.index[ test_index[0] ],
                'test_end':    X.index[ test_index[-1] ],
                **eval_results
            })

            # Feature Importances
            fi = model.get_feature_importance()
            feature_names = model.feature_names_
            feature_importances_df = pd.DataFrame( {
                                                    "feature": feature_names,
                                                    "Importance": fi,            
                                                }).sort_values('Importance', ascending = False )
            feature_importances.append( feature_importances_df )

            # Save Model
            model.save_model( model_dir / f"{model_name}_model_{ fold + 1 }.cbm" )
            models.append( model )

            # Save Model Metrics
            pd.DataFrame( metrics ).to_csv( model_dir / "model_metrics.csv", index = False )


    # Summarize Model Performance
    metrics_df = pd.DataFrame( metrics )
    print( "\n Training Complete!" )
    print( f"Model Metrics: \n{metrics_df}" )

    return models, metrics_df, feature_importances
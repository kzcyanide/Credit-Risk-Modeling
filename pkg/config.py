# Configuration class:-
from sklearn import set_config; 
set_config(transform_output = "pandas")
class Config:
    """
    Configuration class for parameters and CV strategy for tuning and training
    Some parameters may be unused here as this is a general configuration class
    """
    
    # Data preparation:-
    test_req           = "N"
    test_sample_frac   = 0.025
    gpu_switch         = "OFF"
    state              = 42
    target             = "Approved_Flag"
    path               = f"./input/"
    #orig_path         = 
    features           = features = ['pct_tl_open_L6M','pct_tl_closed_L6M','Tot_TL_closed_L12M','pct_tl_closed_L12M','Tot_Missed_Pmnt',
                                     'CC_TL','Home_TL','PL_TL','Secured_TL','Unsecured_TL','Other_TL','Age_Oldest_TL','Age_Newest_TL',
                                     'time_since_recent_payment','max_recent_level_of_deliq','num_deliq_6_12mts','num_times_60p_dpd',
                                     'num_std_12mts','num_sub','num_sub_6mts','num_sub_12mts','num_dbt','num_dbt_12mts','num_lss',
                                     'recent_level_of_deliq','CC_enq_L12m','PL_enq_L12m','time_since_recent_enq','enq_L3m','NETMONTHLYINCOME',
                                     'Time_With_Curr_Empr','CC_Flag','PL_Flag','pct_PL_enq_L6m_of_ever','pct_CC_enq_L6m_of_ever','HL_Flag',
                                     'GL_Flag','MARITALSTATUS','EDUCATION','GENDER','last_prod_enq2','first_prod_enq2','Approved_Flag'
                                     ]
    
    dtl_preproc_req    = "N"
    ftre_plots_req     = 'N'
    ftre_imp_req       = "N"
    
    # Data transforms and scaling:-    
    #conjoin_orig_data  = "N"  

    drop_nulls         = "N"
    sec_ftre_req       = "N"
    scale_req          = "N"
    
    # Model Training:- 
    pstprcs_oof        = "N"
    pstprcs_train      = "N"
    pstprcs_test       = "N"
    ML                 = "Y"
    
    #kaagle
    #pseudo_lbl_req     = "N"
    #pseudolbl_up       = 0.975
    #pseudolbl_low      = 0.00
    
    n_splits           = 5 if test_req == "Y" else 5
    n_repeats          = 1 
    nbrnd_erly_stp     = 100
    mdlcv_mthd         = 'SKF'
    
    # Ensemble:-    
    #ensemble_req       = "N"
    #hill_climb_req     = "N"
    #optuna_req         = "Y"
    #LAD_req            = "N"
    #enscv_mthd         = "RSKF"
    #metric_obj         = 'minimize'
    #ntrials            = 10 if test_req == "Y" else 200
    
    # Global variables for plotting:-
    #grid_specs = {'visible': True, 'which': 'both', 'linestyle': '--', 
    #                       'color': 'lightgrey', 'linewidth': 0.75}
    #title_specs = {'fontsize': 9, 'fontweight': 'bold', 'color': '#992600'}

    xgb_params = {'learning_rate': 0.05097426246942082,
          'n_estimators': 732,
          'max_depth': 4,
          'subsample': 0.8222044805356123,
          'colsample_bytree': 0.9256598045647526,
          'gamma': 0.6509754269480936,
          'reg_alpha': 5.239764778705016,
          'reg_lambda': 2.6504947277703335,
          'objective': 'multi:softmax',
          'num_class': 4 
          }
    
    cb_params = {'learning_rate': 0.03747840176967439,
             'depth': 6,
             'n_estimators': 900,
             'l2_leaf_reg': 0.9918286788888944,
             'colsample_bylevel': 0.9251958707544501,
             'border_count': 192
             }
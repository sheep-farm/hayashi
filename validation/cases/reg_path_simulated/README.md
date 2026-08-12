# reg_path_simulated

Regularisation path (elastic net) on simulated data.

Notes: Simulated data where y = 0.5 + x1 + noise and x2 is noise. Hayashi reg_path selects an elastic-net lambda by BIC; references fit glmnet and sklearn ElasticNet at the same optimal lambda and standardisation.

# rf_simulated

Random forest regression on simulated data.

Simulated data y = 3*x1 + N(0, 0.1). The comparison covers only in-sample R² against scikit-learn at the unchanged `5e-3` tolerance. It does not qualify uncertainty or equivalence of the full forest fitting contract. Unselected standard-error placeholders remain outside numerical comparison.

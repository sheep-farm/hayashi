# synthdid_simulated

Synthetic DiD point-estimate proxy on simulated panel data.

Simulated panel with 20 units, 10 periods, treatment begins at period 6 for unit 0 with ATT=2.0. The references use synthetic-control-style pre-treatment weighting and compare the post-treatment mean gap at the unchanged `5e-2` tolerance.

Both references are classified as `behavioural-proxy`: agreement covers this point summary, which does not establish general synthetic DiD estimator validity, uncertainty or causal identification. Standard errors are unavailable and excluded from the comparison contract.

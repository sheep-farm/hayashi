# eventstudy_simulated

Event study with grouped standard errors on simulated data.

Notes: Simulated panel with 60 units and 5 time periods. Half the units are treated
at time 2 and the other half are never treated. The outcome has a calendar
trend and a post-treatment effect. Standard errors are clustered by unit and
compared with R (sandwich::vcovCL) and Python (statsmodels OLS with
cov_type='cluster').


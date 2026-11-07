# modwt_simulated

MODWT (Haar) on simulated time series.

Simulated series (trend + 16-period sine + noise). Greeners MODWT uses unnormalised Haar filters, equivalent to pywt.swt(..., norm=False). The comparison covers wavelet energy summaries at the unchanged `1e-12` tolerance. Standard errors are undefined and excluded from the tolerance contract; unselected export placeholders do not establish uncertainty agreement.

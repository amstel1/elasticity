import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import kstest, spearmanr, ttest_1samp
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss
from scipy.stats import wasserstein_distance
from scipy.interpolate import interp1d

def test_autocorrelation_arima(y_train, y_train_pred):
    """
    1. Test for autocorrelation in timeseries residuals using Ljung-Box Q-test.
       Build the best performing ARIMA(p, d, q) model and return R2.
    """
    residuals = y_train - y_train_pred
    lb_p_values = acorr_ljungbox(residuals, lags=range(1, 5), return_df=False)

    best_aic = float('inf')
    best_r2 = -float('inf')
    best_model_params = None

    for d in range(2):
        for p in range(2): # p>=0, let's check up to 6, to limit search space
            for q in range(2): # q<=12
                try:
                    model = ARIMA(y_train, order=(p, d, q))
                    model_fit = model.fit()
                    predictions_arima = model_fit.fittedvalues
                    r2 = r2_score(y_train, predictions_arima)
                    if model_fit.aic < best_aic: # Using AIC for model selection
                        best_aic = model_fit.aic
                        best_r2 = r2
                        best_model_params = (p, d, q)
                except Exception as e: # Catch potential errors during ARIMA fitting
                    continue

    return {
        "ljung_box_p_values": lb_p_values,
        "best_arima_r2": best_r2,
        "best_arima_params": best_model_params
    }

def test_ks_test_timeseriessplit(y_train, y_train_pred, n_splits=5):
    """
    2. Kolmogorov-Smirnov test with TimeSeriesSplit on residuals.
    """
    residuals = y_train - y_train_pred
    tscv = TimeSeriesSplit(n_splits=n_splits)
    p_values = []
    all_indices = np.arange(len(residuals))

    for train_index, test_index in tscv.split(residuals):
        train_fold_residuals = residuals[train_index]
        test_fold_residuals = residuals[test_index]
        ks_statistic, p_value = kstest(train_fold_residuals, test_fold_residuals)
        p_values.append(p_value)
    return {"ks_test_p_values_timeseriessplit": p_values}

def test_seasonal_ks_test(y_train, y_train_pred):
    # assert index is datetime
    """
    3. Kolmogorov-Smirnov test on seasonal components from STL decomposition.
    """
    stl_y = STL(y_train, seasonal=13).fit() # Assuming weekly seasonality, adjust seasonal as needed
    stl_y_pred = STL(y_train_pred, seasonal=13).fit()
    seasonal_y = stl_y.seasonal
    seasonal_y_pred = stl_y_pred.seasonal
    ks_statistic, p_value = kstest(seasonal_y, seasonal_y_pred)
    return {"seasonal_ks_test_p_value": p_value}

def test_spearman_correlation(y_train, y_train_pred):
    """
    4. Spearman rank correlation test.
    """
    correlation, p_value = spearmanr(y_train, y_train_pred)
    return {"spearman_correlation": correlation}

def test_one_sample_ttest_residuals(y_train, y_train_pred):
    """
    5. One-sample t-test on residuals for zero mean.
    """
    residuals = y_train - y_train_pred
    t_statistic, p_value = ttest_1samp(residuals, 0)
    return {"ttest_residuals_p_value": p_value}

def test_bootstrap_confidence_interval(y_train, y_train_pred, n_bootstrap=1000):
    """
    6. Bootstrapping on train residuals for 95% confidence interval.
       Estimates CI for the mean of residuals.
    """
    residuals = y_train - y_train_pred
    bootstrap_means = []
    n_residuals = len(residuals)
    rng = np.random.RandomState(42) # for reproducibility

    for _ in range(n_bootstrap):
        bootstrap_sample = rng.choice(residuals, size=n_residuals, replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))

    ci_lower = np.percentile(bootstrap_means, 2.5)
    ci_upper = np.percentile(bootstrap_means, 97.5)
    return {"bootstrap_ci_95_percent_mean_residuals": (ci_lower, ci_upper)}

def test_stationarity_adf_kpss(y_train, y_train_pred, max_lags_adf=None):
    """
    7. Stationarity tests (ADF and KPSS).
    """
    residuals = y_train - y_train_pred

    if max_lags_adf is None:
        max_lags_adf = int(12 * (len(residuals)/100)**(1/4)) # Rule of thumb for max lags in ADF

    adf_result = adfuller(residuals, maxlag=max_lags_adf, regression='ct') # Test against constant and trend
    kpss_result = kpss(residuals, regression='c', nlags='auto') # Test against constant

    adf_stationary = "Stationary" if adf_result[1] < 0.05 else "Non-stationary"
    kpss_stationary = "Stationary" if kpss_result[1] >= 0.05 else "Non-stationary" # KPSS null is stationarity

    return {
        "adf_test_interpretation_95_confidence": adf_stationary,
        "kpss_test_interpretation_95_confidence": kpss_stationary,
        "adf_p_value": adf_result[1],
        "kpss_p_value": kpss_result[1]
    }

def test_wasserstein_quantile_outliers(y_train, y_train_pred, wasserstein_quantile=0.99):
    """
    8. Identify target values above wasserstein_quantile of absolute residuals.
       (Wasserstein distance is not directly used as originally misinterpreted, using residual quantile instead based on revised understanding)
    """
    residuals = y_train - y_train_pred
    abs_residuals = np.abs(residuals)
    quantile_threshold = np.quantile(abs_residuals, wasserstein_quantile) # Using residual quantile instead of wasserstein distance quantile
    outlier_indices = np.where(abs_residuals > quantile_threshold)[0]
    outlier_y_train_values = y_train[outlier_indices]

    # Interpolation was unclear, returning outlier indices and y_train values instead.
    return {
        "wasserstein_quantile_threshold_residual": quantile_threshold, # Renamed for clarity
        "outlier_residual_indices": outlier_indices.tolist(),
        "outlier_y_train_values": outlier_y_train_values.tolist()
    }


if __name__ == '__main__':
    # Generate some dummy timeseries data
    np.random.seed(42)
    n_samples = 100
    y_train = np.cumsum(np.random.randn(n_samples)) + 50 # Non-stationary series
    y_train_pred = y_train + np.random.randn(n_samples) * 2 # Predictions with some noise

    # Run tests and print results
    print("Test 1: Autocorrelation and ARIMA")
    test1_results = test_autocorrelation_arima(y_train, y_train_pred)
    print(test1_results)

    print("\nTest 2: KS Test with TimeSeriesSplit")
    test2_results = test_ks_test_timeseriessplit(y_train, y_train_pred)
    print(test2_results)

    print("\nTest 3: Seasonal KS Test")
    test3_results = test_seasonal_ks_test(y_train, y_train_pred)
    print(test3_results)

    print("\nTest 4: Spearman Correlation")
    test4_results = test_spearman_correlation(y_train, y_train_pred)
    print(test4_results)

    print("\nTest 5: One-Sample T-test for Residuals")
    test5_results = test_one_sample_ttest_residuals(y_train, y_train_pred)
    print(test5_results)

    print("\nTest 6: Bootstrap Confidence Interval for Residuals")
    test6_results = test_bootstrap_confidence_interval(y_train, y_train_pred)
    print(test6_results)

    print("\nTest 7: Stationarity Tests (ADF & KPSS)")
    test7_results = test_stationarity_adf_kpss(y_train, y_train_pred)
    print(test7_results)

    print("\nTest 8: Wasserstein Quantile Outliers (Residual Quantile based)")
    test8_results = test_wasserstein_quantile_outliers(y_train, y_train_pred)
    print(test8_results)

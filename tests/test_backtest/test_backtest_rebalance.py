import pandas as pd
from skportfolio.backtest.rebalance import prepare_rebalance_signal


def test_false_values_except_first():
    index = pd.date_range("2020-01-01", "2020-02-29", freq="D")
    rebalance_frequency_or_signal = True
    expected_result = pd.Series(index=index, data=True)
    expected_result.iloc[0] = False
    result, _ = prepare_rebalance_signal(rebalance_frequency_or_signal, index)
    pd.testing.assert_series_equal(result, expected_result)


def test_true_values_at_every_nth():
    index = pd.date_range("2020-01-01", "2020-02-29", freq="D")
    rebalance_frequency_or_signal = 5
    expected_result = pd.Series(index=index, data=False)
    expected_result.iloc[::5] = True
    expected_result.iloc[0] = False
    result, _ = prepare_rebalance_signal(rebalance_frequency_or_signal, index)
    pd.testing.assert_series_equal(result, expected_result)


def test_false_values_when_input_is_none():
    index = pd.date_range("2020-01-01", "2020-02-29", freq="D")
    rebalance_frequency_or_signal = None
    expected_result = pd.Series(index=index, data=False)
    result, _ = prepare_rebalance_signal(rebalance_frequency_or_signal, index)
    pd.testing.assert_series_equal(result, expected_result)


def test_false_values_when_input_is_false():
    index = pd.date_range("2020-01-01", "2020-02-29", freq="D")
    rebalance_frequency_or_signal = False
    expected_result = pd.Series(index=index, data=False)
    result, _ = prepare_rebalance_signal(rebalance_frequency_or_signal, index)
    pd.testing.assert_series_equal(result, expected_result)


def test_generate_rebalance_on_month_end():
    index = pd.date_range("2020-01-01", "2020-02-29", freq="D")
    rebalance_frequency_or_signal = "M"
    expected_result = pd.Series(index=index, data=False)
    expected_result.loc["2020-01-31"] = True
    expected_result.loc["2020-02-29"] = True

    result, _ = prepare_rebalance_signal(rebalance_frequency_or_signal, index)
    pd.testing.assert_series_equal(result, expected_result)


def test_generate_rebalance_on_month_start():
    index = pd.date_range("2020-01-01", "2020-02-28", freq="D")
    rebalance_frequency_or_signal = "MS"
    expected_result = pd.Series(index=index, data=False)
    expected_result.loc["2020-01-01"] = False
    expected_result.loc["2020-02-01"] = True

    result, _ = prepare_rebalance_signal(rebalance_frequency_or_signal, index)
    pd.testing.assert_series_equal(result, expected_result)

"""
Convenience copy of the empyrical constants, with more constants added 
"""

"""
In many applications, it is necessary to convert Sharpe ratio estimates from one frequency to
another. For example, a Sharpe ratio estimated from monthly data cannot be directly compared with one
estimated from annual data; hence, one statistic must be converted to the same frequency as the other
to yield a fair comparison. Moreover, in some cases, it is possible to derive a more precise estimator of an
annual quantity by using monthly or daily data and then performing time aggregation instead of estimating
the quantity directly using annual data.
In the case of Sharpe ratios, the most common method for performing such time aggregation is to
multiply the higher-frequency Sharpe ratio by the square root of the number of periods contained in
the lower-frequency holding period (e.g., multiply a monthly estimator by 12 to obtain an annual estimator).
"""

APPROX_BDAYS_PER_YEAR = 252
APPROX_DAYS_PER_YEAR = 365
DAYS_PER_WEEK = 7

APPROX_BDAYS_PER_MONTH = 21
APPROX_DAYS_PER_MONTH = 30

MONTHS_PER_YEAR = 12
WEEKS_PER_YEAR = 52
MONTHS_PER_QUARTER = 3
QTRS_PER_YEAR = 4

DAILY = "daily"
WEEKLY = "weekly"
MONTHLY = "monthly"
QUARTERLY = "quarterly"
YEARLY = "yearly"

ANNUALIZATION_FACTORS = {
    DAILY: APPROX_BDAYS_PER_YEAR,
    WEEKLY: WEEKS_PER_YEAR,
    MONTHLY: MONTHS_PER_YEAR,
    QUARTERLY: QTRS_PER_YEAR,
    YEARLY: 1,
}

FREQUENCIES = {
    "D": 1,
    "W": DAYS_PER_WEEK,
    "M": APPROX_BDAYS_PER_MONTH,
    "Q": APPROX_BDAYS_PER_MONTH * MONTHS_PER_QUARTER,
    "Y": APPROX_BDAYS_PER_YEAR,
}

BASE_TARGET_RISK = 0.02
BASE_TARGET_RETURN = 0.02
BASE_RISK_FREE_RATE = 0.02
BASE_MIN_ACCEPTABLE_RETURN = 0.02

MIN_MIN_WEIGHT_BOUND = 0
MIN_MAX_WEIGHT_BOUND = 0.2

MAX_MIN_WEIGHT_BOUND = 0.8
MAX_MAX_WEIGHT_BOUND = 1

N_WEIGHTS_BOUNDS = 3

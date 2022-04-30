"""
Contains definition of expected risk estimators, where risk is typically
measured as a square matrix of asset inter-variability.
"""

import warnings

import numpy as np
import pandas as pd
from pypfopt import risk_models as riskmod
from scipy.cluster.hierarchy import average
from scipy.cluster.hierarchy import complete
from scipy.cluster.hierarchy import single
from scipy.linalg import eigh
from sklearn.covariance import graphical_lasso

from skportfolio._constants import APPROX_BDAYS_PER_YEAR
from skportfolio.riskreturn.covariance.utils import _corr_to_cov
from skportfolio.riskreturn.covariance.utils import _cov_to_corr
from skportfolio.riskreturn.covariance.utils import _denoised_corr
from skportfolio.riskreturn.covariance.utils import _denoised_corr_spectral
from skportfolio.riskreturn.covariance.utils import _denoised_corr_targ_shrink
from skportfolio.riskreturn.covariance.utils import _detoned_corr
from skportfolio.riskreturn.covariance.utils import _find_max_eval
from skportfolio.riskreturn.covariance.utils import _get_pca


def sample_covariance(
    prices: pd.DataFrame, returns_data=False, frequency=APPROX_BDAYS_PER_YEAR
) -> pd.DataFrame:
    """
    Sample covariance
    Parameters
    ----------
    prices
    returns_data
    frequency

    Returns
    -------

    """
    return riskmod.sample_cov(
        prices,
        returns_data=returns_data,
        frequency=frequency,
        fix_method="spectral",
    )


def semicovariance(
    prices: pd.DataFrame, returns_data=False, frequency=APPROX_BDAYS_PER_YEAR
) -> pd.DataFrame:
    """
    Returns semicovariance

    Parameters
    ----------
    prices
    returns_data
    frequency

    Returns
    -------

    """
    return riskmod.semicovariance(
        prices,
        returns_data,
        benchmark=0,  # 1.02 ** (1 / frequency) - 1,
        frequency=frequency,
    )


def covariance_exp(
    prices: pd.DataFrame,
    returns_data: int = False,
    frequency: int = APPROX_BDAYS_PER_YEAR,
    span: int = 60,
) -> pd.DataFrame:
    """
    Exponentially weighted covariance matrix

    Parameters
    ----------
    prices
    returns_data
    frequency
    span

    Returns
    -------

    """
    return riskmod.exp_cov(
        prices, returns_data=returns_data, span=span, frequency=frequency
    )


def covariance_ledoit_wolf(
    prices: pd.DataFrame,
    returns_data=False,
    frequency=APPROX_BDAYS_PER_YEAR,
    shrinkage_target="constant_variance",
):
    """
    Shrinked covariance using the Ledoit-Wolf prior.
    Parameters
    ----------
    prices
    returns_data
    frequency
    shrinkage_target

    Returns
    -------

    """
    return riskmod.CovarianceShrinkage(
        prices, returns_data, frequency=frequency
    ).ledoit_wolf(shrinkage_target)


def covariance_oracle_approx(
    prices: pd.DataFrame, returns_data=False, frequency=APPROX_BDAYS_PER_YEAR
) -> pd.DataFrame:
    """
    Assuming that the data are Gaussian distributed, Chen et al. derived a formula aimed at choosing a shrinkage
    coefficient alpha that yields a smaller Mean Squared Error than the one given by Ledoit and Wolf’s formula
    This method wraps the Oracle Shrinkage approximation from PyPortfolioOpt.

    Parameters
    ----------
    prices: pd.DataFrame
    returns_data: bool
    frequency: int
    Returns
    -------
    The shrinked covariance matrix
    """
    return riskmod.CovarianceShrinkage(
        prices, returns_data
    ).oracle_approximating() * np.sqrt(frequency)


def covariance_glasso(
    prices: pd.DataFrame,
    returns_data=False,
    frequency=APPROX_BDAYS_PER_YEAR,
    alpha=0.5,
) -> pd.DataFrame:
    """
    The GraphicalLasso estimator uses an l1 penalty to enforce sparsity on the precision matrix:
    the higher its alpha parameter, the more sparse the precision matrix.

    Parameters
    ----------
    prices
    returns_data
    frequency
    alpha

    Returns
    -------

    """
    assets = prices.columns
    sample_cov = sample_covariance(
        prices, returns_data=returns_data, frequency=frequency
    ).values
    est_cov = graphical_lasso(sample_cov, alpha=alpha)[0]
    return pd.DataFrame(index=assets, columns=assets, data=est_cov)


def correlation_rmt(
    prices: pd.DataFrame, returns_data=False, frequency=APPROX_BDAYS_PER_YEAR
) -> pd.DataFrame:
    """
    FinRMT uses Random Matrix Theory (RMT) to create a filtered correlation
    matrix from a set of financial time series price data, for example the
    daily closing prices of the stocks in the S&P
     Syntax
    corr_rmt_filtered=FinRMT(priceTS)
             Description
    This function eigendecomposes a correlation matrix of time series
    and splits it into three components, Crandom, Cgroup and Cmarket,
    according to techniques from literature (See, "Systematic Identification
    of Group Identification in Stock Markets, Kim & Jeong, (2008).") and
    returns a filtered correlation matrix containging only the Cgroup
    components.
    The function is intended to be used in conjunction with a community
    detection algorithm (such as the Louvain method) to allow for community
    detecion on time series based networks.
                     Inputs arguments:
    priceTS : an mxn matrix containing timeseries' of stock prices. Each column
    should be a time series for one financial instrument and each row should
    correspond to the value of each instrument at a point in time. For example
    32.00   9.43   127.25   ...
    32.07   9.48   126.98   ...
    32.08   9.53   126.99   ...
     ...    ...     ....    ...
    No header columns or timestamp columns should be included
             Outputs:
    corr_rmt_filtered : The filtered correlation matrix. This matrix can be passed directly to
    a community detection algorithm in place of the modularity matrix
             Example:
    ModularityMatrix = FinRMT(myPriceData)
     ...
    Communities = myCommunityDectionAlg(ModularityMatrix)

    Issues & Comments
    Note that the output of this function can serve as the Modularity
    Matrix (Not the Adjacency matrix) for a generalized Community Detection
    Algorithm. Specifically, one which does not rely on properties of the
    Adjaceny Matrix to create the Modularity Matrix. The Louvain Method
    and methods based on spectral decompositon are examples of such.

    Parameters
    ----------
    prices: pd.DataFrame
        prices of assets
    returns_data: bool
        True if providing returns data instead of prices
    frequency: int
        Default APPROX_BDAYS_PER_YEAR, 252
    Returns
    -------
    Estimated covariance matrix
    """
    [n_samples, n_assets] = prices.shape
    sample_cov = riskmod.sample_cov(
        prices, returns_data=returns_data, frequency=frequency
    )
    corr = riskmod.cov_to_corr(sample_cov)
    # Fix if not exactly symmetric
    corr = (corr + corr.T) / 2

    # Decompose the correlation matrix into its eigenvalues and eigenvectors,
    # store the indices of which columns the sorted eigenvalues come from
    # and arrange the columns in this order

    eigvals, eigvec = eigh(corr)
    eigvec_diag = np.diag(eigvals)

    q_ratio = n_samples / n_assets
    sigma = 1 - np.max(eigvec_diag) / n_assets

    # Find the index of the predicted lambda_max, ensuring to check boundary
    # conditions
    rmt_mag_eig = sigma * (1 + (1.0 / q_ratio) + 2 * np.sqrt(1 / q_ratio))
    rmt_max_idx = np.nonzero(eigvals > rmt_mag_eig)[0]
    if len(rmt_max_idx) == 0:
        rmt_max_idx = n_assets - 1
    else:
        rmt_max_idx = rmt_max_idx[0]

    # Find the index of the predicted lambda_min, ensuring the check boundary
    # conditions
    rmt_min_eig = sigma * (1 + (1.0 / q_ratio) - 2 * np.sqrt(1 / q_ratio))
    rmt_min_idx = np.nonzero(eigvals < rmt_min_eig)[0]
    if len(rmt_min_idx) == 0:
        rmt_min_idx = 0
    else:
        rmt_min_idx = rmt_min_idx[-1]

    # Determine the average Eigenvalue to rebalance the matrix after removing
    # Any of the noise and/or market mode components
    avg_eigen_value = np.mean(eigvals[rmt_min_idx : (rmt_max_idx + 1)])

    # Build a new diagonal matrix consisting of the group eigenvalues
    group_eig = np.eye(n_assets) * avg_eigen_value
    # Add the group component. The N+1 here is just used to increment to the
    # next diagonal element in the matrix
    for i in range(rmt_max_idx, n_assets - 1):
        group_eig[i, i] = eigvec_diag[i, i]

    # Build the component correlation matrix from the new diagonal eigenvalue
    # matrix and eigenvector matrix. The eigenvectors corresponding to zero
    # valued eigenvalue entries in group_eig will not contribute to corr_rmt_filtered
    # This is the matrix product V*Dg*V' but one needs multiple dot products
    corr_rmt_filtered = np.dot(np.dot(eigvec, group_eig), eigvec.T)
    # Replace the diagonals with 1s
    np.fill_diagonal(corr_rmt_filtered, 1)
    return pd.DataFrame(corr_rmt_filtered, columns=prices.columns, index=prices.columns)


def covariance_rmt(
    prices: pd.DataFrame, returns_data=False, frequency=APPROX_BDAYS_PER_YEAR
):
    """
    Calculates the covariance from the Random Matrix Theory filtered correlation matrix using the Marchenko-Pastur
    distribution.

    Parameters
    ----------
    prices: pd.DataFrame
        Prices or returns
    returns_data: bool
        Feeding prices or returns data?
    frequency:
        Annualization factor

    Returns
    -------
    The Marchenko-Pastur spectrally filtered covariance
    """
    corr = correlation_rmt(prices, returns_data, frequency)
    cov = sample_covariance(prices, returns_data, frequency)
    # Calculating the covariance matrix from the de-noised correlation matrix
    cov_denoised = _corr_to_cov(corr, np.diag(cov) ** (1 / 2))
    return cov_denoised


def denoise_covariance(
    cov,
    tn_relation,
    denoise_method="const_resid_eigen",
    detone=False,
    market_component=1,
    kde_bwidth=0.01,
    alpha=0,
):
    """
    De-noises the covariance matrix or the correlation matrix.

    Two denoising methods are supported:
    1. Constant Residual Eigenvalue Method (``const_resid_eigen``)
    2. Spectral Method (``spectral``)
    3. Targeted Shrinkage Method (``target_shrink``)


    The Constant Residual Eigenvalue Method works as follows:

    First, a correlation is calculated from the covariance matrix (if the input is the covariance matrix).

    Second, eigenvalues and eigenvectors of the correlation matrix are calculated using the linalg.eigh
    function from numpy package.

    Third, a maximum theoretical eigenvalue is found by fitting Marcenko-Pastur (M-P) distribution
    to the empirical distribution of the correlation matrix eigenvalues. The empirical distribution
    is obtained through kernel density estimation using the KernelDensity class from sklearn.
    The fit of the M-P distribution is done by minimizing the Sum of Squared estimate of Errors
    between the theoretical pdf and the kernel. The minimization is done by adjusting the variation
    of the M-P distribution.

    Fourth, the eigenvalues of the correlation matrix are sorted and the eigenvalues lower than
    the maximum theoretical eigenvalue are set to their average value. This is how the eigenvalues
    associated with noise are shrinked. The de-noised covariance matrix is then calculated back
    from new eigenvalues and eigenvectors.

    The Spectral Method works just like the Constant Residual Eigenvalue Method, but instead of replacing
    eigenvalues lower than the maximum theoretical eigenvalue to their average value, they are replaced with
    zero instead.

    The Targeted Shrinkage Method works as follows:

    First, a correlation is calculated from the covariance matrix (if the input is the covariance matrix).

    Second, eigenvalues and eigenvectors of the correlation matrix are calculated using the linalg.eigh
    function from numpy package.

    Third, the correlation matrix composed from eigenvectors and eigenvalues related to noise is
    shrunk using the alpha variable. The shrinkage is done by summing the noise correlation matrix
    multiplied by alpha to the diagonal of the noise correlation matrix multiplied by (1-alpha).

    Fourth, the shrinked noise correlation matrix is summed to the information correlation matrix.

    Correlation matrix can also be detoned by excluding a number of first eigenvectors representing
    the market component.

    These algorithms are reproduced with minor modifications from the following book:
    Marcos Lopez de Prado “Machine Learning for Asset Managers”, (2020).

    Parameters
    ----------
    cov: (np.array) Covariance matrix or correlation matrix.
    tn_relation: (float) Relation of sample length T to the number of variables N used to calculate the
                                covariance matrix.
    denoise_method: (str) Denoising methos to use. (``const_resid_eigen`` by default, ``target_shrink``)
    detone: (bool) Flag to detone the matrix. (False by default)
    market_component: (int) Number of fist eigevectors related to a market component. (1 by default)
    kde_bwidth: (float) The bandwidth of the kernel to fit KDE.
    alpha: (float) In range (0 to 1) - shrinkage of the noise correlation matrix to use in the
                          Targeted Shrinkage Method. (0 by default)

    Returns
    -------
    np.ndarray
        De-noised covariance matrix or correlation matrix.
    """

    # Correlation matrix computation (if correlation matrix given, nothing changes)

    corr = _cov_to_corr(cov)

    # Calculating eigenvalues and eigenvectors
    eigenval, eigenvec = _get_pca(corr)

    # Calculating the maximum eigenvalue to fit the theoretical distribution
    maximum_eigen, _ = _find_max_eval(np.diag(eigenval), tn_relation, kde_bwidth)

    # Calculating the threshold of eigenvalues that fit the theoretical distribution
    # from our set of eigenvalues
    num_facts = eigenval.shape[0] - np.diag(eigenval)[::-1].searchsorted(maximum_eigen)

    if denoise_method == "target_shrink":
        # Based on the threshold, de-noising the correlation matrix
        corr = _denoised_corr_targ_shrink(eigenval, eigenvec, num_facts, alpha)
    elif denoise_method == "spectral":
        # Based on the threshold, de-noising the correlation matrix
        corr = _denoised_corr_spectral(eigenval, eigenvec, num_facts)
    else:  # Default const_resid_eigen method
        # Based on the threshold, de-noising the correlation matrix
        corr = _denoised_corr(eigenval, eigenvec, num_facts)

    # Detone the correlation matrix if needed
    if detone:
        corr = _detoned_corr(corr, market_component)

    # Calculating the covariance matrix from the de-noised correlation matrix
    cov_denoised = _corr_to_cov(corr, np.diag(cov) ** (1 / 2))

    return pd.DataFrame(data=cov_denoised, columns=cov.columns, index=cov.index)


# Finding the Constant Residual Eigenvalue De-noised Сovariance matrix


def covariance_crr_denoise(
    prices: pd.DataFrame, returns_data=False, frequency=APPROX_BDAYS_PER_YEAR, **kwargs
) -> pd.DataFrame:
    """
    Estimates the covariance matrix with spectral denoising approach.
    Please see the `denoise_covariance` method.

    Parameters
    ----------
    prices: pd.DataFrame
        Prices data
    returns_data: bool
        Is the input dataframe returns or prices?
    frequency: int
    **kwargs
        kde_bwidth: float, default 0.01
        denoise_method: str, default const_resid_eigen

    Returns
    -------
    The spectrally denoised covariance matrix
    """
    tn_relation = prices.shape[0] / prices.shape[1]
    cov = sample_covariance(prices, returns_data, frequency)
    return denoise_covariance(
        cov=cov,
        tn_relation=tn_relation,
        denoise_method=kwargs.get("denoise_method", "const_resid_eigen"),
        detone=kwargs.get("detone", False),
        kde_bwidth=kwargs.get("kde_bwidth", 0.01),
    )


def covariance_denoise_spectral(
    prices: pd.DataFrame, returns_data=False, frequency=APPROX_BDAYS_PER_YEAR, **kwargs
) -> pd.DataFrame:
    """
    Estimates the covariance matrix with spectral denoising approach.
    Please see the `denoise_covariance` method.
    Parameters
    ----------
    prices: pd.DataFrame
    returns_data: bool
    frequency: int
    **kwargs
        denoise_method: str, default spectral
    Returns
    -------
    The spectrally denoised covariance matrix
    """
    tn_relation = prices.shape[0] / prices.shape[1]
    cov = sample_covariance(prices, returns_data, frequency)
    return denoise_covariance(
        cov=cov,
        tn_relation=tn_relation,
        denoise_method=kwargs.get("denoise_method", "spectral"),
    )


def covariance_target_shrinkage_denoised(
    prices: pd.DataFrame, returns_data=False, frequency=APPROX_BDAYS_PER_YEAR, **kwargs
) -> pd.DataFrame:
    """
    Estimates the covariance matrix with shrinkage approach.
    Please see the `denoise_covariance` method.
    Parameters
    ----------
    prices: pd.DataFrame
    returns_data: bool
    frequency: int
    **kwargs
        kde_bwidth: float, default 0.01
        denoise_method: str, default target_shrink
    Returns
    -------
    pd.DataFrame
        The shrinked, using the target shrink options of denoising method covariance matrix
    """
    tn_relation = prices.shape[0] / prices.shape[1]
    cov = sample_covariance(prices, returns_data, frequency)
    return denoise_covariance(
        cov,
        tn_relation,
        denoise_method=kwargs.get("denoise_method", "target_shrink"),
        detone=False,
        kde_bwidth=kwargs.get("kde_bwidth", 0.01),
    )


def covariance_crr_denoise_detoned(
    prices: pd.DataFrame, returns_data=False, frequency=APPROX_BDAYS_PER_YEAR, **kwargs
) -> pd.DataFrame:
    """
    Estimates the covariance matrix while trying to detone and denoise it.
    Please see the `denoise_covariance` method.
    Parameters
    ----------
    prices: pd.DataFrame
    returns_data: bool
    frequency: int
    kwargs

    Returns
    -------
    The denoise and detoned covariance matrix
    """
    tn_relation = prices.shape[0] / prices.shape[1]
    cov = sample_covariance(prices, returns_data, frequency)
    return denoise_covariance(
        cov,
        tn_relation,
        denoise_method=kwargs.get("denoise_method", "const_resid_eigen"),
        detone=kwargs.get("detone", True),
        market_component=1,
        kde_bwidth=kwargs.get("kde_bwidth", 0.01),
    )


def filter_corr_hierarchical(cor_matrix, method="complete") -> pd.DataFrame:
    """
    Creates a filtered correlation matrix using hierarchical clustering methods from an empirical
    correlation matrix, given that all values are non-negative [0 ~ 1]

    This function allows for three types of hierarchical clustering - complete, single, and average
    linkage clusters. Link to hierarchical clustering methods documentation:
    `<https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html>`_

    It works as follows:

    First, the method creates a hierarchical clustering tree using scipy's hierarchical clustering methods
    from the empirical 2-D correlation matrix.

    Second, it extracts and stores each cluster's filtered value (alpha) and assigns it to it's corresponding leaf.

    Finally, we create a new filtered matrix by assigning each of the correlations to their corresponding
    parent node's alpha value.

    Parameters
    ----------
    cor_matrix: (np.array) Numpy array of an empirical correlation matrix.
    method: (str) Hierarchical clustering method to use. (``complete`` by default, ``single``, ``average``)

    Returns
    -------
    np.array
    The filtered correlation matrix.
    """

    # Check if all matrix elements are positive
    if np.any(cor_matrix < 0):
        warnings.warn(
            "Not all elements in matrix are positive... Returning unfiltered matrix.",
            UserWarning,
        )
        return cor_matrix

    # Check if matrix is 2-D
    if len(cor_matrix.shape) == 2:
        cor_x, cor_y = cor_matrix.shape
    else:
        warnings.warn(
            "Invalid matrix dimensions, input must be 2-D array... Returning unfiltered matrix.",
            UserWarning,
        )
        return cor_matrix

    # Check if matrix dimensions and diagonal values are valid.
    if cor_x == cor_y and np.allclose(
        np.diag(cor_matrix), 1
    ):  # using np.allclose as diag values might be 0.99999
        # Creating new coorelation condensed matrix for the upper triangle and dismissing the diagnol.
        new_cor = cor_matrix.values[np.triu_indices(cor_matrix.shape[0], k=1)]
    else:
        warnings.warn(
            "Invalid matrix, input must be a correlation matrix of size (m x m)... Returning unfiltered matrix.",
            UserWarning,
        )
        return cor_matrix

    # Compute the hierarchical clustering tree
    if method == "complete":
        z_cluster = complete(new_cor)
    elif method == "single":
        z_cluster = single(new_cor)
    elif method == "average":
        z_cluster = average(new_cor)
    else:
        warnings.warn(
            "Invalid method selected, please check docstring... Returning unfiltered matrix.",
            UserWarning,
        )
        return cor_matrix

    # Creates a pd.DataFrame that will act as a dictionary where the index is the leaf node id, and the values are
    # thier corresponding cluster's alpha value
    alpha_values = z_cluster[:, 2]
    alphas = z_cluster[:, 0]
    df_alphas = pd.DataFrame(alpha_values, index=alphas)
    df_alphas.loc[z_cluster[0][1]] = alpha_values[0]

    # Creates the filtered correlation matrix
    alphas_sorterd = df_alphas.sort_index()
    alphas_x = np.tile(alphas_sorterd.values, (1, len(alphas_sorterd.values)))
    filt_corr = np.maximum(alphas_x, alphas_x.T)
    np.fill_diagonal(filt_corr, 1)

    return pd.DataFrame(filt_corr, index=cor_matrix.index, columns=cor_matrix.columns)


def covariance_hierarchical_filter_complete(
    prices: pd.DataFrame,
    returns_data=False,
    frequency: int = APPROX_BDAYS_PER_YEAR,
):
    """
    Wrapper around filter_corr_hierarchical.
    Parameters
    ----------
    prices: pd.DataFrame
    returns_data: bool
    frequency: int

    Returns
    -------
    np.ndarray
    Covariance matrix estimated through hiearchical correlation filtering
    """

    cov = sample_covariance(prices, returns_data, frequency=frequency)
    corr_filt = filter_corr_hierarchical(cov, method="complete")
    return _corr_to_cov(corr_filt, np.sqrt(np.diagonal(cov)))


def covariance_hierarchical_filter_single(
    prices: pd.DataFrame,
    returns_data=False,
    frequency: int = APPROX_BDAYS_PER_YEAR,
):
    """
    Wrapper around filter_corr_hierarchical with single linkage method
    Parameters
    ----------
    prices: pd.DataFrame
    returns_data: bool
    frequency: int


    Returns
    -------
    np.array
        Covariance matrix estimated through hiearchical correlation filtering
    """

    cov = sample_covariance(prices, returns_data, frequency=frequency)
    corr_filt = filter_corr_hierarchical(cov, method="single")
    return _corr_to_cov(corr_filt, np.sqrt(np.diagonal(cov)))


def covariance_hierarchical_filter_average(
    prices: pd.DataFrame,
    returns_data=False,
    frequency: int = APPROX_BDAYS_PER_YEAR,
) -> np.ndarray:
    """
    Wrapper around filter_corr_hierarchical.
    Parameters
    ----------
    prices: pd.DataFrame
    returns_data: bool
    frequency: int

    Returns
    -------
    np.ndarray
        Covariance matrix estimated through hiearchical correlation filtering
    """

    cov = sample_covariance(prices, returns_data, frequency=frequency)
    corr_filt = filter_corr_hierarchical(_cov_to_corr(cov), method="average")
    return _corr_to_cov(corr_filt, np.sqrt(np.diagonal(cov)))


all_risk_models = [
    sample_covariance,
    covariance_crr_denoise,
    covariance_crr_denoise_detoned,
    covariance_hierarchical_filter_average,
    covariance_exp,
    covariance_rmt,
    covariance_glasso,
    covariance_ledoit_wolf,
    covariance_oracle_approx,
    covariance_denoise_spectral,
    covariance_target_shrinkage_denoised,
    covariance_hierarchical_filter_complete,
    covariance_hierarchical_filter_single,
    covariance_hierarchical_filter_average,
]

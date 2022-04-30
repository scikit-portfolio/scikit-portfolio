import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from scipy.optimize import minimize


def marchenko_pastur(variance, ratio_time_samples, n_samples) -> pd.Series:
    """
    Derives the pdf of the Marcenko-Pastur distribution, given the variance of the distribution
    and the ratio between the number of observations and number of features, and number of observations.

    Parameters
    ----------
    variance: (float)
        Variance of the pdf.
    ratio_time_samples: (float)
        Relation of sample length T to the number of variables N (T/N).
    n_samples: (int)
        Number of points to estimate pdf.

    Returns
    -------
    Series of M-P pdf values.
    """
    if not isinstance(variance, float):
        variance = float(variance)

    # Minimum and maximum expected eigenvalues
    lambda_min = variance * (1 - np.sqrt(1 / ratio_time_samples)) ** 2
    lambda_max = variance * (1 + np.sqrt(1 / ratio_time_samples)) ** 2

    # Space of eigenvalues
    lambda_domain = np.linspace(lambda_min, lambda_max, n_samples)

    # Marcenko-Pastur pdf
    mp_pdf = (
        ratio_time_samples
        * np.sqrt((lambda_max - lambda_domain) * (lambda_domain - lambda_min))
        / (2 * np.pi * variance * lambda_domain)
    )
    return pd.Series(mp_pdf, index=lambda_domain)


def _pdf_fit(
    var, eigen_observations, tn_relation, kde_bwidth, num_points=1000
) -> float:
    """
    Calculates the fit (Sum of Squared estimate of Errors) of the empirical pdf
    (kernel density estimation) to the theoretical pdf (Marcenko-Pastur distribution).

    SSE is calculated for num_points, equally spread between minimum and maximum
    expected theoretical eigenvalues.

    Parameters
    ----------
    var: (float)
        Variance of the M-P distribution. (for the theoretical pdf)

    eigen_observations: (np.array) Observed empirical eigenvalues. (for the empirical pdf)
    tn_relation: (float) Relation of sample length T to the number of variables N. (for the theoretical pdf)
    kde_bwidth: (float) The bandwidth of the kernel. (for the empirical pdf)
    num_points: (int) Number of points to estimate pdf. (for the empirical pdf, 1000 by default)

    Returns
    -------
    (float) SSE between empirical pdf and theoretical pdf.
    """

    # Calculating theoretical and empirical pdf
    theoretical_pdf = marchenko_pastur(var, tn_relation, num_points)
    empirical_pdf = _fit_kde(
        eigen_observations, kde_bwidth, eval_points=theoretical_pdf.index.values
    )

    # Fit calculation
    sse = np.sum((empirical_pdf - theoretical_pdf) ** 2)

    return sse


def _find_max_eval(eigen_observations, tn_relation, kde_bwidth):
    """
    Searching for maximum random eigenvalue by fitting Marcenko-Pastur distribution
    to the empirical one - obtained through kernel density estimation. The fit is done by
    minimizing the Sum of Squared estimate of Errors between the theoretical pdf and the
    kernel fit. The minimization is done by adjusting the variation of the M-P distribution.

    Parameters
    ----------
    eigen_observations: (np.array) Observed empirical eigenvalues. (for the empirical pdf)
    tn_relation: (float) Relation of sample length T to the number of variables N. (for the theoretical pdf)
    kde_bwidth: (float) The bandwidth of the kernel. (for the empirical pdf)

    Returns
    -------
    (float, float) Maximum random eigenvalue, optimal variation of the Marcenko-Pastur distribution.
    """

    # Searching for the variation of Marcenko-Pastur distribution for the best fit with the empirical distribution
    optimization = minimize(
        _pdf_fit,
        x0=np.array(0.5),
        args=(eigen_observations, tn_relation, kde_bwidth),
        bounds=((1e-5, 1 - 1e-5),),
    )

    # The optimal solution found
    var = optimization["x"][0]

    # Eigenvalue calculated as the maximum expected eigenvalue based on the input
    maximum_eigen = var * (1 + (1 / tn_relation) ** (1 / 2)) ** 2

    return maximum_eigen, var


def _get_pca(hermit_matrix):
    """
    Calculates eigenvalues and eigenvectors from a Hermitian matrix. In our case, from the correlation matrix.

    Function used to calculate the eigenvalues and eigenvectors is linalg.eigh from numpy package.

    Eigenvalues in the output are placed on the main diagonal of a matrix.

    Parameters
    ----------
    hermit_matrix: (np.array) Hermitian matrix.

    Returns
    -------
    (np.array, np.array) Eigenvalues matrix, eigenvectors array.
    """

    # Calculating eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(hermit_matrix)

    # Index to sort eigenvalues in descending order
    indices = eigenvalues.argsort()[::-1]

    # Sorting
    eigenvalues = eigenvalues[indices]
    eigenvectors = eigenvectors[:, indices]

    # Outputting eigenvalues on the main diagonal of a matrix
    eigenvalues = np.diagflat(eigenvalues)

    return eigenvalues, eigenvectors


def _denoised_corr(eigenvalues, eigenvectors, num_facts):
    """
    De-noises the correlation matrix using the Constant Residual Eigenvalue method.

    The input is the eigenvalues and the eigenvectors of the correlation matrix and the number
    of the first eigenvalue that is below the maximum theoretical eigenvalue.

    De-noising is done by shrinking the eigenvalues associated with noise (the eigenvalues lower than
    the maximum theoretical eigenvalue are set to a constant eigenvalue, preserving the trace of the
    correlation matrix).

    The result is the de-noised correlation matrix.

    Parameters
    ----------
    eigenvalues: (np.array) Matrix with eigenvalues on the main diagonal.
    eigenvectors: (float) Eigenvectors array.
    num_facts: (float) Threshold for eigenvalues to be fixed.

    Returns
    -------
    (np.array) De-noised correlation matrix.
    """

    # Vector of eigenvalues from the main diagonal of a matrix
    eigenval_vec = np.diag(eigenvalues).copy()

    # Replacing eigenvalues after num_facts to their average value
    eigenval_vec[num_facts:] = eigenval_vec[num_facts:].sum() / float(
        eigenval_vec.shape[0] - num_facts
    )

    # Back to eigenvalues on main diagonal of a matrix
    eigenvalues = np.diag(eigenval_vec)

    # De-noised correlation matrix
    corr = np.dot(eigenvectors, eigenvalues).dot(eigenvectors.T)

    # Rescaling the correlation matrix to have 1s on the main diagonal
    corr = _cov_to_corr(corr)

    return corr


def _denoised_corr_spectral(eigenvalues, eigenvectors, num_facts) -> np.ndarray:
    """
    De-noises the correlation matrix using the Spectral method.

    The input is the eigenvalues and the eigenvectors of the correlation matrix and the number
    of the first eigenvalue that is below the maximum theoretical eigenvalue.

    De-noising is done by shrinking the eigenvalues associated with noise (the eigenvalues lower than
    the maximum theoretical eigenvalue are set to zero, preserving the trace of the
    correlation matrix).
    The result is the de-noised correlation matrix.

    Parameters
    ----------
    eigenvalues: (np.array) Matrix with eigenvalues on the main diagonal.
    eigenvectors: (float) Eigenvectors array.
    num_facts: (float) Threshold for eigenvalues to be fixed.

    Returns
    -------
    (np.array) De-noised correlation matrix.
    """

    # Vector of eigenvalues from the main diagonal of a matrix
    eigenval_vec = np.diag(eigenvalues).copy()

    # Replacing eigenvalues after num_facts to zero
    eigenval_vec[num_facts:] = 0

    # Back to eigenvalues on main diagonal of a matrix
    eigenvalues = np.diag(eigenval_vec)

    # De-noised correlation matrix
    corr = np.dot(eigenvectors, eigenvalues).dot(eigenvectors.T)

    # Rescaling the correlation matrix to have 1s on the main diagonal
    corr = _cov_to_corr(corr)

    return corr


def _denoised_corr_targ_shrink(
    eigenvalues, eigenvectors, num_facts, alpha=0
) -> np.ndarray:
    """
    De-noises the correlation matrix using the Targeted Shrinkage method.

    The input is the correlation matrix, the eigenvalues and the eigenvectors of the correlation
    matrix and the number of the first eigenvalue that is below the maximum theoretical eigenvalue
    and the shrinkage coefficient for the eigenvectors and eigenvalues associated with noise.

    Shrinks strictly the random eigenvalues - eigenvalues below the maximum theoretical eigenvalue.

    The result is the de-noised correlation matrix.

    Parameters
    ----------
    eigenvalues: (np.array) Matrix with eigenvalues on the main diagonal.
    eigenvectors: (float) Eigenvectors array.
    num_facts: (float) Threshold for eigenvalues to be fixed.
    alpha: (float) In range (0 to 1) - shrinkage among the eigenvectors.
                          and eigenvalues associated with noise. (0 by default)
    Returns
    -------
    (np.array) De-noised correlation matrix.
    """

    # Getting the eigenvalues and eigenvectors related to signal
    eigenvalues_signal = eigenvalues[:num_facts, :num_facts]
    eigenvectors_signal = eigenvectors[:, :num_facts]

    # Getting the eigenvalues and eigenvectors related to noise
    eigenvalues_noise = eigenvalues[num_facts:, num_facts:]
    eigenvectors_noise = eigenvectors[:, num_facts:]

    # Calculating the correlation matrix from eigenvalues associated with signal
    corr_signal = np.dot(eigenvectors_signal, eigenvalues_signal).dot(
        eigenvectors_signal.T
    )

    # Calculating the correlation matrix from eigenvalues associated with noise
    corr_noise = np.dot(eigenvectors_noise, eigenvalues_noise).dot(eigenvectors_noise.T)

    # Calculating the De-noised correlation matrix
    corr = corr_signal + alpha * corr_noise + (1 - alpha) * np.diag(np.diag(corr_noise))

    return corr


def _detoned_corr(corr, market_component=1) -> np.ndarray:
    """
    De-tones the correlation matrix by removing the market component.

    The input is the eigenvalues and the eigenvectors of the correlation matrix and the number
    of the first eigenvalue that is above the maximum theoretical eigenvalue and the number of
    eigenvectors related to a market component.

    Parameters
    ---------
    corr: (np.array)
        Correlation matrix to detone.
    market_component: (int)
        Number of fist eigevectors related to a market component. (1 by default)

    Returns
    -------
    (np.array) De-toned correlation matrix.
    """

    # Calculating eigenvalues and eigenvectors of the de-noised matrix
    eigenvalues, eigenvectors = _get_pca(corr)

    # Getting the eigenvalues and eigenvectors related to market component
    eigenvalues_mark = eigenvalues[:market_component, :market_component]
    eigenvectors_mark = eigenvectors[:, :market_component]

    # Calculating the market component correlation
    corr_mark = np.dot(eigenvectors_mark, eigenvalues_mark).dot(eigenvectors_mark.T)

    # Removing the market component from the de-noised correlation matrix
    corr = corr - corr_mark

    # Rescaling the correlation matrix to have 1s on the main diagonal
    corr = _cov_to_corr(corr)

    return corr


def _corr_to_cov(corr, std) -> np.ndarray:
    """
    Recovers the covariance matrix from a correlation matrix.

    Requires a vector of standard deviations of variables - square root
    of elements on the main diagonal fo the covariance matrix.

    Formula used: Cov = Corr * OuterProduct(std, std)

    Parameters
    ----------
    corr: (np.array) Correlation matrix.
    std: (np.array) Vector of standard deviations.

    Returns
    -------
    np.array
        Covariance matrix.
    """

    cov = corr * np.outer(std, std)
    return cov


def _cov_to_corr(cov) -> np.ndarray:
    """
    Derives the correlation matrix from a covariance matrix.

    Formula used: Corr = Cov / OuterProduct(std, std)

    Parameters:
    ----------
    cov: (np.array) Covariance matrix.

    Returns
    --------
    np.ndarray
        Covariance matrix.
    """

    # Calculating standard deviations of the elements
    std = np.sqrt(np.diag(cov))

    # Transforming to correlation matrix
    corr = cov / np.outer(std, std)

    # Making sure correlation coefficients are in (-1, 1) range
    corr[corr < -1], corr[corr > 1] = -1, 1

    return corr


def _is_matrix_invertible(matrix):
    """
    Check if a matrix is invertible or not.

    Parameters
    ----------
    matrix: (Numpy matrix) A matrix whose invertibility we want to check.

    Returns
    -------
    bool
        Boolean value depending on whether the matrix is invertible or not.
    """

    return (
        matrix.shape[0] == matrix.shape[1]
        and np.linalg.matrix_rank(matrix) == matrix.shape[0]
    )


def _fit_kde(observations, kde_bwidth=0.01, kde_kernel="gaussian", eval_points=None):
    """
    Fits kernel to a series of observations (in out case eigenvalues), and derives the
    probability density function of observations.

    The function used to fit kernel is KernelDensity from sklearn.neighbors. Fit of the KDE
    can be evaluated on a given set of points, passed as eval_points variable.

    Parameters
    ----------
    observations: (np.array) Array of observations (eigenvalues) eigenvalues to fit kernel to.
    kde_bwidth: (float) The bandwidth of the kernel. (0.01 by default)
    kde_kernel: (str) Kernel to use [``gaussian`` by default, ``tophat``, ``epanechnikov``, ``exponential``,
                             ``linear``,``cosine``].
    eval_points: (np.array) Array of values on which the fit of the KDE will be evaluated.
                                   If None, the unique values of observations are used. (None by default)
    Returns
    -------
    pd.Series
        Series with estimated pdf values in the eval_points.
    """

    # Reshaping array to a vertical one
    observations = observations.reshape(-1, 1)

    # Estimating Kernel Density of the empirical distribution of eigenvalues
    kde: KernelDensity = KernelDensity(kernel=kde_kernel, bandwidth=kde_bwidth).fit(
        observations
    )

    # If no specific values provided, the fit KDE will be valued on unique eigenvalues.
    if eval_points is None:
        eval_points = np.unique(observations).reshape(-1, 1)

    # If the input vector is one-dimensional, reshaping to a vertical one
    if len(eval_points.shape) == 1:
        eval_points = eval_points.reshape(-1, 1)

    # Evaluating the log density model on the given values
    log_prob = kde.score_samples(eval_points)

    # Preparing the output of pdf values
    pdf = pd.Series(np.exp(log_prob), index=eval_points.flatten())

    return pdf

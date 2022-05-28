# from ._efficientfrontier import _BaseMeanVariancePortfolio
#
# class LogMeanRisk(_BaseMeanVariancePortfolio):
#     """
#     The LogMeanRisk criterion is to maximize the expected value of the log-wealth $\mathbb{E}\lbrack \log W \rbrack$.
#     https://www.gwern.net/docs/statistics/decision/1975-thorp.pdf
#
#     We implement LogMeanRisk criterion in terms of convex optimization using the following utility function:
#
#     $$
#     f(\mathbf{w}) = \frac{1}{2} k \mathbf{w}^T \mathbf{C} \mathbf{w} - \mathbf{w}^T \boldsymbol \mu
#     $$
#
#     The user can also slightly modify the above utility function by providing a $\gamma$ parameter multiplying the
#     representing the $l_2$ loss term.
#     """
#
#     def __init__(
#         self,
#         returns_data: bool = False,
#         l2_gamma=0,
#         min_weight=0,
#         max_weight=1,
#         frequency=APPROX_BDAYS_PER_YEAR,
#         rets_estimator: BaseReturnsEstimator = MeanHistoricalLinearReturns(),
#         k: float = 1.0,
#     ):
#         super().__init__(
#             returns_data=returns_data,
#             l2_gamma=l2_gamma,
#             min_weight=min_weight,
#             max_weight=max_weight,
#             frequency=frequency,
#             rets_estimator=rets_estimator,
#         )
#         self.k = k
#
#     def fit(self, X, y=None) -> PortfolioEstimator:
#         ef = self._optimizer(X)
#
#         def kelly_objective(_w, exp_ret, cov, k):
#             variance = cp.quad_form(_w, cov)
#             objective = 0.5 * k * variance - _w @ exp_ret
#             return objective
#
#         ef.convex_objective(
#             kelly_objective,
#             exp_ret=self.rets_estimator.set_returns_data(self.returns_data)
#             .fit(X)
#             .expected_returns_,
#             cov=self.risk_estimator.set_returns_data(self.returns_data)
#             .fit(X)
#             .risk_matrix_,
#             k=self.k,
#         )
#         self.weights_ = pd.Series(
#             index=X.columns, data=ef.weights, name=self.__class__.__name__
#         )
#         return self
#
#     def grid_parameters(self) -> Dict[str, Sequence[Any]]:
#         return dict(
#             **super().grid_parameters(),
#             **{"k": np.logspace(-3, 1, 10)},
#         )
#
#
# class LogBarrier(LogMeanRisk):
#     """
#     Maximizes the LogBarrier portfolio, based on the Kelly criterion. The expected utility function is:
#     $$
#     \mathbf{w}^T \mathbf{C} \mathbf{w}  - k \sum_i \log \left (\mathbf{w}_i^+ \right)
#     $$
#     """
#
#     def fit(self, X, y=None) -> PortfolioEstimator:
#         ef = self._optimizer(X)
#
#         def logarithmic_barrier(_w, cov, k):
#             log_sum = cp.sum(cp.log(_w))
#             portfolio_volatility = cp.quad_form(_w, cov)
#             return portfolio_volatility - k * log_sum
#
#         ef.convex_objective(
#             logarithmic_barrier,
#             cov=self.risk_estimator.set_returns_data(self.returns_data)
#             .fit(X)
#             .risk_matrix_,
#             k=self.k,
#         )
#         self.weights_ = pd.Series(ef.clean_weights(), name=self.__class__.__name__)
#         return self
#
#
# class DeviationRiskParityError1(_BaseMeanVariancePortfolio):
#     """
#     Optimization of deviation risk parity
#     60 Years of Portfolio Optimization: Practical Challenges and Current Trends.
#     Petter N. Kolm, Reha Tütüncü, Frank J. Fabozzi
#     Specifically we implement the Maillard, Roncalli and Teïletche function, described as the DRP1 equation.
#     """
#
#     def fit(self, X, y=None) -> PortfolioEstimator:
#         def deviation_risk_parity(w, C):
#             diff = w * np.dot(C, w) - (w * np.dot(C, w)).reshape(-1, 1)
#             return (diff ** 2).sum().sum()
#
#         ef = self._optimizer(X)
#         cov_matrix = (
#             self.risk_estimator.set_returns_data(self.returns_data).fit(X).risk_matrix_
#         )
#         ef.nonconvex_objective(deviation_risk_parity, cov_matrix)
#         self.weights_ = pd.Series(ef.clean_weights(), name=self.__class__.__name__)
#         return self

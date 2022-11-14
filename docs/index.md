![scikit-portfolio](./imgs/scikit_portfolio.png)

[comment]: <> ([![scikit-portfolio CI]&#40;https://github.com/carrottrade/carrottrade/workflows/Carrottrade%20CI/badge.svg&#41;]&#40;https://github.com/carrottrade/carrottrade/actions/&#41;)

[comment]: <> ([[![Coverage Status]&#40;https://coveralls.io/repos/github/carrottrade/carrottrade/badge.svg?branch=develop&service=github&#41;]&#40;https://coveralls.io/github/carrottrade/carrottrade?branch=develop&#41;)

[comment]: <> ([[![Maintainability]&#40;https://api.codeclimate.com/v1/badges/5737e6d668200b7518ff/maintainability&#41;]&#40;https://codeclimate.com/github/carrottrade/carrottrade/maintainability&#41;)


[comment]: <> (<a class="github-button" href="https://github.com/carrottrade/carrottrade" data-icon="octicon-star" data-size="large" aria-label="Star carrottrade/carrottrade on GitHub">Star</a>)

[comment]: <> (<a class="github-button" href="https://github.com/carrottrade/carrottrade/fork" data-icon="octicon-repo-forked" data-size="large" aria-label="Fork carrottrade/carrottrade on GitHub">Fork</a>)

[comment]: <> (<a class="github-button" href="https://github.com/carrottrade/carrottrade/archive/stable.zip" data-icon="octicon-cloud-download" data-size="large" aria-label="Download carrottrade/carrottrade on GitHub">Download</a>)

# Welcome to scikit-portfolio

Scikit-portfolio is a Python package designed to introduce **data scientists and machine learning engineers** to the 
problem of **optimal portfolio allocation in finance**.
The main idea of scikit-portfolio is to provide many well-known portfolio optimization methods with an easily
accessible **scikit-learn inspired** set of API. 

This approach makes it possible to incorporate portfolio estimators **as if they are classical scikit-learn 
estimators**, thus enabling cross-validation and hyperparameters optimization with the tools of the Python data-science toolkit, and with an eye the highly technical domain of investment portfolio management.

This library is based upon the following open-source tools:

- [PyPortfolioOpt](https://pyportfolioopt.readthedocs.io/): a library for portfolio optimization and risk management.
- [cvxpy](https://cvxpy.org): a tool for convex optimization that makes it very easy to write optimization problem, with complex objective functions and constraints.
- [numpy](https://numpy.org): numerical analysis and linear algebra in Python.
- [pandas](https://pandas.pydata.org): manipulation of structured data tables in Python.
 
## Getting started
If you already have some background of portfolio optimization, and just want to jump start into the usage of the 
library, [read here💻️✍️](getting_started).

Otherwise if you are relatively new to financial portfolio optimization, [read this introduction 👨‍🎓️](portfolio_optimization_theory.md), where we provide the reader a coincise overview over the main ideas and methods in portfolio optimization.

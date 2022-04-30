# Installation

## Via **pip** python package manager...
The simplest way to install the library is through the **pip** python package manager, all the dependencies are automatically installed for you.

```bash
pip install scikit-portfolio
```

## ...or from source via the `scikit-portfolio` repository
Scikit-portfolio is an open source portfolio management library, whose code is hosted on [github.com/carlonicolini/scikit-portfolio](https://github.com/carlonicolini/scikit-portfolio).
We reccomend you install on a virtual envinronment. If you are on Mac/Linux, the make sure to have virtualenv installed.

1. Download the latest version of scikit-portfolio from public repository:
```shell
git clone https://github.com/carlonicolini/scikit-portfolio.git
```

2. Enter downloaded directory
```shell
cd scikitportfolio
```
3. Create and activate a virtual envinronment named `venv`:
```shell
virtualenv venv
source venv/bin/activate
```

4. Either select the stable version:
```shell
git checkout master
```

5. or, if you are a developer, switch to the `development` branch, where you can send *pull requests*: 
```shell
git checkout develop
```

6. Then install the library with `pip`.
```shell
pip install .
```

7. To run all tests, you need to have **pytest*** installed. For this you need the `requirements_dev` packages:
```shell
pip install -r requirements_dev.txt
```
then run `tox`
```shell
make test
```
from skportfolio.datasets import load_dataset, get_dataset_names

datasets = [
    "cripto_large",
    "nasdaq_100",
    "random_returns_matlab",
    "sp500",
    "tech_stocks",
]


def test_load():
    for f in get_dataset_names():
        load_dataset(name=f)

from skportfolio.datasets import load_dataset, get_dataset_names


def test_load():
    for f in get_dataset_names():
        load_dataset(name=f)

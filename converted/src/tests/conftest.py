from pathlib import Path

import pytest


FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent.parent / "examples" / "nntrees"


def load_json(name: str) -> dict:
    path = FIXTURES_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Fixture not found: {path}")
    import json

    with open(path) as f:
        return json.load(f)


# -- session-scoped JSON fixtures ---------------------------------------------------

@pytest.fixture(scope="session")
def transformer_classifier_json():
    return load_json("transformer_classifier")


@pytest.fixture(scope="session")
def auto_encoder_json():
    return load_json("auto_encoder")


@pytest.fixture(scope="session")
def auto_encoder_nested_json():
    return load_json("auto_encoder_nested_submodels")


@pytest.fixture(scope="session")
def skip_connections_json():
    return load_json("skip_connections_with_repetition")


@pytest.fixture(scope="session")
def mninst_skip_json():
    return load_json("mninst_skip")

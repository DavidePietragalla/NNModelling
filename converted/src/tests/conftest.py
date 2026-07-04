# NNModelling — DSL for designing neural networks via visual node editor
# Copyright (C) 2026  Luca Sforza
#
# Licensed under the GNU General Public License v3 or later.
# Commercial licenses are available — contact Luca Sforza.
# See the LICENSE file for details.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
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

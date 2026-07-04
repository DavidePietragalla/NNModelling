# NNModelling — DSL for designing neural networks via visual node editor
# Copyright (C) 2026  Luca Sforza
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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

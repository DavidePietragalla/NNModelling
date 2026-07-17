"""Discover dataset classes installed in the backend Python environment."""

from __future__ import annotations

import inspect
import importlib
import pkgutil
from typing import Any, get_type_hints

from dataset.ds import Dataset

from backend.models import DatasetInfo, DatasetParameter


def _type_name(annotation: Any) -> str:
    """Return a stable display name for a constructor annotation."""

    if annotation is inspect.Parameter.empty:
        return "string"
    if annotation in (str, int, float, bool):
        return annotation.__name__
    return str(annotation).replace("typing.", "")


def discover_datasets() -> list[DatasetInfo]:
    """Find concrete ``Dataset`` subclasses in the trusted dataset package."""

    import dataset

    result: list[DatasetInfo] = []
    module_names = [dataset.__name__]
    if hasattr(dataset, "__path__"):
        module_names.extend(
            f"{dataset.__name__}.{module.name}"
            for module in pkgutil.iter_modules(dataset.__path__)
        )

    seen: set[str] = set()
    for module_name in module_names:
        module = importlib.import_module(module_name)
        for class_name, candidate in inspect.getmembers(module, inspect.isclass):
            if candidate is Dataset or not issubclass(candidate, Dataset):
                continue
            target = f"{candidate.__module__}.{class_name}"
            if target in seen:
                continue
            seen.add(target)
            try:
                signature = inspect.signature(candidate.__init__)
                hints = get_type_hints(candidate.__init__)
            except (TypeError, ValueError):
                signature = None
                hints = {}
            parameters: list[DatasetParameter] = []
            if signature:
                for name, parameter in signature.parameters.items():
                    if name == "self" or parameter.kind in (
                        inspect.Parameter.VAR_POSITIONAL,
                        inspect.Parameter.VAR_KEYWORD,
                    ):
                        continue
                    default = None if parameter.default is inspect.Parameter.empty else parameter.default
                    try:
                        # The API must remain JSON serializable.
                        import json

                        json.dumps(default)
                    except (TypeError, ValueError):
                        default = str(default)
                    parameters.append(
                        DatasetParameter(
                            name=name,
                            type=_type_name(hints.get(name, parameter.annotation)),
                            default=default,
                            required=parameter.default is inspect.Parameter.empty,
                        )
                    )
            result.append(
                DatasetInfo(
                    target=target,
                    name=class_name,
                    doc=inspect.getdoc(candidate) or "",
                    parameters=parameters,
                )
            )
    return sorted(result, key=lambda item: item.target)


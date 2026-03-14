from __future__ import annotations

from importlib import import_module


_INVERSE_MODEL_SPECS: dict[str, tuple[str, str]] = {
    "VishalNet": ("model.CNN2Layer", "VishalNet"),
    "tcnc": ("model.tcn", "TCN_IV_1D_C"),
    "GRU_MM": ("model.M2M_LSTM", "GRU_MM"),
    "Unet_1D": ("model.Unet_1D", "Unet_1D"),
    "Transformer": ("model.Transformer", "TransformerModel"),
}


def available_inverse_models() -> list[str]:
    available: list[str] = []
    for model_name, (module_name, class_name) in _INVERSE_MODEL_SPECS.items():
        try:
            module = import_module(module_name)
            getattr(module, class_name)
            available.append(model_name)
        except Exception:
            continue
    return available


def resolve_inverse_model_class(model_name: str):
    if model_name not in _INVERSE_MODEL_SPECS:
        known = ", ".join(sorted(_INVERSE_MODEL_SPECS.keys()))
        raise ValueError(f"Unknown model_name: {model_name}. Known models: {known}")

    module_name, class_name = _INVERSE_MODEL_SPECS[model_name]
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        available = ", ".join(available_inverse_models()) or "<none>"
        if exc.name == module_name:
            msg = (
                f"Configured model_name='{model_name}' requires missing module '{module_name}'. "
                f"Currently available inverse models: {available}"
            )
        else:
            msg = (
                f"Configured model_name='{model_name}' could not be imported because dependency "
                f"'{exc.name}' is missing while loading '{module_name}'. "
                f"Currently available inverse models: {available}"
            )
        raise ImportError(msg) from exc

    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        available = ", ".join(available_inverse_models()) or "<none>"
        raise ImportError(
            f"Configured model_name='{model_name}' expects class '{class_name}' in '{module_name}', "
            f"but it was not found. Currently available inverse models: {available}"
        ) from exc

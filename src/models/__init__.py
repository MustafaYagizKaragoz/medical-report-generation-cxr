from .cnn_lstm import ImageCaptioningModel

# Transformer modelini lazy import et:
# Kaggle'da torch/_transformers versiyon uyumsuzluğu olduğunda
# sadece Transformer devre dışı kalır, CNN-LSTM etkilenmez.
try:
    from .transformer_model import MedicalTransformer
except Exception as _e:  # noqa: BLE001
    import warnings
    warnings.warn(
        f"MedicalTransformer yüklenemedi (torch/transformers uyumsuzluğu?): {_e}",
        ImportWarning,
        stacklevel=2,
    )
    MedicalTransformer = None  # type: ignore[assignment,misc]

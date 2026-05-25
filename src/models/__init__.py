from .cnn_lstm import ImageCaptioningModel

# Transformer modelini lazy import et:
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

# ViT + DistilGPT-2 MTL modeli
try:
    from .vit_distil2gpt import ViTDistilGPT2ForMTL
except Exception as _e:
    import warnings
    warnings.warn(f"ViTDistilGPT2ForMTL yüklenemedi: {_e}", ImportWarning, stacklevel=2)
    ViTDistilGPT2ForMTL = None  # type: ignore[assignment,misc]

# Swin-B + DistilGPT-2 MTL modeli
try:
    from .swin_distilgpt2 import SwinDistilGPT2ForMTL
except Exception as _e:
    import warnings
    warnings.warn(f"SwinDistilGPT2ForMTL yüklenemedi: {_e}", ImportWarning, stacklevel=2)
    SwinDistilGPT2ForMTL = None  # type: ignore[assignment,misc]

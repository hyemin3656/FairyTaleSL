from .base import Prediction, SignRecognizer

try:
    from .tsn_recognizer import TSNRecognizer
    __all__ = ["Prediction", "SignRecognizer", "TSNRecognizer"]
except Exception as e:
    import warnings
    warnings.warn(f"TSNRecognizer 로드 실패 (mmaction/mmcv 미설치): {e}")
    TSNRecognizer = None  # type: ignore
    __all__ = ["Prediction", "SignRecognizer", "TSNRecognizer"]

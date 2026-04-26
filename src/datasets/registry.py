import threading

from src.datasets.base import DatasetAdapter

_REGISTRY: dict[str, type[DatasetAdapter]] = {}
_built_ins_lock = threading.Lock()
_built_ins_loaded = False


def register(name: str):
    def decorator(cls: type[DatasetAdapter]) -> type[DatasetAdapter]:
        _REGISTRY[name] = cls
        return cls
    return decorator


def _load_built_ins() -> None:
    import src.datasets.generic      # noqa: F401
    import src.datasets.plantvillage  # noqa: F401


def get_adapter(name: str) -> DatasetAdapter:
    global _built_ins_loaded
    if not _built_ins_loaded:
        with _built_ins_lock:
            if not _built_ins_loaded:
                _load_built_ins()
                _built_ins_loaded = True
    from src.datasets.generic import GenericAdapter
    cls = _REGISTRY.get(name, GenericAdapter)
    return cls()


def list_adapters() -> list[str]:
    global _built_ins_loaded
    if not _built_ins_loaded:
        with _built_ins_lock:
            if not _built_ins_loaded:
                _load_built_ins()
                _built_ins_loaded = True
    return sorted(_REGISTRY.keys())

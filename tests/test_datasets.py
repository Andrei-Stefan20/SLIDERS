"""Tests for dataset adapters and registry."""

from src.datasets.base import DatasetAdapter
from src.datasets.registry import get_adapter, list_adapters, register, _REGISTRY


class TestRegistry:
    def test_get_adapter_returns_instance(self):
        adapter = get_adapter("generic")
        assert isinstance(adapter, DatasetAdapter)

    def test_get_adapter_unknown_falls_back_to_generic(self):
        from src.datasets.generic import GenericAdapter
        adapter = get_adapter("nonexistent_adapter_xyz")
        assert isinstance(adapter, GenericAdapter)

    def test_list_adapters_contains_builtins(self):
        adapters = list_adapters()
        assert "generic" in adapters
        assert "plantvillage" in adapters

    def test_register_decorator(self):
        @register("_test_adapter_tmp")
        class _TmpAdapter(DatasetAdapter):
            def parse_path(self, path: str) -> tuple[str, str, bool]:
                return "a", "b", False

        assert "_test_adapter_tmp" in _REGISTRY
        adapter = get_adapter("_test_adapter_tmp")
        assert isinstance(adapter, _TmpAdapter)
        del _REGISTRY["_test_adapter_tmp"]

    def test_thread_safety(self):
        import threading
        errors = []
        def _get():
            try:
                get_adapter("generic")
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=_get) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors


class TestGenericAdapter:
    def setup_method(self):
        self.adapter = get_adapter("generic")

    def test_parse_path_returns_folder(self):
        cat, sub, is_ref = self.adapter.parse_path("/data/images/roses/img001.jpg")
        assert cat == "roses"
        assert sub == "roses"
        assert is_ref is False

    def test_direction_mode(self):
        assert self.adapter.direction_mode == "global"


class TestPlantVillageAdapter:
    def setup_method(self):
        self.adapter = get_adapter("plantvillage")

    def test_parse_healthy(self):
        path = "/data/Apple___healthy/img001.jpg"
        cat, sub, is_ref = self.adapter.parse_path(path)
        assert cat == "Apple"
        assert "healthy" in sub.lower()
        assert is_ref is True

    def test_parse_diseased(self):
        path = "/data/Tomato___Early_blight/img001.jpg"
        cat, sub, is_ref = self.adapter.parse_path(path)
        assert cat == "Tomato"
        assert is_ref is False

    def test_parse_no_separator(self):
        path = "/data/unknown_class/img.jpg"
        cat, sub, is_ref = self.adapter.parse_path(path)
        assert cat == "unknown_class"
        assert sub == "unknown_class"

    def test_direction_mode(self):
        assert self.adapter.direction_mode == "reference"



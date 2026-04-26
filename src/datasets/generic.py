from pathlib import PurePath

from src.datasets.base import DatasetAdapter
from src.datasets.registry import register


@register("generic")
class GenericAdapter(DatasetAdapter):

    def parse_path(self, path: str) -> tuple[str, str, bool]:
        folder = PurePath(path).parent.name
        return folder, folder, False

    @property
    def direction_mode(self) -> str:
        return "global"

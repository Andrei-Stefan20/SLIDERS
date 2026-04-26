from pathlib import PurePath

from src.datasets.base import DatasetAdapter
from src.datasets.registry import register


@register("plantvillage")
class PlantVillageAdapter(DatasetAdapter):

    def parse_path(self, path: str) -> tuple[str, str, bool]:
        folder = PurePath(path).parent.name
        if "___" in folder:
            plant, condition = folder.split("___", 1)
        else:
            plant, condition = folder, folder
        is_healthy = "healthy" in condition.lower()
        return plant.strip(), condition.strip(), is_healthy

    @property
    def direction_mode(self) -> str:
        return "reference"

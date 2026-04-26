from abc import ABC, abstractmethod


class DatasetAdapter(ABC):

    @abstractmethod
    def parse_path(self, path: str) -> tuple[str, str, bool]:
        """Return (category, subcategory, is_reference) for an image path."""

    @property
    def direction_mode(self) -> str:
        return "global"

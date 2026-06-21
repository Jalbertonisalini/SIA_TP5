import numpy as np
from src.data.emojis import EMOJIS

class EmojiLoader:
    def __init__(self):
        self._emojis = EMOJIS
        self._labels = list(self._emojis.keys())
        self._dataset = np.array([self._emojis[label].flatten() for label in self._labels])

    def get_all_data(self) -> tuple[np.ndarray, list[str]]:
        return self._dataset, self._labels

    def get_emoji(self, label: str) -> np.ndarray:
        return self._emojis[label]

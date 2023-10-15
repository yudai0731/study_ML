from abc import ABC, abstractclassmethod
import numpy as np


class AbstractModel(ABC):
    def __init__(self):
        self.is_trained = False

    @abstractclassmethod
    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """モデルの学習を行う

        Args:
            x (np.ndarray): input data
            y (np.ndarray): output data
        """
        pass

    @abstractclassmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """学習したモデルによる推論を行う

        Args:
            x (np.ndarray): input data

        Returns:
            np.ndarray: prediction
        """
        pass

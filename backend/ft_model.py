import os

from typing import Optional

from backend.utils import common

from tensorflow.keras import models

class NNClassifier:
    MODEL_NAME: str = 'fake_news_keras'
    MAX_FEATURES: int = 10000    # number of words to consider as features
    MAXLEN: int = 200            # cut off the text after this number of words
                            # (among the MAX_FEATURES most common words)
    EMB_DIM: int = 100           # embedding dimension

    def __init__(self, text: str) -> None:
        self.raw_text = text
        self._model: Optional[models.Model] = None
        common.logger.info(f'NNClassifier init with parameters: text = {self.raw_text}')

    @property
    def _model_path(self) -> str:
        return os.path.join(common.PROJECT_FOLDER, f'model_data/{self.MODEL_NAME}.h5')

    def _preprocess(self) -> None:
        raise NotImplemented

    def load_model(self) -> None:
        self._model = models.load_model(self._model_path)
        common.logger.info(f'Model loaded successfully from {self._model_path}')

    @property
    def model(self) -> models.Model:
        if self._model is None:
            self.load_model()
        return self._model
import os
import pickle
from io import BytesIO

import numpy as np

import contextlib
import h5py

from typing import Optional

from backend.utils import common, s3_settings

from tensorflow.keras import models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence

class NNClassifier:
    MODEL_NAME: str = 'fake_news_keras'
    MAXLEN: int = 200            # cut off the text after this number of words
                                 # (among the most common words)

    def __init__(self, text: str) -> None:
        self.raw_text = text
        self._model: Optional[models.Model] = None
        self._tokenizer: Optional[Tokenizer] = None
        self.s3_resource = s3_settings.session.resource('s3')
        common.logger.info(f'NNClassifier initialized')

    @property
    def _model_path(self) -> str:
        return os.path.join(common.PROJECT_FOLDER, f'data/{self.MODEL_NAME}.h5')

    def _preprocess(self) -> np.ndarray:
        if self._tokenizer is None:
            self.load_model_from_s3()
        pred_sequence: np.ndarray = np.array(self._tokenizer.texts_to_sequences(np.array([self.raw_text])))
        pred_sequence = sequence.pad_sequences(pred_sequence, maxlen=self.MAXLEN)
        return pred_sequence

    # def load_model(self) -> None:
    #     if self._model is None:
    #         self._model = models.load_model(self._model_path)
    #         common.logger.info(f'Model loaded successfully from {self._model_path}')
    #
    #     with open('./data/tokenizer.pickle', 'rb') as f:
    #         self._tokenizer = pickle.load(f)
    #     common.logger.info(f'Tokenizer loaded successfully')

    def load_model_from_s3(self) -> None:
        model_file: str = f'{self.MODEL_NAME}.h5'
        tokenizer_file: str = 'tokenizer.pickle'
        model_obj = self.s3_resource.Object(s3_settings.S3_BUCKET, model_file).get()['Body'].read()
        tokenizer_obj = self.s3_resource.Object(s3_settings.S3_BUCKET, tokenizer_file).get()['Body'].read()

        def helper_settings(body):
            file_access_property_list = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
            file_access_property_list.set_fapl_core(backing_store=False)
            file_access_property_list.set_file_image(body)

            file_id_args = {
                'fapl': file_access_property_list,
                'flags': h5py.h5f.ACC_RDONLY,
                'name': b'this should never matter',
            }
            return file_id_args

        h5_file_args = {
            'backing_store': False,
            'driver': 'core',
            'mode': 'r',
        }

        file_id_args_model = helper_settings(model_obj)

        if self._model is None:
            with contextlib.closing(h5py.h5f.open(**file_id_args_model)) as file_id:
                with h5py.File(file_id, **h5_file_args) as h5_file:
                    self._model = models.load_model(h5_file)
            # with BytesIO() as f:
            #     self.s3_resource.Bucket(s3_settings.S3_BUCKET).download_fileobj(model_file, f)
            #     f.seek(0)
            #     self._model = f.read()
            common.logger.info(f'Model loaded successfully')

        self._tokenizer = pickle.loads(tokenizer_obj)
        # with BytesIO() as f:
        #     self.s3_resource.Bucket(s3_settings.S3_BUCKET).download_fileobj(tokenizer_file, f)
        #     f.seek(0)
        #     self._tokenizer = pickle.load(f)
        common.logger.info(f'Tokenizer loaded successfully')

    @property
    def model(self) -> models.Model:
        '''
        This is a method for future development
        '''
        if self._model is None:
            self.load_model_from_s3()
        return self._model

    def predict(self) -> float:
        seq = self._preprocess()
        pred = self._model.predict(seq)
        return round(float(pred[0][0]), 3)

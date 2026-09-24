# model_loader.py
import joblib
import numpy as np
import pandas as pd
import logging
import os

logger = logging.getLogger(__name__)


class Predictor:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.metrics = {}
        self.metadata = {}
        self.feature_cols = []
        self.best_model_name = None
        self.is_loaded = False

    def load_model(self, path='full_pipeline.joblib'):
        if not os.path.exists(path):
            logger.error(f"❌ Файл не найден: {path}")
            return False
        try:
            artifact = joblib.load(path)
            self.model = artifact['model']
            self.preprocessor = artifact['preprocessor']
            self.metrics = artifact['metrics']
            self.metadata = artifact['metadata']
            self.feature_cols = artifact['feature_cols']
            self.best_model_name = artifact['best_model_name']
            self.is_loaded = True

            logger.info(f"✅ Модель {self.best_model_name} загружена")
            logger.info(f"   R²:   {self.metrics.get('R²', 0):.3f}")
            logger.info(f"   MAE:  {self.metrics.get('MAE', 0):,.0f} TRY")
            logger.info(f"   MAPE: {self.metrics.get('MAPE', 0):.2f}%")
            return True
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки: {e}")
            self.is_loaded = False
            return False

    def predict(self, input_data: dict) -> dict:
        """Делает предсказание с интервалом ±MAE."""
        if not self.is_loaded:
            raise RuntimeError("Модель не загружена")

        # Собираем DataFrame с правильными колонками
        row = {col: input_data.get(col) for col in self.feature_cols}
        X = pd.DataFrame([row], columns=self.feature_cols)

        # Препроцессор + модель
        X_enc = self.preprocessor.transform(X)
        log_pred = self.model.predict(X_enc)[0]
        price = float(np.expm1(log_pred))

        mae = self.metrics['MAE']
        return {
            'prediction': price,
            'min_price': max(0, price - mae),
            'max_price': price + mae,
            'confidence_interval': f"±{mae:,.0f} TRY",
            'mae': mae,
            'r2_score': self.metrics['R²'],
            'mape': self.metrics['MAPE'],
        }

    def get_model_info(self) -> dict:
        return {
            'status': 'loaded' if self.is_loaded else 'not_loaded',
            'model_type': self.best_model_name,
            'r2_score': self.metrics.get('R²'),
            'mae': self.metrics.get('MAE'),
            'rmse': self.metrics.get('RMSE'),
            'mape': self.metrics.get('MAPE'),
            'features_count': len(self.feature_cols),
        }


_predictor = None


def get_predictor() -> Predictor:
    global _predictor
    if _predictor is None:
        _predictor = Predictor()
        _predictor.load_model()
    return _predictor
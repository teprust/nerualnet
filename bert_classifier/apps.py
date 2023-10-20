from django.apps import AppConfig
from .prediction import BertClassificationPredictor


# Инициализация классификатора
class BertClassifierConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'bert_classifier'
    predictor = BertClassificationPredictor()
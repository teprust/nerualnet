import torch
from transformers import AutoTokenizer, BertForSequenceClassification

# Определяем класс, реализующий классификатор текста.
# При его инициализации в память будет загружаться модель. Метод predict получает на вход текст, возвращает метку
# класса / наименование класса. Конечный функционал можно реализовать на свое усмотрение.
class BertClassificationPredictor:
    def __init__(self, pretrained_model="cointegrated/rubert-tiny2-cedr-emotion-detection"): # используем уже обученную модель

        # Инициализация токенизатора и модели BERT
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.model = BertForSequenceClassification.from_pretrained(pretrained_model)

    def predict(self, text):

        # Токенизация текста (разбиение на слова)
        tokenized_text = self.tokenizer(text, return_tensors='pt')

        # Получение предсказания с помощью модели BERT (отключаем вычисление градиента => не меняем веса модели)
        with torch.no_grad():
            # Вызов модели с токенизированным текстом
            pred = self.model(input_ids=tokenized_text.input_ids,
                              attention_mask=tokenized_text.attention_mask,
                              token_type_ids=tokenized_text.token_type_ids)

            # Преобразование предсказания в вероятности классов
            prediction_labels = ['no_emotion', 'joy', 'sadness', 'surprise', 'fear', 'anger']

            # Вероятность принадлежности к разным эмоциональным классам (сумма равна 1)
            prediction_values = torch.softmax(pred.logits, -1).cpu().numpy()[0]

            # Сортировка и форматирование предсказания
            prediction_sorted = sorted([x for x in zip(prediction_labels, prediction_values)],
                                       key=lambda x: x[1], reverse=True)
            # Выделяем три наиболее вероятных эмоциональных класса
            prediction_output = "  |  ".join([f"{k}:".ljust(12, ' ') + f"{v:.5f}"
                                              for k, v in prediction_sorted[:3]])

        return prediction_output

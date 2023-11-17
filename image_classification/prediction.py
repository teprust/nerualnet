import torch
from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights


class ResNetImagePredictor:
    def __init__(self):
        # Загрузка предобученных весов ResNet-50
        self.weights = ResNet50_Weights.DEFAULT
        # Создание модели ResNet-50 с загруженными весами
        self.model = resnet50(weights=self.weights)
        # Перевод модели в режим оценки (без обучения)
        self.model.eval()
        # Получение значений для входных изображений на основе предобученных весов
        self.preprocess = self.weights.transforms()

    def predict(self, img_path):
        with torch.no_grad():
            img = read_image(img_path)

            # Применение preprocess к изображению, добавление размера пакета
            batch = self.preprocess(img).unsqueeze(0)

            # Подача изображения в модель, получение предсказаний
            prediction = self.model(batch).squeeze(0).softmax(0)

            # Форматирование топ-3 результатов
            results = [f"{self.weights.meta['categories'][i]}: " + f"{100 * v:.1f}%"
                       for v, i in zip(*torch.topk(prediction, 3))]

            # Соединение результатов в строку с разделителями
            results = " | ".join(results)

            return results

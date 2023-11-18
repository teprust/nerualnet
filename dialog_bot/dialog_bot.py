import torch
from collections import deque
from transformers import AutoTokenizer, AutoModelWithLMHead


class DialogBotRuGPTSmall:
    def __init__(self, pretrained_model='tinkoff-ai/ruDialoGPT-small'):

        # Инициализация бота с заданной моделью выше и предварительной загрузкой токенизатора и модели
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.model = AutoModelWithLMHead.from_pretrained(pretrained_model)

        # Сохраняем предыдущие запросы к боту (2 вопроса и 2 ответа)
        self.context = deque([], maxlen=4)  # 2 inputs, 2 answers

        # Символы для ответов, разбитые по языкам и специальным символам
        self.rus_alphabet = "йцукенгшщзхъфывапролджэячсмитьбюёЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭЯЧСМИТЬБЮЁ"
        self.en_alphabet = "qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM"
        self.symbols = "0123456789., !?\':;@#$&*()-=+"

    # Функция ответа на вопросы
    def predict(self, text):
        # Если поле диалога переполнено, удаляем два самых старых элемента (вопрос и ответ)
        if len(self.context) == self.context.maxlen:
            _ = self.context.popleft()
            _ = self.context.popleft()

        # Добавляем в поле с диалогом новый вопрос пользователя
        self.context.append(text)

        # Строим входной текст, разделяя его на части для первого и второго участника диалога (мы и бот)
        input_text = [f"@@ПЕРВЫЙ@@ {t}" if not i % 2 else f"@@ВТОРОЙ@@ {t}"
                      for i, t in enumerate(self.context)]
        input_text = " ".join(input_text) + " @@ВТОРОЙ@@ "

        # Токенизируем текст
        tokenized_text = self.tokenizer(input_text, return_tensors='pt')

        # Генерация ответа
        with torch.no_grad(): #(отключаем вычисление градиента => не меняем веса модели)
            generated_token_ids = self.model.generate(
                # Распаковка ключевых слов
                **tokenized_text,
                top_k=10,
                top_p=0.95,
                num_beams=3,
                num_return_sequences=1,
                do_sample=True,
                no_repeat_ngram_size=2,
                temperature=1.2,
                repetition_penalty=1.2,
                length_penalty=1.0,
                eos_token_id=50257,
                max_new_tokens=40
            )
            # Переводим результат в текст
            context_with_response = [self.tokenizer.decode(sample_token_ids) for sample_token_ids in
                                     generated_token_ids]
            # Извлекаем ответ из сгенерированного текста
            answer = context_with_response[0].split("@@ВТОРОЙ@@")[-1].replace("@@ПЕРВЫЙ@@", "")
            filtered_answer = [letter for letter in answer if letter in self.rus_alphabet
                               or letter in self.en_alphabet or letter in self.symbols]

            # Фильтруем ответ, оставляя только символы трех наборов алфавитов выше
            filtered_answer = "".join(filtered_answer).strip()

            # Добавляем отфильтрованный ответ в поле диалога
            self.context.append(filtered_answer)

        return answer

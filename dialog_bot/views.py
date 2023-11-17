from django.http import HttpResponseRedirect
from bert_classifier.models import Task
from django.shortcuts import get_object_or_404, render
from .apps import DialogBotConfig
from django.urls import reverse


def detail(request, task_id):
    task = get_object_or_404(Task, pk=task_id)
    latest_predictions = task.interactionml_set.order_by('-id')[:3][::-1]
    context = {
        'task': task,
        'latest_predictions': latest_predictions
    }
    return render(request, 'dialog_bot/detail.html', context)


def predict(request, task_id):

    # Ищем задачу
    task = get_object_or_404(Task, pk=task_id)

    # Получение текста ввода из POST-запроса
    input_text = request.POST['text']

    # Генерация ответа от бота
    output = DialogBotConfig.dialog_bot.predict(input_text)

    # Создание новой записи с входными данными и сгенерированным выводом
    task.interactionml_set.create(input_data=input_text, output_data=output)

    return HttpResponseRedirect(reverse('dialog_bot:detail', args=(task.id,)))

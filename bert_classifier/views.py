from django.http import HttpResponseRedirect
from .models import Task
from django.shortcuts import get_object_or_404, render
from .apps import BertClassifierConfig
from django.urls import reverse


# Необходимо импортировать модель Task из файла с описанием всех моделей. Для загрузки, наполнения
# и отображения шаблона HTML документа будем использовать метод render.
def index(request):
    tasks_list = Task.objects.all()
    context = {'tasks_list': tasks_list}
    return render(request, 'bert_classifier/index.html', context)

# По идентификатору задачи (task_id) получаем содержимое, либо ошибку Http404, если объекта не существует.
# Метод get_object_or_404 позволяет проделать это сокращённым кодом.
def detail(request, task_id):
    task = get_object_or_404(Task, pk=task_id)
    latest_predictions = task.interactionml_set.order_by('-id')[:6][::-1]
    context = {
        'task': task,
        'latest_predictions': latest_predictions
    }
    return render(request, 'bert_classifier/detail.html', context)

# По task_id получаем содержимое запроса, отправляем в классификатор, отображаем его ответ, предварительно
# создав взаимодействие (InteractionML). После исполнения остаемся на той же странице, но с обновленной информацией.
def predict(request, task_id):
    task = get_object_or_404(Task, pk=task_id)

    input_text = request.POST['text']
    output = BertClassifierConfig.predictor.predict(input_text)

    task.interactionml_set.create(input_data=input_text, output_data=output)
    return HttpResponseRedirect(reverse('bert_classifier:detail', args=(task.id,)))

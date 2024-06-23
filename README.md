# bert-mix-up
Augmenting data with mixup for text classification

## Аугментация данных 

Аугментация - это построение данных из исходных, с целью увеличения выборки и снижения переобучения. 

В качестве encoder была взята модель https://huggingface.co/google-bert/bert-base-cased , датасет - https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes


Mix up эмбеддингов на уровне токена(слева) и на уровне предложения(справа)


![mix_up_image](https://github.com/BratsunKD/bert-mix-up/assets/147521482/51f6f98d-8684-4f3c-8e73-4ab0c71ed130)

Смешивание на уровне эмбеддингов токенов: 



![word_mix_up](https://github.com/BratsunKD/bert-mix-up/assets/147521482/c9d83374-49c0-4398-a85a-2342c5ec053d)

На уровне эмбеддингов текстов:


![sentence_mix_up](https://github.com/BratsunKD/bert-mix-up/assets/147521482/73c4a35a-28cb-42ef-9fbf-ce236d8119ce)
Лямбда берется из распределения Beta(α, α), где α - гиперпараметр. 

Обучение проводилось следующим образом: 
1) Из каждого батча размером n случайным образом берется m = (1 - p)*n примеров (p - гиперпараметр).
2) Формировуются m пар, которые дальше проходили через mix-up слой.
3) На выходе обучается с помощью смешанной кросс-энтропии.
Остальные p*n примеров из батча учавствуют в стандартном обучении. 

Таким образом, параметр p позволяет регулировать долю замиксованных примеров.

## Запуск решения 

Cкачать зависимости:
```
pip install -r requirements.txt
```
Все настройки обучения и моделей в /configs

Запуска обучения модели на одной из 3-х задач: none, embedding, sentences.
```
bash run_train.sh none
bash run_train.sh embedding
bash run_train.sh sentences
```
Запуск предсказания для тестовой выборки: 
```
bash run_predict.sh none
bash run_predict.sh embedding
bash run_predict.sh sentences
```
## Результаты 

Результаты на тестовой отложенной выборке.

| mix-up type | f1-score |
| ------------| ---------|
|  none       |   0.852  |
|  embedding  |   0.858  |
|  sentences  |   0.837  |










# Anti-spoofing

## Установка

Для установки для начала потребуется  установить библиотеки и 
скачать модель
```shell
pip install -r ./requirements.txt
axel -n 8 https://disk.yandex.ru/d/mDBnOnSdipU61A
```



## Модель

Реализация основана на статье RawNet2 с использованием hydra в качестве конфига.



## Проверка на данных

Чтобы запустить тренировку модели на части LibriSpeech нужно изменить config, а именно часть, отвечающую за датасеты.
Выглядит она так
```yaml
defaults:
  - arch: RawNet
  - data: tts_dataset
  - loss: tts_loss
  - scheduler: LRStep
  - collate: main_collate
  - metrics: base_metrics
  - preprocessing: base_preprocessing
  - trainer: base_trainer
  - optimizer: Adam

name: base_config
n_gpu: 1
checkpoint:
```
Каждый отдельный файл отвечает за свой небольшой блок, из-за чего его удобно править.

Также можно протестировать модель на произвольном датасете, для этого нужно поменять конфиг данных для тренировки6 устроен он следующим образом:
```yaml
defaults:
  - arch: RawNet
  # - augmentations: base_augmentations

name: test_config
n_gpu: 1
checkpoint: ~/rawnet.pth
```
Соответсвенно, нужно поменять расположение модели и при этом засунуть все нужные данные для проверки в data/test

## Credits

This repository is based on a heavily modified fork
of [pytorch-template](https://github.com/victoresque/pytorch-template) repository.


## Проектная работа по дисциплине "Архитектура систем ИИ"
#### Автор: Строкова Анастасия (P4140)
### Решение задачи super-resolution с помощью нейронной сети

### Часть 1. Определение границ проекта

#### Цель -  создание инструмента для апробации и сравнения архитектур  нейронных сетей, нацеленных на получение изображений сверхвысокого  разрешения (Super – Resolution) фотографий, сделанных на  непрофессиональную камеру.
#### Задачи:
1. Проанализировать найденный датасет.
2. Спроектировать архитектуру системы искусственного интеллекта.
3. Подготовить данные для обучения нейронных сетей EDSR (Enhanced Deep Residual Networks for Single Image Super-Resolution) и SRGAN (Single Image Super-Resolution using a Generative Adversarial Network).
4. Обучить модель на основе обучающей выборки и провести оценку полученных результатов.
5. Подготовить тестовый набор данных и провести оценку полученных результатов на данном наборе.
6. Выбрать оптимальную модель нейронной сети для решения задачи Super-Resolution, исходя из качества и скорости работы алгоритмов.
7. Выполнить развертывание наилучшей модели.

#### Характеристика датасета
Датасет содержит 1000 изображений в каждой категории (800 изображений в обучающей выборке и по 100 изображений в валидационной и тестовой):
* Изображения High-Resolution
* Уменьшенные изображения интерполяцией, коэффициент x2
* Уменьшенные изображения интерполяцией, коэффициент x3
* Уменьшенные изображения интерполяцией, коэффициент x4

<img src="https://github.com/kivirciks/super-resolution/blob/main/pictures/Photoes.png" width="400">
Рисунок 1. Распределение первых 10-ти фотографий относительно параметров Entropy и Complexity

<img src="https://github.com/kivirciks/super-resolution/blob/main/pictures/%D0%A1%D0%BB%D0%BE%D0%B6%D0%BD%D0%BE%D1%81%D1%82%D1%8C.png" width="400">
Рисунок 2. Увеличение Complexity с уменьшением изображения (черный - x4, зеленый - x3, красный - x2, синий - HR)

<img src="https://github.com/kivirciks/super-resolution/blob/main/pictures/%D0%AD%D0%BD%D1%82%D1%80%D0%BE%D0%BF%D0%B8%D1%8F.png" width="400">
Рисунок 3. Уменьшение Entropy с уменьшением изображения (черный - x4, зеленый - x3, красный - x2, синий - HR)

#### Демонстрация целесообразности использования выбранного датасета
* Датасет имеет достаточный объем для обучения модели, решающий задачу Super-Resolution
* Датасет содержит изображений разных категорий (природа, животные, улицы, люди и др.)
* Интерполяция позволяет постепенно уменьшить изображения, тем самым дав алгоритму отследить расположение пикселей, о чем готовит точечная диаграмма на рисунке 1

#### Источники
* [Датасет](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
* [Репозиторий проекта](https://github.com/kivirciks/super-resolution)


### Часть 2. Разработка архитектуры системы

#### UML-диаграммы
##### Диаграмма активностей
<img src="https://github.com/kivirciks/super-resolution/blob/main/pictures/Workflow.png" width="1000">
Рисунок 4. Диаграмма активностей

##### Диаграмма компонентов
<img src="https://github.com/kivirciks/super-resolution/blob/main/pictures/UML_Component.png" width="400">
Рисунок 5. Диаграмма компонентов

##### Диаграмма развертывания
<img src="https://github.com/kivirciks/super-resolution/blob/main/pictures/UML_Deployment.png" width="400">
Рисунок 6. Диаграмма развертывания

### Часть 3. Подготовка исходных данных
* Реализация предусматривает обращение к API kaggle для выгрузки файлов, их распаковку и кеширование результата (https://www.kaggle.com/datasets/avstrokova/div2-k-dataset-for-super-resolution). Данный способ был добавлен для дальнейших исследований (в рамках магистерской диссертациии), для загрузки собственного датасета.
* После скачивания датасета DIV2K фотографии подвергаются первичной предобработке. Фотографии распределяются по Low-Resolution и High-Resolution директориям. Происходит аугментация фотографий (обрезка, поворот, отражение), а также перевод PNG изображения в набор векторов по модели RGB. Затем происходит усреднение канала
* preprocessing.ipynb - https://github.com/kivirciks/super-resolution/blob/main/preprocessing.ipynb

### Часть 4. Обучение моделей
#### EDSR - Enhanced Deep Residual Networks (в основе сверточная нейронная сеть CNN, 2017 год)
<img src="https://github.com/kivirciks/super-resolution/blob/main/pictures/EDSR.PNG" width="400">
Программный код нейронной сети EDSR: https://github.com/kivirciks/super-resolution/blob/main/train_edsr.py <br>
Основано на идее из статьи: https://arxiv.org/abs/1707.02921

#### RDN - Residual Dense Network (в основе улучшенная сверточная нейронная сеть CNN, 2018 год)
<img src="https://github.com/kivirciks/super-resolution/blob/main/pictures/RDB.PNG" width="600">
Программный код нейронной сети RDN: https://github.com/kivirciks/super-resolution/blob/main/train_rdn.py <br>
Основано на идее из статьи: https://arxiv.org/abs/1802.08797

#### SRGAN - Super-Resolution Using a Generative Adversarial Network (в основе генеративно-состязательная сеть GAN, 2017 год)
<img src="https://github.com/kivirciks/super-resolution/blob/main/pictures/SRGAN.png" width="600">
Программный код нейронной сети SRGAN: https://github.com/kivirciks/super-resolution/blob/main/train_srgan.py <br>
Основано на идее из статьи: [https://arxiv.org/abs/1802.08797](https://arxiv.org/abs/1609.04802)

### Часть 5. Выбор оптимальной модели
Для сравнения работы нейронных сетей будет использоваться параметр PNSR - peak signal-to-noise ratio (пиковое отношение сигнала к шуму). PSNR наиболее часто используется для измерения уровня искажений при сжатии изображений. Проще всего его определить через среднеквадратичную ошибку (СКО) или MSE (англ. mean square error). <br>
<img src="https://github.com/kivirciks/super-resolution/blob/main/pictures/PNSR.PNG" width="300">

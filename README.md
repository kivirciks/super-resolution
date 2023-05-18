## Проектная работа по дисциплине "Архитектура систем ИИ"
#### Автор: Строкова Анастасия (P4140)
### Решение задачи super-resolution с помощью нейронной сети

### Часть 1. Определение границ проекта

#### Цель -  создание инструмента для апробации и сравнения архитектур  нейронных сетей, нацеленных на получение изображений сверхвысокого  разрешения (Super – Resolution) фотографий, сделанных на  непрофессиональную камеру.
#### Задачи:
1. Проанализировать найденный датасет.
2. Спроектировать архитектуру системы искусственного интеллекта.
3. Подготовить данные для обучения нейронных сетей EDSR (Enhanced Deep Residual Networks for Single Image Super-Resolution), RDN (Residual Dense Network) и SRGAN (Single Image Super-Resolution using a Generative Adversarial Network).
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
* Реализация предусматривает обращение к API kaggle для выгрузки файлов, их распаковку и кеширование результата (https://www.kaggle.com/datasets/avstrokova/div2-k-dataset-for-super-resolution). Данный способ был добавлен для дальнейших исследований (в рамках магистерской диссертациии), для загрузки собственного датасета. В данном репозитории идет обращение напрямую к датасету DIV2K с выгрузкой необходимых папок.
* После скачивания датасета DIV2K фотографии подвергаются первичной предобработке. Фотографии распределяются по Low-Resolution и High-Resolution директориям. Происходит аугментация фотографий (обрезка, поворот, отражение), а также перевод PNG изображения в набор векторов по модели RGB. Затем происходит усреднение канала
* preprocessing.ipynb - https://github.com/kivirciks/super-resolution/blob/main/preprocessing.ipynb <br>

Функция для скачивания train High-Resolution датасета (аналогично для Low-Resolurion, а также для test High-Resolution и test Low-Resolution)
```python
# Параметры для High-Resolution части датасета
def hr_dataset(self):
    # Если уже не скачано, но скачивается архив
    if not os.path.exists(self._hr_images_dir()):
        download_archive(self._hr_images_archive(), self.images_dir, extract=True)
    # Сохраняется в кэш
    ds = self._images_dataset(self._hr_image_files()).cache(self._hr_cache_file())
    # Если не сохранен индекс изображения, то тоже сохраняется
    if not os.path.exists(self._hr_cache_index()):
        self._populate_cache(ds, self._hr_cache_file())
    return ds
```
Далее для предобработки датасета были написаны функции `random_crop` (обрезка), `random_flip` (отражение), `random_rotate` (вращение). <br>
Функции для скачивания датасета с нужными параметрами (в данной работе рассматривается только бикубическая интерполяция с коэффициентом ухудшения 4, соответственно `scale=4, downgrade='bicubic'`)
```python
# Путь, откуда скачивается датасет
def download_archive(file, target_dir, extract=True):
    source_url = f'http://data.vision.ee.ethz.ch/cvl/DIV2K/{file}'
    target_dir = os.path.abspath(target_dir)
    tf.keras.utils.get_file(file, source_url, cache_subdir=target_dir, extract=extract)
    os.remove(os.path.join(target_dir, file))

# Непосредственно получение датасета
train = DIV2K(scale=4, downgrade='bicubic', subset='train')
train_ds = train.dataset(batch_size=16, random_transform=True)
```
Немаловажно отметить, что изображения цветные, а значит при переводе картинки (RGB) в массив будет формироваться вектор из красного (R), зеленого (G) и синего (B) каналов. Для удобства усредним значения пикселей и "агрегируем" в канал Y.
```python
DIV2K_RGB_MEAN = np.array([0.4488, 0.4371, 0.4040]) * 255
```
### Часть 4. Обучение моделей
#### Сохранение весов на Яндекс Диск
Веса обученных нейронных сетей отправляются по REST API на Яндекс.Диск. <br>
Инструкция по подключению Yandex API: https://yandex.ru/dev/id/doc/ru/how-to <br>
При указании прав доступа прописываю следующие разрешения:
```
cloud_api:disk.app_folder # Доступ к папке приложения на Диске
cloud_api:disk.read # Чтение всего Диска
cloud_api:disk.write # Запись в любом месте на Диске
cloud_api:disk.info # Доступ к информации о Диске
```
В результате репозиторий Super-Resolution на GitHub был подключен к Яндекс Диску. Был сформирован ClientID и Client secret. <br>
<img src="https://github.com/kivirciks/super-resolution/blob/main/pictures/YandexDisc_API.PNG" width="300"> <br>
Пример сохранения весов для модели
```python
import yadisk
y = yadisk.YaDisk(token="y0_AgAAAAAZdSRIAAnWpQAAAADiIR-G69xDHp3vSUKGjYeHSNjcH6B_kQw")
# Сохранение весов
model_edsr.save_weights(y.upload('edsr_weights.h5', '/weights_dir/edsr_weights.h5'))
```
#### EDSR - Enhanced Deep Residual Networks (в основе сверточная нейронная сеть CNN, 2017 год)
<img src="https://github.com/kivirciks/super-resolution/blob/main/pictures/EDSR.PNG" width="400">
Программный код нейронной сети EDSR: https://github.com/kivirciks/super-resolution/blob/main/train_edsr.py <br>
Основано на идее из статьи: https://arxiv.org/abs/1707.02921
```python
# Задание модели EDSR
def edsr(scale, num_filters=64, num_res_blocks=8, res_block_scaling=None):
    x_in = Input(shape=(None, None, 3))
    x = Lambda(normalize)(x_in)

    # Задание слоев Conv
    x = b = Conv2D(num_filters, 3, padding='same')(x)
    for i in range(num_res_blocks):
        b = res_block(b, num_filters, res_block_scaling)
    b = Conv2D(num_filters, 3, padding='same')(b)
    x = Add()([x, b])
    x = upsample(x, scale, num_filters)
    x = Conv2D(3, 3, padding='same')(x)
    x = Lambda(denormalize)(x)
    return Model(x_in, x, name="edsr")
```
```python
# Задание слоев ResBlock
def res_block(x_in, filters, scaling):
    x = Conv2D(filters, 3, padding='same', activation='relu')(x_in)
    x = Conv2D(filters, 3, padding='same')(x)
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    x = Add()([x_in, x])
    return x
```
```python
# Задание слоев Upsample
def upsample(x, scale, num_filters):
    def upsample_1(x, factor, **kwargs):
        x = Conv2D(num_filters * (factor ** 2), 3, padding='same', **kwargs)(x)
        return Lambda(pixel_shuffle(scale=factor))(x)
    if scale == 2:
        x = upsample_1(x, 2, name='conv2d_1_scale_2')
    elif scale == 3:
        x = upsample_1(x, 3, name='conv2d_1_scale_3')
    elif scale == 4:
        x = upsample_1(x, 2, name='conv2d_1_scale_2')
        x = upsample_1(x, 2, name='conv2d_2_scale_2')
    return x
```
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

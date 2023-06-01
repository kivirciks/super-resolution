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

<img src="https://github.com/kivirciks/super-resolution/blob/main/pictures/Complexity.png" width="400">
Рисунок 2. Увеличение Complexity с уменьшением изображения (черный - x4, зеленый - x3, красный - x2, синий - HR)

<img src="https://github.com/kivirciks/super-resolution/blob/main/pictures/Entropy.png" width="400">
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
<img src="https://github.com/kivirciks/super-resolution/blob/main/pictures/UML_Workflow.png" width="1000">
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

#### EDSR - Enhanced Deep Residual Networks (2017 год)
<img src="https://github.com/kivirciks/super-resolution/blob/main/pictures/EDSR.PNG" width="400">
Программный код нейронной сети EDSR: https://github.com/kivirciks/super-resolution/blob/main/train_edsr.py <br>
Основано на идее из статьи: https://arxiv.org/abs/1707.02921 <br>

#### FSRCNN - Fast Super-Resolution Convolutional Neural Network (2016 год)
<img src="https://github.com/kivirciks/super-resolution/blob/main/pictures/FSRCNN.PNG" width="400">
Программный код нейронной сети FSRCNN: https://github.com/kivirciks/super-resolution/blob/main/train_edsr.py <br>
Основано на идее из статьи: https://arxiv.org/abs/1608.00367 <br>

#### SRCNN - Super-Resolution Convolutional Neural Network (2015 год)
<img src="https://github.com/kivirciks/super-resolution/blob/main/pictures/SRCNN.PNG" width="400">
Программный код нейронной сети SRCNN: https://github.com/kivirciks/super-resolution/blob/main/train_edsr.py <br>
Основано на идее из статьи: https://arxiv.org/abs/1501.00092 <br>

#### SRGAN - Super-Resolution Using a Generative Adversarial Network (в основе генеративно-состязательная сеть GAN, 2017 год)
<img src="https://github.com/kivirciks/super-resolution/blob/main/pictures/SRGAN.png" width="600">
Программный код нейронной сети SRGAN: https://github.com/kivirciks/super-resolution/blob/main/train_srgan.py <br>
Основано на идее из статьи: https://arxiv.org/abs/1802.08797

#### SubPixelCNN - Efficient Sub-Pixel Convolutional Neural Network (2016 год)
<img src="https://github.com/kivirciks/super-resolution/blob/main/pictures/SUB.PNG" width="400">
Программный код нейронной сети SubPixelCNN: https://github.com/kivirciks/super-resolution/blob/main/train_edsr.py <br>
Основано на идее из статьи: https://arxiv.org/abs/1609.05158 <br>

#### VDSR - Very Deep Convolutional Networks (2015 год)
<img src="https://github.com/kivirciks/super-resolution/blob/main/pictures/VDSR.PNG" width="400">
Программный код нейронной сети VDSR: https://github.com/kivirciks/super-resolution/blob/main/train_edsr.py <br>
Основано на идее из статьи: https://arxiv.org/abs/1511.04587 <br>

Также была предусмотрена проверка того, что сохранение и замена весов на происходило только в том случае, когда значение PNSR было больше, чем существующее.
```python
    # Сканирует папку с весами и удаляет все, кроме:
    # - max_best новейшие «лучшие» веса
    # - max_n_weights самые последние веса "других"
    def _remove_old_weights(self, max_n_weights, max_best=5):
        w_list = {}
        w_list['all'] = [w for w in self.callback_paths['weights'].iterdir() if '.pth' in w.name]
        w_list['best'] = [w for w in w_list['all'] if 'best' in w.name]
        w_list['others'] = [w for w in w_list['all'] if w not in w_list['best']]
        # удаление весов, которые хуже
        epochs_set = {}
        epochs_set['best'] = list(
            set([self.epoch_n_from_weights_name(w.name) for w in w_list['best']])
        )
        epochs_set['others'] = list(
            set([self.epoch_n_from_weights_name(w.name) for w in w_list['others']])
        )
        keep_max = {'best': max_best, 'others': max_n_weights}
        for type in ['others', 'best']:
            if len(epochs_set[type]) > keep_max[type]:
                epoch_list = np.sort(epochs_set[type])[::-1]
                epoch_list = epoch_list[0: keep_max[type]]
                for w in w_list[type]:
                    if self.epoch_n_from_weights_name(w.name) not in epoch_list:
                        w.unlink()
```
В дополнении, была предусмотрена возможность дообучения модели, веса которой уже сохранены:
```python
from keras.models import save_model, load_model
model = load_model(model_path.pth)
model.fit(new_x_train, new_y_train, ...)
```
### Часть 5. Выбор оптимальной модели
Для сравнения работы нейронных сетей будет использоваться параметр PSNR - peak signal-to-noise ratio (пиковое отношение сигнала к шуму). PSNR наиболее часто используется для измерения уровня искажений при сжатии изображений. Проще всего его определить через среднеквадратичную ошибку (СКО) или MSE (англ. mean square error). <br>
<img src="https://github.com/kivirciks/super-resolution/blob/main/pictures/PNSR.PNG" width="300">
```python
# Оценка значения PSNR: PSNR = 20 * log10(MAXp) - 10 * log10(MSE)
def PSNR(y_true, y_pred, MAXp=1):
    # y_true - реальное значение
    # y_pred - предсказываемое значение
    # MAXp - максимальное значение диапазона пикселей (по умолчанию=1).
    return -10.0 * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.0)
```
Каждое обучение длилось по 20 эпох, 200 шагов в каждом. Значение PNSR записывалось в файл metrics.txt:
```python
with open('metrics.txt', 'w') as f:
    f.write(f"dbpn PSNR: {dbpn_report}")
    f.write("\n")
    f.write(f"drcn PSNR: {drcn_report}")
    f.write("\n")
    f.write(f"edsr PSNR: {edsr_report}")
    f.write("\n")
    f.write(f"fsrcnn PSNR: {fsrcnn_report}")
    f.write("\n")
    f.write(f"srcnn PSNR: {srcnn_report}")
    f.write("\n")
    f.write(f"srgan PSNR: {srgan_report}")
    f.write("\n")
    f.write(f"sub PSNR: {sub_report}")
    f.write("\n")
    f.write(f"vdsr PSNR: {vdsr_report}")
```
Аналогично для скорости преобразования черно-белых и цветных изображений, речь о которых пойдет ниже (metrics_black_photo.txt и metrics_color_photo.txt). <br>
Для дополнительной оценки моделей была измерена скорость преобразования Super-Resolution одной цветной и одной черно-белой фотографии.
```python
import time 
start = time.time() ## точка отсчета времени
...
end = time.time() - start ## собственно время работы программы
print(end) ## вывод времени
```
Часть программного кода, отвечающая за преобразование Super-Resolution:
```python
out = model(data)
out = out.cpu()
out_img_y = out.data[0].numpy()
out_img_y *= 255.0
out_img_y = out_img_y.clip(0, 255)
out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')

out_img.save(args.output)
print('output image saved to ', args.output)
```
Однако при решении задачи Super-Resolution не столько важны скорость обработки (в нашем случае выбивается только SRGAN, однако минута на обработку также приемлема), сколько качество итогового изображения. Поэтому автором была выставлена экспертная оценка полученному изображению. В выборе модели для развертывания при подсчете итоговый балл будет умножаться на экспертный коэффициент, который может быть от 2 до 10. Для оценки полученные фотографии проранжируются по "привлекательности" для глаза. Балл по группе (черно-белые илицветные) повторяться не может.
Итоговые значения представлены ниже.
<table border="1">
   <tr>
    <th>Модель</th>
    <th>PNSR, dB</th>
    <th>Время обработки цветной фотографии</th>
    <th>Время обработки черно-белой фотографии</th>
    <th>Экспертная оценка</th> 
   </tr>
   <tr>
    <th>EDSR</th>
    <th>8.9392</th>
    <th>5.749220848083496</th>
    <th>7.394894599914551</th>
    <th>4</th>
   </tr>
   <tr>
    <th>FSRCNN</th>
    <th>23.6084</th>
    <th>0.6419787406921387</th>
    <th>0.8607726097106934</th>
    <th>2</th>
   </tr>
   <tr>
    <th>SRCNN</th>
    <th>23.0745</th>
    <th>0.5174150466918945</th>
    <th>0.5501530170440674</th>
    <th>9</th>
   </tr>
   <tr>
    <th>SRGAN</th>
    <th>20.9873</th>
    <th>42.673673152923584</th>
    <th>49.49873065948486</th>
    <th>7</th>
   </tr>
   <tr>
    <th>SubPixelCNN</th>
    <th>22.4866</th>
    <th>0.828690767288208</th>
    <th>0.6773681640625</th>
    <th>10</th>
   </tr>
   <tr>
    <th>VDSR</th>
    <th>23.5409</th>
    <th>0.9105465412139893</th>
    <th>0.7319796085357666</th>
    <th>1</th>
   </tr>
 </table>
 
 Примеры работы нейронной сети:
 <table border="1">
   <tr>
    <th>Модель</th>
    <th>Цветное изображение</th>
    <th>Черно-белое изображение</th>
   </tr>
   <tr>
    <th>EDSR</th>
    <th><img src="https://github.com/kivirciks/super-resolution/blob/main/pictures/Photo_Color_EDSR.jpg" width="200"></th>
    <th><img src="https://github.com/kivirciks/super-resolution/blob/main/pictures/Photo_Black_EDSR.jpg" width="200"></th>
   </tr>
   <tr>
    <th>FSRCNN</th>
    <th><img src="https://github.com/kivirciks/super-resolution/blob/main/pictures/Photo_Color_FSRCNN.jpg" width="200"></th>
    <th><img src="https://github.com/kivirciks/super-resolution/blob/main/pictures/Photo_Black_FSRCNN.jpg" width="200"></th>
   </tr>
   <tr>
    <th>SRCNN</th>
    <th><img src="https://github.com/kivirciks/super-resolution/blob/main/pictures/Photo_Color_SRCNN.jpg" width="200"></th>
    <th><img src="https://github.com/kivirciks/super-resolution/blob/main/pictures/Photo_Black_SRCNN.jpg" width="200"></th>
   </tr>
   <tr>
    <th>SRGAN</th>
    <th><img src="https://github.com/kivirciks/super-resolution/blob/main/pictures/Photo_Color_SRGAN.jpg" width="200"></th>
    <th><img src="https://github.com/kivirciks/super-resolution/blob/main/pictures/Photo_Black_SRGAN_Generator.jpg.jpg" width="200"></th>
   </tr>
   <tr>
    <th>SubPixelCNN</th>
    <th><img src="https://github.com/kivirciks/super-resolution/blob/main/pictures/Photo_Color_SUB.jpg" width="200"></th>
    <th><img src="https://github.com/kivirciks/super-resolution/blob/main/pictures/Photo_Black_SUB.jpg" width="200"></th>
   </tr>
   <tr>
    <th>VDSR</th>
    <th><img src="https://github.com/kivirciks/super-resolution/blob/main/pictures/Photo_Color_VDSR.jpg" width="200"></th>
    <th><img src="https://github.com/kivirciks/super-resolution/blob/main/pictures/Photo_Black_VDSR.jpg" width="200"></th>
   </tr>
 </table>
 
 **Алгоритм выбора лучшей модели:
 1. Значимость метрик PNSR, скорости обработки цветной фотографии и черно-белой составило 50%, 30% и 20% соответственно.
 2. Сортировка значений PNSR от большего к меньшему. Первому начисляется 6 баллов, последнему - 1. Балл умножался на 0,5 (вес критерия 50%).
 3. Сортировка скорости обработки цветного изображения от меньшего к большему. Первому начисляется 6 баллов, последнему - 1. Балл умножается на 0,3 (вес критерия 30%).
 4. Сортировка скорости обработки черно-белого изображения от меньшего к большему. Первому начисляется 6 баллов, последнему - 1. Балл умножается на 0,2 (вес критерия 20%).
 5. Суммируется значение 2, 3 и 4 пунктов и умножается на экспертный коэффициент.
 6. Веса модели с большим итоговым баллом берутся для развертывания.
 
### Часть 6. Развертывание
Для сравнения работы нейронных сетей будет использоваться параметр PSNR - peak signal-to-noise ratio (пиковое отношение сигнала к шуму). PSNR наиболее часто используется для измерения уровня искажений при сжатии изображений. Проще всего его определить через среднеквадратичную ошибку (СКО) или MSE (англ. mean square error). <br>
<img src="https://github.com/kivirciks/super-resolution/blob/main/pictures/PNSR.PNG" width="300">

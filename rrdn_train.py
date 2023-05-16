import logging
import os
import zipfile
import wget
from time import time
import yaml
from pathlib import Path
import numpy as np
from tqdm import tqdm
import imageio
import argparse
from datetime import datetime
import yaml

# нейросетевые библиотеки Keras
import tensorflow as tf
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.layers import concatenate, Input, Activation, Add, Conv2D, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import backend as K
import tensorflow.keras.backend as K

# Логгирование
def get_logger(name, job_dir='.'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        # обработчик потока передает события регистрации на stdout
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch_formatter = logging.Formatter('%(message)s')
        ch.setFormatter(ch_formatter)
        logger.addHandler(ch)
        
        # обработчик потока проверяет, что события записались в файл
        if not os.path.exists(job_dir):
            os.makedirs(job_dir)        
        fh = logging.FileHandler(filename=os.path.join(job_dir, 'log_file'))
        fh.setLevel(logging.DEBUG)
        fh_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(fh_formatter)
        logger.addHandler(fh)    
    return logger

# DataHandler генерирует расширенные пакеты, используемые для обучения или проверки.
# lr_dir - директория с изображениями низкого разрешения
# hr_dir - директория с изображениями высокого разрешения
# patch_size - integer, размер патчей, извлеченных из изображений низкого разрешения
# scale - integet, повышающий коэффициент
# n_validation_samples - integer, размер тестового набора
class DataHandler:
    def __init__(self, lr_dir, hr_dir, patch_size, scale, n_validation_samples=None):
        self.folders = {'hr': hr_dir, 'lr': lr_dir}  # директории хранения изображений
        self.extensions = ('.png', '.jpeg', '.jpg')  # допустимые расширения изображений
        self.img_list = {}  # list of file names
        self.n_validation_samples = n_validation_samples
        self.patch_size = patch_size
        self.scale = scale
        self.patch_size = {'lr': patch_size, 'hr': patch_size * self.scale}
        self.logger = get_logger(__name__)
        self._make_img_list()
        self._check_dataset()
    
    # Создает словарь списков допустимых изображений, содержащихся в lr_dir и hr_dir 
    def _make_img_list(self):       
        for res in ['hr', 'lr']:
            file_names = os.listdir(self.folders[res])
            file_names = [file for file in file_names if file.endswith(self.extensions)]
            self.img_list[res] = np.sort(file_names)        
        if self.n_validation_samples:
            samples = np.random.choice(
                range(len(self.img_list['hr'])), self.n_validation_samples, replace=False)
            for res in ['hr', 'lr']:
                self.img_list[res] = self.img_list[res][samples]
    
    # Проверка работоспособности для набора данных
    def _check_dataset(self):
        # порядок этих утверждений важен для тестирования
        assert len(self.img_list['hr']) == self.img_list['hr'].shape[0], 'UnevenDatasets'
        assert self._matching_datasets(), 'Input/LabelsMismatch'
    
    # Сопоставление имени файлов для LR и HR
    def _matching_datasets(self):
        # LR_name.png = HR_name+x+scale.png
        # или
        # LR_name.png = HR_name.png
        LR_name_root = [x.split('.')[0].rsplit('x', 1)[0] for x in self.img_list['lr']]
        HR_name_root = [x.split('.')[0] for x in self.img_list['hr']]
        return np.all(HR_name_root == LR_name_root)
    
    # Определяет, является ли патч сложным или недостаточно ровным
    def _not_flat(self, patch, flatness):        
        if max(np.std(patch, axis=0).mean(), np.std(patch, axis=1).mean()) < flatness:
            return False
        else:
            return True     
    
    # Обрезка изображений. Возвращает пакет для каждого изображения в наборе проверки
    def _crop_imgs(self, imgs, batch_size, flatness):  
        slices = {}
        crops = {}
        crops['lr'] = []
        crops['hr'] = []
        accepted_slices = {}
        accepted_slices['lr'] = []
        # Получение случайных координаты верхних левых углов в пространстве LR
        top_left = {'x': {}, 'y': {}}
        # Умножение на масштаб, чтобы получить кадровые координаты
        n = 50 * batch_size
        
        # Получает batch_size + n возможных координат
        for i, axis in enumerate(['x', 'y']):
            top_left[axis]['lr'] = np.random.randint(
                0, imgs['lr'].shape[i] - self.patch_size['lr'] + 1, batch_size + n)
            top_left[axis]['hr'] = top_left[axis]['lr'] * self.scale
            
        for res in ['lr', 'hr']:
            slices[res] = np.array(
                [
                    {'x': (x, x + self.patch_size[res]), 'y': (y, y + self.patch_size[res])}
                    for x, y in zip(top_left['x'][res], top_left['y'][res])
                ])
            
        for slice_index, s in enumerate(slices['lr']):
            candidate_crop = imgs['lr'][s['x'][0]: s['x'][1], s['y'][0]: s['y'][1], slice(None)]
            if self._not_flat(candidate_crop, flatness) or n == 0:
                crops['lr'].append(candidate_crop)
                accepted_slices['lr'].append(slice_index)
            else:
                n -= 1
            if len(crops['lr']) == batch_size:
                break
        # Принимает пакет только в том случае, если стандартное отклонение интенсивности пикселей выше заданного порога 
        # ИЛИ никакие патчи не могут быть дополнительно отброшены (уже отброшено n)      
        accepted_slices['hr'] = slices['hr'][accepted_slices['lr']]
        # Квадратные кадры размером patch_size берутся из выделенных верхних левых углов 
        for s in accepted_slices['hr']:
            candidate_crop = imgs['hr'][s['x'][0]: s['x'][1], s['y'][0]: s['y'][1], slice(None)]
            crops['hr'].append(candidate_crop)
        
        crops['lr'] = np.array(crops['lr'])
        crops['hr'] = np.array(crops['hr'])
        return crops
    
    # Поворачивает и переворачивает входное изображение в соответствии с transform_selection
    def _apply_transform(self, img, transform_selection):       
        rotate = {
            0: lambda x: x,
            1: lambda x: np.rot90(x, k=1, axes=(1, 0)),  # вращение вправо
            2: lambda x: np.rot90(x, k=1, axes=(0, 1)),  # вращение влево
        }        
        flip = {
            0: lambda x: x,
            1: lambda x: np.flip(x, 0),  # отражение по горизонтальной оси
            2: lambda x: np.flip(x, 1),  # отражение по вертикальной оси
        }        
        rot_direction = transform_selection[0]
        flip_axis = transform_selection[1]        
        img = rotate[rot_direction](img)
        img = flip[flip_axis](img)        
        return img
    
    # Преобразует каждое отдельное изображение в пакете независимо
    def _transform_batch(self, batch, transforms):        
        t_batch = np.array(
            [self._apply_transform(img, transforms[i]) for i, img in enumerate(batch)])
        return t_batch
   
    # Возвращает словарь с ключами ('lr', 'hr'), содержащими обучающие пакеты патчей изображений низкого и высокого разрешения
    # batch_size - integer
    # flatness - float [0,1] - jпределяет уровень детализации, которому должны соответствовать патчи
    # 0 означает, что принимается любой патч
    def get_batch(self, batch_size, idx=None, flatness=0.0):        
        if not idx:
            # randomly select one image. idx is given at validation time.
            idx = np.random.choice(range(len(self.img_list['hr'])))
        img = {}
        for res in ['lr', 'hr']:
            img_path = os.path.join(self.folders[res], self.img_list[res][idx])
            img[res] = imageio.imread(img_path) / 255.0
        batch = self._crop_imgs(img, batch_size, flatness)
        transforms = np.random.randint(0, 3, (batch_size, 2))
        batch['lr'] = self._transform_batch(batch['lr'], transforms)
        batch['hr'] = self._transform_batch(batch['hr'], transforms)       
        return batch
    
    # Возвращает пакет для каждого изображения в наборе для тестирования
    def get_validation_batches(self, batch_size):
        if self.n_validation_samples:
            batches = []
            for idx in range(self.n_validation_samples):
                batches.append(self.get_batch(batch_size, idx, flatness=0.0))
            return batches
        else:
            self.logger.error(
                'No validation set size specified. (not operating in a validation set?)'
            )
            raise ValueError(
                'No validation set size specified. (not operating in a validation set?)'
            )
    
    # Возвращает пакет для каждого изображения в проверочном наборе
    # Подготовка к передаче в модуль оценки
    def get_validation_set(self, batch_size):
        if self.n_validation_samples:
            batches = self.get_validation_batches(batch_size)
            valid_set = {'lr': [], 'hr': []}
            for batch in batches:
                for res in ('lr', 'hr'):
                    valid_set[res].extend(batch[res])
            for res in ('lr', 'hr'):
                valid_set[res] = np.array(valid_set[res])
            return valid_set
        else:
            self.logger.error(
                'No validation set size specified. (not operating in a validation set?)'
            )
            raise ValueError(
                'No validation set size specified. (not operating in a validation set?)'
            )
# Преобразование трехмерного массива в масштабированный четырехмерный пакет размера 1            
def process_array(image_array, expand=True):
    image_batch = image_array / 255.0
    if expand:
        image_batch = np.expand_dims(image_batch, axis=0)
    return image_batch

# Преобразует 4-мерный выходной тензор в подходящий формат изображения
def process_output(output_tensor):    
    sr_img = output_tensor.clip(0, 1) * 255
    sr_img = np.uint8(sr_img)
    return sr_img

# Заполняет изображение патчем со значениями края отступа
def pad_patch(image_patch, padding_size, channel_last=True):
    if channel_last:
        return np.pad(
            image_patch,
            ((padding_size, padding_size), (padding_size, padding_size), (0, 0)),
            'edge',)
    else:
        return np.pad(
            image_patch,
            ((0, 0), (padding_size, padding_size), (padding_size, padding_size)),
            'edge',)
def unpad_patches(image_patches, padding_size):
    return image_patches[:, padding_size:-padding_size, padding_size:-padding_size, :]

# Разбивает изображение на частично перекрывающиеся участки
# Патчи перекрываются на padding_size пикселей
# Заполняется изображение дважды:
# - сначала заполняется до размера, кратного размеру патча
# - затем идет работа с отступами на границах
def split_image_into_overlapping_patches(image_array, patch_size, padding_size=2):
    # image_array - numpy массив входного изображения
    # patch_size - размер патчей из исходного изображения (без отступов)
    # padding_size - размер области перекрытия
    xmax, ymax, _ = image_array.shape
    x_remainder = xmax % patch_size
    y_remainder = ymax % patch_size
    # берется по модулю, чтобы избжать расширения patch_size вместо 0
    x_extend = (patch_size - x_remainder) % patch_size
    y_extend = (patch_size - y_remainder) % patch_size 
    # утверждение, что образ делится на обычные патчи
    extended_image = np.pad(image_array, ((0, x_extend), (0, y_extend), (0, 0)), 'edge')
    # добавление отступов вокруг изображения для упрощения вычислений
    padded_image = pad_patch(extended_image, padding_size, channel_last=True)    
    xmax, ymax, _ = padded_image.shape
    patches = []
    x_lefts = range(padding_size, xmax - padding_size, patch_size)
    y_tops = range(padding_size, ymax - padding_size, patch_size)
    
    for x in x_lefts:
        for y in y_tops:
            x_left = x - padding_size
            y_top = y - padding_size
            x_right = x + patch_size + padding_size
            y_bottom = y + patch_size + padding_size
            patch = padded_image[x_left:x_right, y_top:y_bottom, :]
            patches.append(patch)    
    return np.array(patches), padded_image.shape

# восстановление изображения из перекрывающихся участков
# после масштабирования размера и отступы также должны быть масштабированы
def stich_together(patches, padded_image_shape, target_shape, padding_size=4):
    # patches - патчи, полученные с помощью split_image_into_overlapping_patches
    # padded_image_shape - размер дополненного изображения, созданного в split_image_into_overlapping_patches
    # target_shape - размер конечного изображения
    # padding_size - размер области перекрытия
    xmax, ymax, _ = padded_image_shape
    patches = unpad_patches(patches, padding_size)
    patch_size = patches.shape[1]
    n_patches_per_row = ymax // patch_size    
    complete_image = np.zeros((xmax, ymax, 3))    
    row = -1
    col = 0
    for i in range(len(patches)):
        if i % n_patches_per_row == 0:
            row += 1
            col = 0
        complete_image[
        row * patch_size: (row + 1) * patch_size, col * patch_size: (col + 1) * patch_size, :
        ] = patches[i]
        col += 1
    return complete_image[0: target_shape[0], 0: target_shape[1], :]

# Оценка значения PSNR: PSNR = 20 * log10(MAXp) - 10 * log10(MSE)
def PSNR(y_true, y_pred, MAXp=1):
    # y_true - реальное значение
    # y_pred - предсказываемое значение
    # MAXp - максимальное значение диапазона пикселей (по умолчанию=1).
    return -10.0 * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.0)

# Изображение имеет значения от 0 до 1
def RGB_to_Y(image):    
    R = image[:, :, :, 0]
    G = image[:, :, :, 1]
    B = image[:, :, :, 2]    
    Y = 16 + (65.738 * R) + 129.057 * G + 25.064 * B
    return Y / 255.0

# Оценивает значение PSNR на канале Y: PSNR = 20 * log10(MAXp) - 10 * log10(MSE)
def PSNR_Y(y_true, y_pred, MAXp=1):
    y_true = RGB_to_Y(y_true)
    y_pred = RGB_to_Y(y_pred)
    return -10.0 * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.0)
logger = get_logger(__name__)

# Задание аргументов
def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction', action='store_true', dest='prediction')
    parser.add_argument('--training', action='store_true', dest='training')
    parser.add_argument('--summary', action='store_true', dest='summary')
    parser.add_argument('--default', action='store_true', dest='default')
    parser.add_argument('--config', action='store', dest='config_file')
    return parser

# Parse CLI arguments - анализ аргументов командной строки
def parse_args():
    parser = _get_parser()
    args = vars(parser.parse_args())
    if args['prediction'] and args['training']:
        logger.error('Select only prediction OR training.')
        raise ValueError('Select only prediction OR training.')
    return args

def get_timestamp():
    ts = datetime.now()
    time_stamp = '{y}-{m:02d}-{d:02d}_{h:02d}{mm:02d}'.format(
        y=ts.year, m=ts.month, d=ts.day, h=ts.hour, mm=ts.minute
    )
    return time_stamp

def check_parameter_keys(parameter, needed_keys, optional_keys=None, default_value=None):
    if needed_keys:
        for key in needed_keys:
            if key not in parameter:
                logger.error('{p} is missing key {k}'.format(p=parameter, k=key))
                raise
    if optional_keys:
        for key in optional_keys:
            if key not in parameter:
                logger.info('Setting {k} in {p} to {d}'.format(k=key, p=parameter, d=default_value))
                parameter[key] = default_value

# Извлекает параметры архитектуры из имени файла весов.
# Работает только со стандартизированным именем веса
def get_config_from_weights(w_path, arch_params, name):    
    w_path = os.path.basename(w_path)
    parts = w_path.split(name)[1]
    parts = parts.split('_')[0]
    parts = parts.split('-')
    new_param = {}
    for param in arch_params:
        param_part = [x for x in parts if param in x]
        param_value = int(param_part[0].split(param)[1])
        new_param[param] = param_value
    return new_param

# Выбор CLI с заданными параметрами
def select_option(options, message='', val=None):
    while val not in options:
        val = input(message)
        if val not in options:
            logger.error('Invalid choice.')
    return val

# Множественный выбор CLI с заданными параметрами
def select_multiple_options(options, message='', val=None):
    n_options = len(options)
    valid_selections = False
    selected_options = []
    while not valid_selections:
        for i, opt in enumerate(np.sort(options)):
            logger.info('{}: {}'.format(i, opt))
        val = input(message + ' (space separated selection)\n')
        vals = val.split(' ')
        valid_selections = True
        for v in vals:
            if int(v) not in list(range(n_options)):
                logger.error('Invalid choice.')
                valid_selections = False
            else:
                selected_options.append(options[int(v)])    
    return selected_options

# Выбор логического значения CLI
def select_bool(message=''):
    options = ['y', 'n']
    message = message + ' (' + '/'.join(options) + ') '
    val = None
    while val not in options:
        val = input(message)
        if val not in options:
            logger.error('Input y (yes) or n (no).')
    if val == 'y':
        return True
    elif val == 'n':
        return False

# Выбор CLI с неотрицательным числом с плавающей запятой
def select_positive_float(message=''):
    value = -1
    while value < 0:
        value = float(input(message))
        if value < 0:
            logger.error('Invalid choice.')
    return value

# CLI выбор неотрицательного целого числа
def select_positive_integer(message='', value=-1):
    while value < 0:
        value = int(input(message))
        if value < 0:
            logger.error('Invalid choice.')
    return value

# Выбор весов из CLI
def browse_weights(weights_dir, model='generator'):    
    exit = False
    while exit is False:
        weights = np.sort(os.listdir(weights_dir))[::-1]
        print_sel = dict(zip(np.arange(len(weights)), weights))
        for k in print_sel.keys():
            logger_message = '{item_n}: {item} \n'.format(item_n=k, item=print_sel[k])
            logger.info(logger_message)        
        sel = select_positive_integer('>>> Select folder or weights for {}\n'.format(model))
        if weights[sel].endswith('hdf5'):
            weights_path = os.path.join(weights_dir, weights[sel])
            exit = True
        else:
            weights_dir = os.path.join(weights_dir, weights[sel])
    return weights_path

# Интерфейс командной строки для настройки сеанса обучения или прогнозирования
# Принимает в качестве входных данных путь к файлу конфигурации (без расширения '.py') и аргументы, анализируемые из CLI
def setup(config_file='config.yml', default=False, training=False, prediction=False):
    conf = yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader)   
    if training:
        session_type = 'training'
    elif prediction:
        session_type = 'prediction'
    else:
        message = '(t)raining or (p)rediction? (t/p) '
        session_type = {'t': 'training', 'p': 'prediction'}[select_option(['t', 'p'], message)]
    if default:
        all_default = 'y'
    else:
        all_default = select_bool('Default options for everything?')    
    if all_default:
        generator = conf['default']['generator']
        if session_type == 'prediction':
            dataset = conf['default']['test_set']
            conf['generators'][generator] = get_config_from_weights(
                conf['weights_paths']['generator'], conf['generators'][generator], generator)
        elif session_type == 'training':
            dataset = conf['default']['training_set']        
        return session_type, generator, conf, dataset    
    logger.info('Select SR (generator) network')
    generators = {}
    for i, gen in enumerate(conf['generators']):
        generators[str(i)] = gen
        logger.info('{}: {}'.format(i, gen))
    generator = generators[select_option(generators)]    
    load_weights = input('Load pretrained weights for {}? ([y]/n/d) '.format(generator))
    if load_weights == 'n':
        default = select_bool('Load default parameters for {}?'.format(generator))
        if not default:
            for param in conf['generators'][generator]:
                value = select_positive_integer(message='{}:'.format(param))
                conf['generators'][generator][param] = value
        else:
            logger.info('Default {} parameters.'.format(generator))
    elif (load_weights == 'd') and (conf['weights_paths']['generator']):
        logger.info('Loading default weights for {}'.format(generator))
        logger.info(conf['weights_paths']['generator'])
        conf['generators'][generator] = get_config_from_weights(
            conf['weights_paths']['generator'], conf['generators'][generator], generator)
    else:
        conf['weights_paths']['generator'] = browse_weights(conf['dirs']['weights'], generator)
        conf['generators']['generator'] = get_config_from_weights(
            conf['weights_paths']['generator'], conf['generators'][generator], generator)
    logger.info('{} parameters:'.format(generator))
    logger.info(conf['generators'][generator])    
    if session_type == 'training':
        default_loss_weights = select_bool('Use default weights for loss components?')
        if not default_loss_weights:
            conf['loss_weights']['generator'] = select_positive_float(
                'Input coefficient for pixel-wise generator loss component ')
        use_discr = select_bool('Use an Adversarial Network?')
        if use_discr:
            conf['default']['discriminator'] = True
            discr_w = select_bool('Use pretrained discriminator weights?')
            if discr_w:
                conf['weights_paths']['discriminator'] = browse_weights(
                    conf['dirs']['weights'], 'discriminator')
            if not default_loss_weights:
                conf['loss_weights']['discriminator'] = select_positive_float(
                    'Input coefficient for Adversarial loss component ')
        use_feature_extractor = select_bool('Use feature extractor?')
        if use_feature_extractor:
            conf['default']['feature_extractor'] = True
            if not default_loss_weights:
                conf['loss_weights']['feature_extractor'] = select_positive_float(
                    'Input coefficient for conv features loss component ')
        default_metrics = select_bool('Monitor default metrics?')
        if not default_metrics:
            suggested_list = suggest_metrics(use_discr, use_feature_extractor)
            selected_metrics = select_multiple_options(
                list(suggested_list.keys()), message='Select metrics to monitor.') 
            conf['session']['training']['monitored_metrics'] = {}
            for metric in selected_metrics:
                conf['session']['training']['monitored_metrics'][metric] = suggested_list[metric]
            print(conf['session']['training']['monitored_metrics'])    
    dataset = select_dataset(session_type, conf)    
    return session_type, generator, conf, dataset

def suggest_metrics(discriminator=False, feature_extractor=False, loss_weights={}):
    suggested_metrics = {}
    if not discriminator and not feature_extractor:
        suggested_metrics['val_loss'] = 'min'
        suggested_metrics['train_loss'] = 'min'
        suggested_metrics['val_PSNR'] = 'max'
        suggested_metrics['train_PSNR'] = 'max'
    if feature_extractor or discriminator:
        suggested_metrics['val_generator_loss'] = 'min'
        suggested_metrics['train_generator_loss'] = 'min'
        suggested_metrics['val_generator_PSNR'] = 'max'
        suggested_metrics['train_generator_PSNR'] = 'max'
    if feature_extractor:
        suggested_metrics['val_feature_extractor_loss'] = 'min'
        suggested_metrics['train_feature_extractor_loss'] = 'min'
    return suggested_metrics

# Фрагмент командной строки для выбора набора данных для обучения
def select_dataset(session_type, conf):
    if session_type == 'training':
        logger.info('Select training set')
        datasets = {}
        for i, data in enumerate(conf['training_sets']):
            datasets[str(i)] = data
            logger.info('{}: {}'.format(i, data))
        dataset = datasets[select_option(datasets)]        
        return dataset
    else:
        logger.info('Select test set')
        datasets = {}
        for i, data in enumerate(conf['test_sets']):
            datasets[str(i)] = data
            logger.info('{}: {}'.format(i, data))
        dataset = datasets[select_option(datasets)]        
        return dataset

# Коллекция полезных функций для управления тренировками    
class TrainerHelper:
    # generator - Keras model, супермасштабирующая или генераторная сеть
    # logs_dir - путь к каталогу, в котором сохраняются журналы tensorboard
    # weights_dir - путь к каталогу, в котором сохранены веса
    # lr_train_dir - путь к каталогу, в котором хранятся изображения LR
    # feature_extractor - Keras model, cеть экстракторов признаков для компонента глубоких признаков функции потери
    # discriminator - Keras model, дискриминаторная сеть для состязательного компонента потери
    # dataname - string, используется для определения того, какой набор данных используется для сеанса обучения.
    # fallback_save_every_n_epochs - integer, определяет, через сколько эпох сохранять веса (если не идет улучшение метрики)
    # max_n_best_weights - максимальное количество весов, которые являются лучшими для некоторой сохраненной метрики
    # max_n_other_weights - максимальное количество не лучших весов, которые сохраняются
    def __init__(
            self,
            generator,
            weights_dir,
            logs_dir,
            lr_train_dir,
            feature_extractor=None,
            discriminator=None,
            dataname=None,
            weights_generator=None,
            weights_discriminator=None,
            fallback_save_every_n_epochs=2,
            max_n_other_weights=5,
            max_n_best_weights=5,
    ):
        self.generator = generator
        self.dirs = {'logs': Path(logs_dir), 'weights': Path(weights_dir)}
        self.feature_extractor = feature_extractor
        self.discriminator = discriminator
        self.dataname = dataname    
        # для генератора
        if weights_generator:
            self.pretrained_generator_weights = Path(weights_generator)
        else:
            self.pretrained_generator_weights = None   
        # для дискриминатора
        if weights_discriminator:
            self.pretrained_discriminator_weights = Path(weights_discriminator)
        else:
            self.pretrained_discriminator_weights = None
        
        self.fallback_save_every_n_epochs = fallback_save_every_n_epochs
        self.lr_dir = Path(lr_train_dir)
        self.basename = self._make_basename()
        self.session_id = self.get_session_id(basename=None)
        self.session_config_name = 'session_config.yml'
        self.callback_paths = self._make_callback_paths()
        self.weights_name = self._weights_name(self.callback_paths)
        self.best_metrics = {}
        self.since_last_epoch = 0
        self.max_n_other_weights = max_n_other_weights
        self.max_n_best_weights = max_n_best_weights
        self.logger = get_logger(__name__)
    
    # Объединяет имя генератора и параметры его архитектуры
    def _make_basename(self):
        gen_name = self.generator.name
        params = [gen_name]
        for param in np.sort(list(self.generator.params.keys())):
            params.append('{g}{p}'.format(g=param, p=self.generator.params[param]))
        return '-'.join(params)
    
    # Возвращает уникальный идентификатор сеанса
    def get_session_id(self, basename):
        time_stamp = get_timestamp()
        if basename:
            session_id = '{b}_{ts}'.format(b=basename, ts=time_stamp)
        else:
            session_id = time_stamp
        return session_id
    
    # Проверяет, доступен ли файл session_config.yml в папке предварительно обученных весов
    def _get_previous_conf(self):
        if self.pretrained_generator_weights:
            session_config_path = (
                    self.pretrained_generator_weights.parent / self.session_config_name)
            if session_config_path.exists():
                return yaml.load(session_config_path.read_text(), Loader=yaml.FullLoader)
            else:
                self.logger.warning('Could not find previous configuration')
                return {}        
        return {}
    
    # Добавляет к существующим настройкам (если есть) текущий словарь настроек по ключу session_id
    def update_config(self, training_settings):
        session_settings = self._get_previous_conf()
        session_settings.update({self.session_id: training_settings})        
        return session_settings
    
    # Создает пути, используемые для управления журналами и хранилищем весов
    def _make_callback_paths(self):
        callback_paths = {}
        callback_paths['weights'] = self.dirs['weights'] / self.basename / self.session_id
        callback_paths['logs'] = self.dirs['logs'] / self.basename / self.session_id
        return callback_paths
    
    # Создает строку, используемую для обозначения весов сеанса обучения
    def _weights_name(self, callback_paths):
        w_name = {
            'generator': callback_paths['weights']
                         / (self.basename + '{metric}_epoch{epoch:03d}.hdf5')
        }
        if self.discriminator:
            w_name.update(
                {
                    'discriminator': callback_paths['weights']
                                     / (self.discriminator.name + '{metric}_epoch{epoch:03d}.hdf5')
                }
            )
        return w_name
    
    # Вывод параметров обучения
    def print_training_setting(self, settings):
        self.logger.info('\nTraining details:')
        for k in settings[self.session_id]:
            if isinstance(settings[self.session_id][k], dict):
                self.logger.info('  {}: '.format(k))
                for kk in settings[self.session_id][k]:
                    self.logger.info(
                        '    {key}: {value}'.format(
                            key=kk, value=str(settings[self.session_id][k][kk])
                        )
                    )
            else:
                self.logger.info(
                    '  {key}: {value}'.format(key=k, value=str(settings[self.session_id][k]))
                )
    # Сохраняет вес моделей, отличных от None
    def _save_weights(self, epoch, generator, discriminator=None, metric=None, best=False):
        if best:
            gen_path = self.weights_name['generator'].with_name(
                (self.weights_name['generator'].name).format(
                    metric='_best-' + metric, epoch=epoch + 1
                )
            )
        else:
            gen_path = self.weights_name['generator'].with_name(
                (self.weights_name['generator'].name).format(metric='', epoch=epoch + 1)
            )
        # Не может сохранить модель из-за слоя TF внутри Лямбда (Pixel Shuffle)
        generator.save_weights(gen_path.as_posix())
        if discriminator:
            if best:
                discr_path = self.weights_name['discriminator'].with_name(
                    (self.weights_name['discriminator'].name).format(
                        metric='_best-' + metric, epoch=epoch + 1
                    )
                )
            else:
                discr_path = self.weights_name['discriminator'].with_name(
                    (self.weights_name['discriminator'].name).format(metric='', epoch=epoch + 1)
                )
            discriminator.model.save_weights(discr_path.as_posix())
        try:
            self._remove_old_weights(self.max_n_other_weights, max_best=self.max_n_best_weights)
        except Exception as e:
            self.logger.warning('Could not remove weights: {}'.format(e))
    
    # Сканирует папку с весами и удаляет все, кроме:
    # - max_best новейшие «лучшие» веса
    # - max_n_weights самые последние веса "других"
    def _remove_old_weights(self, max_n_weights, max_best=5):
        w_list = {}
        w_list['all'] = [w for w in self.callback_paths['weights'].iterdir() if '.hdf5' in w.name]
        w_list['best'] = [w for w in w_list['all'] if 'best' in w.name]
        w_list['others'] = [w for w in w_list['all'] if w not in w_list['best']]
        # remove older best
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
    
    # Управляет операциями, выполняемыми в конце каждой эпохи: проверка метрик, сохранение веса, логирование
    def on_epoch_end(self, epoch, losses, generator, discriminator=None, metrics={}):
        self.logger.info(losses)
        monitor_op = {'max': np.greater, 'min': np.less}
        extreme = {'max': -np.Inf, 'min': np.Inf}
        for metric in metrics:
            if metric in losses.keys():
                if metric not in self.best_metrics.keys():
                    self.best_metrics[metric] = extreme[metrics[metric]]
                if monitor_op[metrics[metric]](losses[metric], self.best_metrics[metric]):
                    self.logger.info(
                        '{} improved from {:10.5f} to {:10.5f}'.format(
                            metric, self.best_metrics[metric], losses[metric]
                        )
                    )
                    self.logger.info('Saving weights')
                    self.best_metrics[metric] = losses[metric]
                    self._save_weights(epoch, generator, discriminator, metric=metric, best=True)
                    self.since_last_epoch = 0
                    return True
                else:
                    self.logger.info('{} did not improve.'.format(metric))
                    if self.since_last_epoch >= self.fallback_save_every_n_epochs:
                        self.logger.info('Saving weights anyways.')
                        self._save_weights(epoch, generator, discriminator, best=False)
                        self.since_last_epoch = 0
                        return True
            else:
                self.logger.warning('{} is not monitored, cannot save weights.'.format(metric))
        self.since_last_epoch += 1
        return False
    
    # Извлекает номер последней эпохи из названия стандартизированных весов
    # Работает только в том случае, если веса содержат «эпоху», за которой следуют 3 целых числа, например: some-architectureepoch023suffix.hdf5
    def epoch_n_from_weights_name(self, w_name):
        try:
            starting_epoch = int(w_name.split('epoch')[1][0:3])
        except Exception as e:
            self.logger.warning(
                'Could not retrieve starting epoch from the weights name: \n{}'.format(w_name)
            )
            self.logger.error(e)
            starting_epoch = 0
        return starting_epoch
    
    # Функция, выполняемая перед тренировкой
    def initialize_training(self, object):
        # Завершает большинство функций этого класса:
        # - загружает веса, если они заданы, имена генераторов для сеанса
        # - создает каталоги и распечатывает тренировку
        object.weights_generator = self.pretrained_generator_weights
        object.weights_discriminator = self.pretrained_discriminator_weights
        object._load_weights()
        w_name = object.weights_generator
        if w_name:
            last_epoch = self.epoch_n_from_weights_name(w_name.name)
        else:
            last_epoch = 0
        self.callback_paths = self._make_callback_paths()
        self.callback_paths['weights'].mkdir(parents=True)
        self.callback_paths['logs'].mkdir(parents=True)
        object.settings['training_parameters']['starting_epoch'] = last_epoch
        self.settings = self.update_config(object.settings)
        self.print_training_setting(self.settings)
        yaml.dump(
            self.settings, (self.callback_paths['weights'] / self.session_config_name).open('w')
        )
        return last_epoch

# Объект класса для настройки и проведения обучения
# Принимает на вход генератор, который создает изображения SR
# Условно также сеть дискриминатора и генератор признаков для построения компонентов потери
# Компилирует модель (модели) и обучает в стиле GANS, если предоставлен дискриминатор, в противном случае проводит обычное обучение ISR
class Trainer:
    # generator: Keras model, the super-scaling, or generator, network
    # discriminator: Keras model, the discriminator network for the adversarial component of the perceptual loss.
    # feature_extractor: Keras model, feature extractor network for the deep features component of perceptual loss function.
    # lr_train_dir: path to the directory containing the Low-Res images for training.
    # hr_train_dir: path to the directory containing the High-Res images for training.
    # lr_valid_dir: path to the directory containing the Low-Res images for validation.
    # hr_valid_dir: path to the directory containing the High-Res images for validation.
    # learning_rate: float.
    # loss_weights: dictionary, use to weigh the components of the loss function. Contains 'generator' for the generator loss component, and can contain 'discriminator' and 'feature_extractor' for the discriminator and deep features components respectively.
    # logs_dir: path to the directory where the tensorboard logs are saved
    # weights_dir: path to the directory where the weights are saved
    #  dataname: string, used to identify what dataset is used for the training session
    # weights_generator: path to the pre-trained generator's weights, for transfer learning
    # weights_discriminator: path to the pre-trained discriminator's weights, for transfer learning
    #  n_validation:integer, number of validation samples used at training from the validation set
    # flatness: dictionary. Determines determines the 'flatness' threshold level for the training patches. See the TrainerHelper class for more details
    # lr_decay_frequency: integer, every how many epochs the learning rate is reduced
    # lr_decay_factor: 0 < float <1, learning rate reduction multiplicative factor

    def __init__(
            self,
            generator,
            discriminator,
            feature_extractor,
            lr_train_dir,
            hr_train_dir,
            lr_valid_dir,
            hr_valid_dir,
            loss_weights={'generator': 1.0, 'discriminator': 0.003, 'feature_extractor': 1 / 12},
            log_dirs={'logs': 'logs', 'weights': 'weights'},
            fallback_save_every_n_epochs=2,
            dataname=None,
            weights_generator=None,
            weights_discriminator=None,
            n_validation=None,
            flatness={'min': 0.0, 'increase_frequency': None, 'increase': 0.0, 'max': 0.0},
            learning_rate={'initial_value': 0.0004, 'decay_frequency': 100, 'decay_factor': 0.5},
            adam_optimizer={'beta1': 0.9, 'beta2': 0.999, 'epsilon': None},
            losses={
                'generator': 'mae',
                'discriminator': 'binary_crossentropy',
                'feature_extractor': 'mse',
            },
            metrics={'generator': 'PSNR_Y'},
    ):
        self.generator = generator
        self.discriminator = discriminator
        self.feature_extractor = feature_extractor
        self.scale = generator.scale
        self.lr_patch_size = generator.patch_size
        self.learning_rate = learning_rate
        self.loss_weights = loss_weights
        self.weights_generator = weights_generator
        self.weights_discriminator = weights_discriminator
        self.adam_optimizer = adam_optimizer
        self.dataname = dataname
        self.flatness = flatness
        self.n_validation = n_validation
        self.losses = losses
        self.log_dirs = log_dirs
        self.metrics = metrics
        if self.metrics['generator'] == 'PSNR_Y':
            self.metrics['generator'] = PSNR_Y
        elif self.metrics['generator'] == 'PSNR':
            self.metrics['generator'] = PSNR
        self._parameters_sanity_check()
        self.model = self._combine_networks()

        self.settings = {}
        self.settings['training_parameters'] = locals()
        self.settings['training_parameters']['lr_patch_size'] = self.lr_patch_size
        self.settings = self.update_training_config(self.settings)

        self.logger = get_logger(__name__)

        self.helper = TrainerHelper(
            generator=self.generator,
            weights_dir=log_dirs['weights'],
            logs_dir=log_dirs['logs'],
            lr_train_dir=lr_train_dir,
            feature_extractor=self.feature_extractor,
            discriminator=self.discriminator,
            dataname=dataname,
            weights_generator=self.weights_generator,
            weights_discriminator=self.weights_discriminator,
            fallback_save_every_n_epochs=fallback_save_every_n_epochs,
        )

        self.train_dh = DataHandler(
            lr_dir=lr_train_dir,
            hr_dir=hr_train_dir,
            patch_size=self.lr_patch_size,
            scale=self.scale,
            n_validation_samples=None,
        )
        self.valid_dh = DataHandler(
            lr_dir=lr_valid_dir,
            hr_dir=hr_valid_dir,
            patch_size=self.lr_patch_size,
            scale=self.scale,
            n_validation_samples=n_validation,
        )

    def _parameters_sanity_check(self):
        """ Parameteres sanity check. """

        if self.discriminator:
            assert self.lr_patch_size * self.scale == self.discriminator.patch_size
            self.adam_optimizer
        if self.feature_extractor:
            assert self.lr_patch_size * self.scale == self.feature_extractor.patch_size

        check_parameter_keys(
            self.learning_rate,
            needed_keys=['initial_value'],
            optional_keys=['decay_factor', 'decay_frequency'],
            default_value=None,
        )
        check_parameter_keys(
            self.flatness,
            needed_keys=[],
            optional_keys=['min', 'increase_frequency', 'increase', 'max'],
            default_value=0.0,
        )
        check_parameter_keys(
            self.adam_optimizer,
            needed_keys=['beta1', 'beta2'],
            optional_keys=['epsilon'],
            default_value=None,
        )
        check_parameter_keys(self.log_dirs, needed_keys=['logs', 'weights'])

    def _combine_networks(self):
        """
        Constructs the combined model which contains the generator network,
        as well as discriminator and geature extractor, if any are defined.
        """

        lr = Input(shape=(self.lr_patch_size,) * 2 + (3,))
        sr = self.generator.model(lr)
        outputs = [sr]
        losses = [self.losses['generator']]
        loss_weights = [self.loss_weights['generator']]

        if self.discriminator:
            self.discriminator.model.trainable = False
            validity = self.discriminator.model(sr)
            outputs.append(validity)
            losses.append(self.losses['discriminator'])
            loss_weights.append(self.loss_weights['discriminator'])
        if self.feature_extractor:
            self.feature_extractor.model.trainable = False
            sr_feats = self.feature_extractor.model(sr)
            outputs.extend([*sr_feats])
            losses.extend([self.losses['feature_extractor']] * len(sr_feats))
            loss_weights.extend(
                [self.loss_weights['feature_extractor'] / len(sr_feats)] * len(sr_feats)
            )
        combined = Model(inputs=lr, outputs=outputs)
        # https://stackoverflow.com/questions/42327543/adam-optimizer-goes-haywire-after-200k-batches-training-loss-grows
        optimizer = Adam(
            beta_1=self.adam_optimizer['beta1'],
            beta_2=self.adam_optimizer['beta2'],
            lr=self.learning_rate['initial_value'],
            epsilon=self.adam_optimizer['epsilon'],
        )
        combined.compile(
            loss=losses, loss_weights=loss_weights, optimizer=optimizer, metrics=self.metrics
        )
        return combined

    def _lr_scheduler(self, epoch):
        """ Scheduler for the learning rate updates. """

        n_decays = epoch // self.learning_rate['decay_frequency']
        lr = self.learning_rate['initial_value'] * (self.learning_rate['decay_factor'] ** n_decays)
        # no lr below minimum control 10e-7
        return max(1e-7, lr)

    def _flatness_scheduler(self, epoch):
        if self.flatness['increase']:
            n_increases = epoch // self.flatness['increase_frequency']
        else:
            return self.flatness['min']

        f = self.flatness['min'] + n_increases * self.flatness['increase']

        return min(self.flatness['max'], f)

    def _load_weights(self):
        """
        Loads the pretrained weights from the given path, if any is provided.
        If a discriminator is defined, does the same.
        """

        if self.weights_generator:
            self.model.get_layer('generator').load_weights(str(self.weights_generator))

        if self.discriminator:
            if self.weights_discriminator:
                self.model.get_layer('discriminator').load_weights(str(self.weights_discriminator))
                self.discriminator.model.load_weights(str(self.weights_discriminator))

    def _format_losses(self, prefix, losses, model_metrics):
        """ Creates a dictionary for tensorboard tracking. """

        return dict(zip([prefix + m for m in model_metrics], losses))

    def update_training_config(self, settings):
        """ Summarizes training setting. """

        _ = settings['training_parameters'].pop('weights_generator')
        _ = settings['training_parameters'].pop('self')
        _ = settings['training_parameters'].pop('generator')
        _ = settings['training_parameters'].pop('discriminator')
        _ = settings['training_parameters'].pop('feature_extractor')
        settings['generator'] = {}
        settings['generator']['name'] = self.generator.name
        settings['generator']['parameters'] = self.generator.params
        settings['generator']['weights_generator'] = self.weights_generator

        _ = settings['training_parameters'].pop('weights_discriminator')
        if self.discriminator:
            settings['discriminator'] = {}
            settings['discriminator']['name'] = self.discriminator.name
            settings['discriminator']['weights_discriminator'] = self.weights_discriminator
        else:
            settings['discriminator'] = None

        if self.discriminator:
            settings['feature_extractor'] = {}
            settings['feature_extractor']['name'] = self.feature_extractor.name
            settings['feature_extractor']['layers'] = self.feature_extractor.layers_to_extract
        else:
            settings['feature_extractor'] = None

        return settings

    def train(self, epochs, steps_per_epoch, batch_size, monitored_metrics):
        """
        Carries on the training for the given number of epochs.
        Sends the losses to Tensorboard.

        Args:
            epochs: how many epochs to train for.
            steps_per_epoch: how many batches epoch.
            batch_size: amount of images per batch.
            monitored_metrics: dictionary, the keys are the metrics that are monitored for the weights
                saving logic. The values are the mode that trigger the weights saving ('min' vs 'max').
        """

        self.settings['training_parameters']['steps_per_epoch'] = steps_per_epoch
        self.settings['training_parameters']['batch_size'] = batch_size
        starting_epoch = self.helper.initialize_training(
            self
        )  # load_weights, creates folders, creates basename

        self.tensorboard = TensorBoard(log_dir=str(self.helper.callback_paths['logs']))
        self.tensorboard.set_model(self.model)

        # validation data
        validation_set = self.valid_dh.get_validation_set(batch_size)
        y_validation = [validation_set['hr']]
        if self.discriminator:
            discr_out_shape = list(self.discriminator.model.outputs[0].shape)[1:4]
            valid = np.ones([batch_size] + discr_out_shape)
            fake = np.zeros([batch_size] + discr_out_shape)
            validation_valid = np.ones([len(validation_set['hr'])] + discr_out_shape)
            y_validation.append(validation_valid)
        if self.feature_extractor:
            validation_feats = self.feature_extractor.model.predict(validation_set['hr'])
            y_validation.extend([*validation_feats])

        for epoch in range(starting_epoch, epochs):
            self.logger.info('Epoch {e}/{tot_eps}'.format(e=epoch, tot_eps=epochs))
            K.set_value(self.model.optimizer.lr, self._lr_scheduler(epoch=epoch))
            self.logger.info('Current learning rate: {}'.format(K.eval(self.model.optimizer.lr)))

            flatness = self._flatness_scheduler(epoch)
            if flatness:
                self.logger.info('Current flatness treshold: {}'.format(flatness))

            epoch_start = time()
            for step in tqdm(range(steps_per_epoch)):
                batch = self.train_dh.get_batch(batch_size, flatness=flatness)
                y_train = [batch['hr']]
                training_losses = {}

                ## Discriminator training
                if self.discriminator:
                    sr = self.generator.model.predict(batch['lr'])
                    d_loss_real = self.discriminator.model.train_on_batch(batch['hr'], valid)
                    d_loss_fake = self.discriminator.model.train_on_batch(sr, fake)
                    d_loss_fake = self._format_losses(
                        'train_d_fake_', d_loss_fake, self.discriminator.model.metrics_names
                    )
                    d_loss_real = self._format_losses(
                        'train_d_real_', d_loss_real, self.discriminator.model.metrics_names
                    )
                    training_losses.update(d_loss_real)
                    training_losses.update(d_loss_fake)
                    y_train.append(valid)

                ## Generator training
                if self.feature_extractor:
                    hr_feats = self.feature_extractor.model.predict(batch['hr'])
                    y_train.extend([*hr_feats])

                model_losses = self.model.train_on_batch(batch['lr'], y_train)
                model_losses = self._format_losses('train_', model_losses, self.model.metrics_names)
                training_losses.update(model_losses)

                self.tensorboard.on_epoch_end(epoch * steps_per_epoch + step, training_losses)
                self.logger.debug('Losses at step {s}:\n {l}'.format(s=step, l=training_losses))

            elapsed_time = time() - epoch_start
            self.logger.info('Epoch {} took {:10.1f}s'.format(epoch, elapsed_time))

            validation_losses = self.model.evaluate(
                validation_set['lr'], y_validation, batch_size=batch_size
            )
            validation_losses = self._format_losses(
                'val_', validation_losses, self.model.metrics_names
            )

            if epoch == starting_epoch:
                remove_metrics = []
                for metric in monitored_metrics:
                    if (metric not in training_losses) and (metric not in validation_losses):
                        msg = ' '.join([metric, 'is NOT among the model metrics, removing it.'])
                        self.logger.error(msg)
                        remove_metrics.append(metric)
                for metric in remove_metrics:
                    _ = monitored_metrics.pop(metric)

            # should average train metrics
            end_losses = {}
            end_losses.update(validation_losses)
            end_losses.update(training_losses)

            self.helper.on_epoch_end(
                epoch=epoch,
                losses=end_losses,
                generator=self.model.get_layer('generator'),
                discriminator=self.discriminator,
                metrics=monitored_metrics,
            )
            self.tensorboard.on_epoch_end(epoch, validation_losses)
        self.tensorboard.on_train_end(None)
        
        
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19

class Cut_VGG19:
    """
    Class object that fetches keras' VGG19 model trained on the imagenet dataset
    and declares <layers_to_extract> as output layers. Used as feature extractor
    for the perceptual loss function.

    Args:
        layers_to_extract: list of layers to be declared as output layers.
        patch_size: integer, defines the size of the input (patch_size x patch_size).

    Attributes:
        loss_model: multi-output vgg architecture with <layers_to_extract> as output layers.
    """
    
    def __init__(self, patch_size, layers_to_extract):
        self.patch_size = patch_size
        self.input_shape = (patch_size,) * 2 + (3,)
        self.layers_to_extract = layers_to_extract
        self.logger = get_logger(__name__)
        
        if len(self.layers_to_extract) > 0:
            self._cut_vgg()
        else:
            self.logger.error('Invalid VGG instantiation: extracted layer must be > 0')
            raise ValueError('Invalid VGG instantiation: extracted layer must be > 0')
    
    def _cut_vgg(self):
        """
        Loads pre-trained VGG, declares as output the intermediate
        layers selected by self.layers_to_extract.
        """
        
        vgg = VGG19(weights='imagenet', include_top=False, input_shape=self.input_shape)
        vgg.trainable = False
        outputs = [vgg.layers[i].output for i in self.layers_to_extract]
        self.model = Model([vgg.input], outputs)
        self.model._name = 'feature_extractor'
        self.name = 'vgg19'  # used in weights naming
        
        


from tensorflow.keras.layers import Input, Activation, Dense, Conv2D, BatchNormalization, \
    LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

class Discriminator:
    """
    Implementation of the discriminator network for the adversarial
    component of the perceptual loss.

    Args:
        patch_size: integer, determines input size as (patch_size, patch_size, 3).
        kernel_size: size of the kernel in the conv blocks.

    Attributes:
        model: Keras model.
        name: name used to identify what discriminator is used during GANs training.
        model._name: identifies this network as the discriminator network
            in the compound model built by the trainer class.
        block_param: dictionary, determines the number of filters and the strides for each
            conv block.

    """
    
    def __init__(self, patch_size, kernel_size=3):
        self.patch_size = patch_size
        self.kernel_size = kernel_size
        self.block_param = {}
        self.block_param['filters'] = (64, 128, 128, 256, 256, 512, 512)
        self.block_param['strides'] = (2, 1, 2, 1, 1, 1, 1)
        self.block_num = len(self.block_param['filters'])
        self.model = self._build_disciminator()
        optimizer = Adam(0.0002, 0.5)
        self.model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        self.model._name = 'discriminator'
        self.name = 'srgan-large'
    
    def _conv_block(self, input, filters, strides, batch_norm=True, count=None):
        """ Convolutional layer + Leaky ReLU + conditional BN. """
        
        x = Conv2D(
            filters,
            kernel_size=self.kernel_size,
            strides=strides,
            padding='same',
            name='Conv_{}'.format(count),
        )(input)
        x = LeakyReLU(alpha=0.2)(x)
        if batch_norm:
            x = BatchNormalization(momentum=0.8)(x)
        return x
    
    def _build_disciminator(self):
        """ Puts the discriminator's layers together. """
        
        HR = Input(shape=(self.patch_size, self.patch_size, 3))
        x = self._conv_block(HR, filters=64, strides=1, batch_norm=False, count=1)
        for i in range(self.block_num):
            x = self._conv_block(
                x,
                filters=self.block_param['filters'][i],
                strides=self.block_param['strides'][i],
                count=i + 2,
            )
        x = Dense(self.block_param['filters'][-1] * 2, name='Dense_1024')(x)
        x = LeakyReLU(alpha=0.2)(x)
        # x = Flatten()(x)
        x = Dense(1, name='Dense_last')(x)
        HR_v_SR = Activation('sigmoid')(x)
        
        discriminator = Model(inputs=HR, outputs=HR_v_SR)
        return discriminator
    
    
import numpy as np

class ImageModel:
    """ISR models parent class.

    Contains functions that are common across the super-scaling models.
    """
    
    def predict(self, input_image_array, by_patch_of_size=None, batch_size=10, padding_size=2):
        """
        Processes the image array into a suitable format
        and transforms the network output in a suitable image format.

        Args:
            input_image_array: input image array.
            by_patch_of_size: for large image inference. Splits the image into
                patches of the given size.
            padding_size: for large image inference. Padding between the patches.
                Increase the value if there is seamlines.
            batch_size: for large image inferce. Number of patches processed at a time.
                Keep low and increase by_patch_of_size instead.
        Returns:
            sr_img: image output.
        """
        
        if by_patch_of_size:
            lr_img = process_array(input_image_array, expand=False)
            patches, p_shape = split_image_into_overlapping_patches(
                lr_img, patch_size=by_patch_of_size, padding_size=padding_size
            )
            # return patches
            for i in range(0, len(patches), batch_size):
                batch = self.model.predict(patches[i: i + batch_size])
                if i == 0:
                    collect = batch
                else:
                    collect = np.append(collect, batch, axis=0)
            
            scale = self.scale
            padded_size_scaled = tuple(np.multiply(p_shape[0:2], scale)) + (3,)
            scaled_image_shape = tuple(np.multiply(input_image_array.shape[0:2], scale)) + (3,)
            sr_img = stich_together(
                collect,
                padded_image_shape=padded_size_scaled,
                target_shape=scaled_image_shape,
                padding_size=padding_size * scale,
            )
        
        else:
            lr_img = process_array(input_image_array)
            sr_img = self.model.predict(lr_img)[0]
        
        sr_img = process_output(sr_img)
        return sr_img

WEIGHTS_URLS = {
    'gans': {
        'arch_params': {'C': 4, 'D': 3, 'G': 32, 'G0': 32, 'x': 4, 'T': 10},
        'url': 'https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/ISR/rrdn-C4-D3-G32-G032-T10-x4-GANS/rrdn-C4-D3-G32-G032-T10-x4_epoch299.hdf5',
        'name': 'rrdn-C4-D3-G32-G032-T10-x4_epoch299.hdf5',
    },
}


def make_model(arch_params, patch_size):
    """ Returns the model.

    Used to select the model.
    """
    
    return RRDN(arch_params, patch_size)


def get_network(weights):
    if weights in WEIGHTS_URLS.keys():
        arch_params = WEIGHTS_URLS[weights]['arch_params']
        url = WEIGHTS_URLS[weights]['url']
        name = WEIGHTS_URLS[weights]['name']
    else:
        raise ValueError('Available RRDN network weights: {}'.format(list(WEIGHTS_URLS.keys())))
    c_dim = 3
    kernel_size = 3
    return arch_params, c_dim, kernel_size, url, name


class RRDN(ImageModel):
    """Implementation of the Residual in Residual Dense Network for image super-scaling.

    The network is the one described in https://arxiv.org/abs/1809.00219 (Wang et al. 2018).

    Args:
        arch_params: dictionary, contains the network parameters C, D, G, G0, T, x.
        patch_size: integer or None, determines the input size. Only needed at
            training time, for prediction is set to None.
        beta: float <= 1, scaling parameter for the residual connections.
        c_dim: integer, number of channels of the input image.
        kernel_size: integer, common kernel size for convolutions.
        upscaling: string, 'ups' or 'shuffle', determines which implementation
            of the upscaling layer to use.
        init_val: extreme values for the RandomUniform initializer.
        weights: string, if not empty, download and load pre-trained weights.
            Overrides other parameters.

    Attributes:
        C: integer, number of conv layer inside each residual dense blocks (RDB).
        D: integer, number of RDBs inside each Residual in Residual Dense Block (RRDB).
        T: integer, number or RRDBs.
        G: integer, number of convolution output filters inside the RDBs.
        G0: integer, number of output filters of each RDB.
        x: integer, the scaling factor.
        model: Keras model of the RRDN.
        name: name used to identify what upscaling network is used during training.
        model._name: identifies this network as the generator network
            in the compound model built by the trainer class.
    """
    
    def __init__(
            self, arch_params={}, patch_size=None, beta=0.2, c_dim=3, kernel_size=3, init_val=0.05, weights=''
    ):
        if weights:
            arch_params, c_dim, kernel_size, url, fname = get_network(weights)
        
        self.params = arch_params
        self.beta = beta
        self.c_dim = c_dim
        self.C = self.params['C']
        self.D = self.params['D']
        self.G = self.params['G']
        self.G0 = self.params['G0']
        self.T = self.params['T']
        self.scale = self.params['x']
        self.initializer = RandomUniform(minval=-init_val, maxval=init_val, seed=None)
        self.kernel_size = kernel_size
        self.patch_size = patch_size
        self.model = self._build_rdn()
        self.model._name = 'generator'
        self.name = 'rrdn'
        if weights:
            weights_path = tf.keras.utils.get_file(fname=fname, origin=url)
            self.model.load_weights(weights_path)
    
    def _dense_block(self, input_layer, d, t):
        """
        Implementation of the (Residual) Dense Block as in the paper
        Residual Dense Network for Image Super-Resolution (Zhang et al. 2018).

        Residuals are incorporated in the RRDB.
        d is an integer only used for naming. (d-th block)
        """
        
        x = input_layer
        for c in range(1, self.C + 1):
            F_dc = Conv2D(
                self.G,
                kernel_size=self.kernel_size,
                padding='same',
                kernel_initializer=self.initializer,
                name='F_%d_%d_%d' % (t, d, c),
            )(x)
            F_dc = Activation('relu', name='F_%d_%d_%d_Relu' % (t, d, c))(F_dc)
            x = concatenate([x, F_dc], axis=3, name='RDB_Concat_%d_%d_%d' % (t, d, c))
        
        # DIFFERENCE: in RDN a kernel size of 1 instead of 3 is used here
        x = Conv2D(
            self.G0,
            kernel_size=3,
            padding='same',
            kernel_initializer=self.initializer,
            name='LFF_%d_%d' % (t, d),
        )(x)
        return x
    
    def _RRDB(self, input_layer, t):
        """Residual in Residual Dense Block.

        t is integer, for naming of RRDB.
        beta is scalar.
        """
        
        # SUGGESTION: MAKE BETA LEARNABLE
        x = input_layer
        for d in range(1, self.D + 1):
            LFF = self._dense_block(x, d, t)
            LFF_beta = MultiplyBeta(self.beta)(LFF)
            x = Add(name='LRL_%d_%d' % (t, d))([x, LFF_beta])
        x = MultiplyBeta(self.beta)(x)
        x = Add(name='RRDB_%d_out' % (t))([input_layer, x])
        return x
    
    def _pixel_shuffle(self, input_layer):
        """ PixelShuffle implementation of the upscaling part. """
        x = Conv2D(
            self.c_dim * self.scale ** 2,
            kernel_size=3,
            padding='same',
            kernel_initializer=self.initializer,
            name='PreShuffle',
        )(input_layer)

        return PixelShuffle(self.scale)(x)
    
    def _build_rdn(self):
        LR_input = Input(shape=(self.patch_size, self.patch_size, 3), name='LR_input')
        pre_blocks = Conv2D(
            self.G0,
            kernel_size=self.kernel_size,
            padding='same',
            kernel_initializer=self.initializer,
            name='Pre_blocks_conv',
        )(LR_input)
        # DIFFERENCE: in RDN an extra convolution is present here
        for t in range(1, self.T + 1):
            if t == 1:
                x = self._RRDB(pre_blocks, t)
            else:
                x = self._RRDB(x, t)
        # DIFFERENCE: in RDN a conv with kernel size of 1 after a concat operation is used here
        post_blocks = Conv2D(
            self.G0,
            kernel_size=3,
            padding='same',
            kernel_initializer=self.initializer,
            name='post_blocks_conv',
        )(x)
        # Global Residual Learning
        GRL = Add(name='GRL')([post_blocks, pre_blocks])
        # Upscaling
        PS = self._pixel_shuffle(GRL)
        # Compose SR image
        SR = Conv2D(
            self.c_dim,
            kernel_size=self.kernel_size,
            padding='same',
            kernel_initializer=self.initializer,
            name='SR',
        )(PS)
        return Model(inputs=LR_input, outputs=SR)

class PixelShuffle(tf.keras.layers.Layer):
    def __init__(self, scale, *args, **kwargs):
        super(PixelShuffle, self).__init__(*args, **kwargs)
        self.scale = scale

    def call(self, x):
        return tf.nn.depth_to_space(x, block_size=self.scale, data_format='NHWC')

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'scale': self.scale,
        })
        return config

class MultiplyBeta(tf.keras.layers.Layer):
    def __init__(self, beta, *args, **kwargs):
        super(MultiplyBeta, self).__init__(*args, **kwargs)
        self.beta = beta

    def call(self, x, **kwargs):
        return x * self.beta

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'beta': self.beta,
        })
        return config
    
lr_train_patch_size = 40
layers_to_extract = [5, 9]
scale = 2
hr_train_patch_size = lr_train_patch_size * scale

rrdn  = RRDN(arch_params={'C':4, 'D':3, 'G':64, 'G0':64, 'T':10, 'x':scale}, patch_size=lr_train_patch_size)
f_ext = Cut_VGG19(patch_size=hr_train_patch_size, layers_to_extract=layers_to_extract)
discr = Discriminator(patch_size=hr_train_patch_size, kernel_size=3)

wget.download('http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X2.zip')
wget.download('http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X2.zip')
wget.download('http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip')
wget.download('http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip')

archive_train_LR = 'DIV2K_train_LR_bicubic_X2.zip'
with zipfile.ZipFile(archive_train_LR, 'r') as zip_file:
    zip_file.extractall()

archive_valid_LR = 'DIV2K_valid_LR_bicubic_X2.zip'
with zipfile.ZipFile(archive_valid_LR, 'r') as zip_file:
    zip_file.extractall()
    
archive_train_HR = 'DIV2K_train_HR.zip'
with zipfile.ZipFile(archive_train_HR, 'r') as zip_file:
    zip_file.extractall()

archive_valid_HR = 'DIV2K_valid_HR.zip'
with zipfile.ZipFile(archive_valid_HR, 'r') as zip_file:
    zip_file.extractall()
    
loss_weights = {
  'generator': 0.0,
  'feature_extractor': 0.0833,
  'discriminator': 0.01
}
losses = {
  'generator': 'mae',
  'feature_extractor': 'mse',
  'discriminator': 'binary_crossentropy'
}

log_dirs = {'logs': './logs', 'weights': './weights'}

learning_rate = {'initial_value': 0.0004, 'decay_factor': 0.5, 'decay_frequency': 30}

flatness = {'min': 0.0, 'max': 0.15, 'increase': 0.01, 'increase_frequency': 5}

trainer = Trainer(
    generator=rrdn,
    discriminator=discr,
    feature_extractor=f_ext,
    lr_train_dir='DIV2K_train_LR_bicubic/X2',
    hr_train_dir='DIV2K_train_HR',
    lr_valid_dir='DIV2K_valid_LR_bicubic/X2',
    hr_valid_dir='DIV2K_valid_HR',
    loss_weights=loss_weights,
    learning_rate=learning_rate,
    flatness=flatness,
    dataname='image_dataset',
    log_dirs=log_dirs,
    weights_generator=None,
    weights_discriminator=None,
    n_validation=40,
)

trainer.train(
    epochs=4,
    steps_per_epoch=20,
    batch_size=16,
    monitored_metrics={'val_PSNR_Y': 'max'}
)


rrdn.save_weights('rrdn_weights.h5')

import yadisk
y = yadisk.YaDisk(token="y0_AgAAAAAZdSRIAAnWpQAAAADiIR-G69xDHp3vSUKGjYeHSNjcH6B_kQw")
# Сохранение весов
rrdn.save_weights(y.upload('rrdn_weights.h5', '/weights_dir/rrdn_weights.h5'))

# Коннектор для работы с популярными датасетами

Коннектор может быть использован для подключения к датасетам, предназначенным для:
- задачи локализации (на данный момент реализовано полнофункциональное подключение
к датасетам в форматах Dota и PASCALVOC. В тестовом режиме работает подключение к датасету Onera);
- задачи поиска структурных изменений на разновременных изображениях (на данный момент реализован базовый
функционал к датасету в формате CD2014).


Объект ```LocalisationDatasetConnector``` позволяет:
- осуществлять выборку данных заданного набора классов;
- выбирать данные для заданного подмножества изображений;
- фильтровать данные разметки по размеру изображений;
- конвертировать данные в формат разметки COCO;
- вычислять описательную статистику для любого среза данных;
- визуализировать срез данных для заданного изображения.

Объект ```ChangeDetectionDatasetConnector``` позволяет:
- выбирать данные для заданного подмножества изображений;
- вычислять описательную статистику для любого среза данных;
- визуализировать срез данных для заданной группы.

#### 1. Установка
``` python setup.py install```

#### 2. Пример. Подключение к датасету в формате Dota

```python
import os.path as path
import matplotlib.pyplot as plt

from connector.dota import DotaDatasetConnector

# Указываем путь до данных на файловом ресурсе
root_dir = '//f125.sils.local/doc/PROJECTS/ML/Data/Dota/train'
conn = DotaDatasetConnector.connect(imagedir=path.join(root_dir, 'images'),
                                    labeldir=path.join(root_dir, 'labelTxt'))

# Указываем критерии формирования среза данных
# 1. Выбрать объекты класса ['plane', 'roundabout', 'storage-tank']
# 2. Выбрать объекты размерами больше 10px по ширине и высоте

labels = ['plane', 'roundabout', 'storage-tank']
data = conn.select_labels(labels=labels).select_objects_with_size(wlim=(10, None), hlim=(10, None))

# Выводим в консоль описательную статистику для среза данных 
data.describe(filename=None)

# Наносим разметку на изображение P0178.png и визуализируем результат
result = data.draw_image_annotation(image_idx='P0178.png')

plt.imshow(result)
plt.show()
```
![img](doc/dota.png)

#### 3. Пример. Подключение к датасету в формате PASCAL VOC.

```python
from connector.pascalvoc import PASCALVOCDatasetConnector

# Указываем путь до данных на файловом ресурсе
root_dir = '//f125.sils.local/doc/PROJECTS/ML/Data/bpla/20190809'
new_dataset_dir = '//f125.sils.local/doc/PROJECTS/ML/Data/bpla/20190809/custom'
conn = PASCALVOCDatasetConnector.connect(imagedir=root_dir,
                                         labeldir=root_dir)

# Указываем критерии формирования среза данных

labels = ['building materials', 'pole']
data = conn.select_labels(labels=labels).select_objects_with_size(wlim=(10, None), hlim=(10, None))

# Выводим в консоль описательную статистику для среза данных 
data.describe(filename=None)

# Нарезаем датасет на изображения размерами (512х512), и сохраняем их в директорию new_dataset_dir
data.convert(roi_shape=(512, 512), imagedir=new_dataset_dir, labeldir=new_dataset_dir, rel_shift=0.5)

# Подключаемся к сконвертированному датасету
converted_dataset = PASCALVOCDatasetConnector.connect(imagedir=new_dataset_dir, labeldir=new_dataset_dir)

# Вычисляем описательные статистики методом бутстрапа и сохраняем в файл ./stat.coeffs
converted_dataset.calculate_stat_coeffs(n_bootstrap=100, filename="stat.coeffs")

# Конвертируем данные в формат COCO(json-like) и сохраняем в файл ./coco_format.json.
converted_dataset.create_coco_format_annotation(jsonfile="coco_format.json")

```
#### 4. Пример. Подключение к датасету CD2014.

```python
import os.path as path
from connector.cd2014 import CDDatasetConnector

root_dir = '//f125.sils.local/doc/PROJECTS/ML/Data/change_detection/cd2014'
conn = CDDatasetConnector.connect(root_dir=root_dir,
                                  filename=path.join(root_dir, "train.txt"))

cluster_video = ["PTZ", "nightVideos", "cameraJitter"]
data1 = conn.select_cluster(cluster_video)

group_video = ["skating"]
data2 = conn.select_group(group_video)

data = data1 + data2

data.describe()

print(data.calculate_stat_coeffs(n_bootsrap=100))
```




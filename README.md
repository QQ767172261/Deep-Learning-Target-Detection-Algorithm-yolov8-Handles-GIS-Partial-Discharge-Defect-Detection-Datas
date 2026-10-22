如何使用深度学习目标检测算法yolov8训练  处理GIS局部放电缺陷检测数据集 建立基于深度学习YOLOv8GIS局部放电缺陷检测的模型
@[toc]
以下文字及代码可供参考。

 GIS局部放电缺陷检测数据集4243+VOC标注
 
 自由粒子 绝缘放电  电晕放电  自由粒子 .
 电晕放电、悬浮放电、自由粒子、绝缘放电四类，同学可用于电气工程专业在计算机视觉中的深度学习目标检测图像检测识别等。
 
 1 自由粒子 ![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/83f33c5a1ca54fc1821de2f426322bee.jpeg)
2绝缘放电
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4f4a54424be24ca38f92b3f980866ca1.jpeg)
3电晕放电
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/83177282833c4242ab65cad3cae8f9a0.jpeg)
4自由粒子
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/52199f1c9491405ab0f672b3e974bf50.jpeg)
对于电气工程中的GIS局部放电缺陷检测，选择深度学习模型进行目标检测。（数据集4243张图片和VOC格式的标签，类别包括自由粒子、绝缘放电、电晕放电、悬浮放电），YOLOv8是一个非常合适的选择。它在速度和准确性方面表现出色，并且易于使用和部署。

使用YOLOv8处理你的数据集的基本步骤：
仅供参考。
### 1. 数据准备

首先，确保你的数据集已经按照Pascal VOC格式标注完成。如果需要将这些标注转换为YOLO格式，请参考之前提供的代码示例。

### 2. 安装依赖

安装YOLOv8及其他必要的库：
```bash
pip install ultralytics opencv-python
```

### 3. 配置YOLOv8

创建一个`dataset.yaml`文件来配置你的数据集路径和类别信息：
```yaml
train: path/to/train/images
val: path/to/val/images

nc: 4  # number of classes
names: ['free_particle', 'insulation_discharge', 'corona_discharge', 'floating_discharge']  # class names
```
请根据你的实际目录结构更新`path/to/train/images`和`path/to/val/images`。

### 4. 训练模型

使用YOLOv8进行训练：
```python
from ultralytics import YOLO

# Load a pretrained model
model = YOLO('yolov8n.pt')  # You can choose other models like yolov8s, yolov8m, etc.

# Train the model
results = model.train(data='path/to/dataset.yaml', epochs=50, imgsz=640)
```
你可以调整`epochs`和`imgsz`参数以适应你的需求。更多关于训练参数的信息可以参考YOLOv8的官方文档。

### 5. 模型评估

在验证集上评估模型性能：
```python
metrics = model.val()
print(metrics)
```
这将输出模型在验证集上的性能指标，如mAP（mean Average Precision）等。

### 6. 使用模型进行预测

训练完成后，你可以使用该模型对新的图像进行预测：
```python
# Load your trained model
model = YOLO('path/to/best.pt')

# Predict on an image
results = model('path/to/image.jpg')
```
结果将包含检测到的对象及其位置信息。

以下是针对GIS局部放电缺陷检测数据集的完整流程，包括数据集准备、预处理、安装依赖、配置YOLOv8、训练评估、构建GUI应用程序、读取图像、显示图像以及运行代码。我们将分步骤详细说明并附上完整代码。

---

### **1. 数据集准备**

你的数据集包含4243张图片和VOC格式的标签，类别包括：

- 自由粒子 (`free_particle`)
- 绝缘放电 (`insulation_discharge`)
- 电晕放电 (`corona_discharge`)
- 悬浮放电 (`floating_discharge`)

确保数据集目录结构如下：
```
dataset/
├── images/
│   ├── train/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   └── val/
│       ├── img1001.jpg
│       ├── img1002.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── img1.xml
    │   ├── img2.xml
    │   └── ...
    └── val/
        ├── img1001.xml
        ├── img1002.xml
        └── ...
```

---

### **2. 数据集预处理**

将Pascal VOC格式的XML标签转换为YOLO格式（txt文件）。

#### 转换脚本
```python
import os
import xml.etree.ElementTree as ET

def convert_voc_to_yolo(xml_file, output_dir, class_mapping):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    size = root.find('size')
    img_w = int(size.find('width').text)
    img_h = int(size.find('height').text)

    yolo_lines = []
    for obj in root.findall('object'):
        cls_name = obj.find('name').text
        if cls_name not in class_mapping:
            continue
        cls_id = class_mapping[cls_name]

        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)

        x_center = (xmin + xmax) / 2.0 / img_w
        y_center = (ymin + ymax) / 2.0 / img_h
        width = (xmax - xmin) / img_w
        height = (ymax - ymin) / img_h

        yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    filename = os.path.splitext(os.path.basename(xml_file))[0]
    with open(os.path.join(output_dir, f"{filename}.txt"), 'w') as f:
        f.write("\n".join(yolo_lines))

# 示例用法
xml_dir = "path/to/xml"
output_dir = "path/to/yolo_labels"
class_mapping = {
    "free_particle": 0,
    "insulation_discharge": 1,
    "corona_discharge": 2,
    "floating_discharge": 3
}

os.makedirs(output_dir, exist_ok=True)
for xml_file in os.listdir(xml_dir):
    if xml_file.endswith(".xml"):
        convert_voc_to_yolo(os.path.join(xml_dir, xml_file), output_dir, class_mapping)
```

---

### **3. 安装依赖**

确保安装以下库：
```bash
pip install ultralytics opencv-python PyQt5
```

---

### **4. 配置YOLOv8**

创建一个`dataset.yaml`文件，内容如下：
```yaml
train: path/to/train/images
val: path/to/val/images

nc: 4  # 类别数量
names: ['free_particle', 'insulation_discharge', 'corona_discharge', 'floating_discharge']
```

---

### **5. 训练模型**

使用YOLOv8进行训练：
```python
from ultralytics import YOLO

# 加载预训练模型
model = YOLO('yolov8n.pt')  # 或选择其他模型如 yolov8s, yolov8m 等

# 开始训练
results = model.train(data='path/to/dataset.yaml', epochs=50, imgsz=640)
```

---

### **6. 模型评估**

在验证集上评估模型性能：
```python
metrics = model.val()
print(metrics)
```

---

### **7. 构建GUI应用程序**

创建一个简单的GUI应用程序来加载图像、显示检测结果。

#### GUI代码
```python
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from ultralytics import YOLO
import cv2

class DetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.model = YOLO("path/to/best.pt")  # 加载训练好的模型

    def initUI(self):
        self.setWindowTitle("GIS 局部放电检测")

        self.label = QLabel(self)
        self.label.setText("选择一张图像进行检测。")

        self.btn_load = QPushButton("加载图像", self)
        self.btn_load.clicked.connect(self.load_image)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.btn_load)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图像", "", "Images (*.png *.jpg *.jpeg)", options=options)
        if file_path:
            self.detect_objects(file_path)

    def detect_objects(self, image_path):
        img = cv2.imread(image_path)
        results = self.model(img)

        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        height, width, channel = img.shape
        bytes_per_line = 3 * width
        q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.label.setPixmap(QPixmap.fromImage(q_img))
        self.label.adjustSize()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DetectionApp()
    window.show()
    sys.exit(app.exec_())
```

---

### **8. 运行代码**

将上述代码保存为独立的Python文件，并运行：
```bash
python gui_app.py
```

---

### **总结**

通过以上步骤，同学你已经完成了从数据准备到模型训练、评估，再到构建GUI应用程序的完整流程。同学你可以根据具体需求调整代码中的路径、参数等。

仅供参考，


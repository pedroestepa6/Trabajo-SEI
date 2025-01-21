"""
Pasos para entrenar modelo de reconocimiento con Intel RealSense

Descripción:
Este script captura imágenes RGB y datos de profundidad utilizando la cámara Intel RealSense
y define un modelo de red neuronal para realizar reconocimiento facial.

Autor: Pedro Estepa

"""

##CAPTURA DE DATOS RGB y de profundidad
import pyrealsense2 as rs
import numpy as np
import cv2

# Configuración de la cámara RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Captura de un par de frames
frames = pipeline.wait_for_frames()
depth_frame = frames.get_depth_frame()
color_frame = frames.get_color_frame()

# Convertir las imágenes a arrays de numpy
depth_image = np.asanyarray(depth_frame.get_data())
color_image = np.asanyarray(color_frame.get_data())

##GENERACIÓN DE LA NUBE DE PUNTOS

pc = rs.pointcloud()
points = pc.calculate(depth_frame)
pc.map_to(color_frame)

# Obtener los vértices y las coordenadas de textura
v = points.get_vertices()
t = points.get_texture_coordinates()
verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv

##PREPROCESAMIENTO DE LOS DATOS

# Normalizar las coordenadas de los vértices
verts = (verts - np.mean(verts, axis=0)) / np.std(verts, axis=0)

# Redimensionar las imágenes RGB y de profundidad
color_image_resized = cv2.resize(color_image, (128, 128))
depth_image_resized = cv2.resize(depth_image, (128, 128))

# Normalizar las imágenes
color_image_normalized = color_image_resized / 255.0
depth_image_normalized = depth_image_resized / 255.0

##Definición del modelo
import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetRGBD(nn.Module):
    def __init__(self, num_classes):
        super(PointNetRGBD, self).__init__()
        # Red de nubes de puntos
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        
        # Red de procesamiento de imágenes RGB
        self.rgb_conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.rgb_conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.rgb_fc1 = nn.Linear(64 * 32 * 32, 512)
        
        # Red de procesamiento de imágenes de profundidad
        self.depth_conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.depth_conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.depth_fc1 = nn.Linear(64 * 32 * 32, 512)
        
        # Combinación de ambas entradas
        self.fc3 = nn.Linear(512 + 512 + 256, num_classes)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x, rgb, depth):
        # Procesamiento de nubes de puntos
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        
        # Procesamiento de imágenes RGB
        rgb = F.relu(self.rgb_conv1(rgb))
        rgb = F.max_pool2d(F.relu(self.rgb_conv2(rgb)), 2)
        rgb = rgb.view(rgb.size(0), -1)
        rgb = F.relu(self.rgb_fc1(rgb))
        
        # Procesamiento de imágenes de profundidad
        depth = F.relu(self.depth_conv1(depth))
        depth = F.max_pool2d(F.relu(self.depth_conv2(depth)), 2)
        depth = depth.view(depth.size(0), -1)
        depth = F.relu(self.depth_fc1(depth))
        
        # Combinación de ambas entradas
        combined = torch.cat([x, rgb, depth], dim=1)
        combined = self.dropout(combined)
        out = self.fc3(combined)
        return out

# Ejemplo de uso
num_classes = 10
model = PointNetRGBD(num_classes)
input_points = torch.rand(32, 3, 1024)  # Batch de 32 nubes de puntos, cada una con 1024 puntos
input_rgb = torch.rand(32, 3, 128, 128)  # Batch de 32 imágenes RGB
input_depth = torch.rand(32, 1, 128, 128)  # Batch de 32 imágenes de profundidad
output = model(input_points, input_rgb, input_depth)
print(output.shape)  # Salida: [32, num_classes]

##Entrenamiento del modelo

# Definir el conjunto de datos y el optimizador
train_points, train_rgb, train_depth, train_labels = load_data()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Entrenar el modelo
for epoch in range(10):
    model.train()
    optimizer.zero_grad()
    output = model(train_points, train_rgb, train_depth)
    loss = F.cross_entropy(output, train_labels)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
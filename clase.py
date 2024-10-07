import sys
import os
import laspy
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, 
                             QPushButton, QProgressBar, QMessageBox, QFileDialog, 
                             QScrollArea, QCheckBox, QHBoxLayout, QSlider)
from PyQt5.QtCore import Qt

class LASClassificationViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Suriyaco Valley - Clasificación y Medición de Puntos LAS")
        self.setGeometry(100, 100, 1600, 1000)

        # Configuración de la interfaz
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Etiqueta principal
        self.label = QLabel("Haz clic en 'Cargar Archivos LAS' para iniciar...")
        self.label.setStyleSheet("font-size: 18px; padding: 10px; font-weight: bold; color: #333;")
        self.layout.addWidget(self.label)

        # Visualizador PyVista
        self.pv_widget = QtInteractor(self)
        self.layout.addWidget(self.pv_widget.interactor)

        # Barra de progreso
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                height: 20px;
                text-align: center;
                font-size: 14px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
        """)
        self.layout.addWidget(self.progress_bar)

        # Botones y controles
        button_layout = QHBoxLayout()
        
        self.load_button = QPushButton("Cargar Archivos LAS")
        self.load_button.clicked.connect(self.load_las_files)
        self.load_button.setStyleSheet("background-color: #007bff; color: white; padding: 8px; border-radius: 4px; font-size: 14px;")
        button_layout.addWidget(self.load_button)

        self.start_button = QPushButton("Iniciar Clasificación")
        self.start_button.clicked.connect(self.start_classification)
        self.start_button.setEnabled(False)
        self.start_button.setStyleSheet("background-color: #28a745; color: white; padding: 8px; border-radius: 4px; font-size: 14px;")
        button_layout.addWidget(self.start_button)

        self.point_size_slider = QSlider(Qt.Horizontal)
        self.point_size_slider.setMinimum(1)
        self.point_size_slider.setMaximum(10)
        self.point_size_slider.setValue(5)
        self.point_size_slider.valueChanged.connect(self.update_point_size)
        button_layout.addWidget(QLabel("Tamaño del Punto:"))
        button_layout.addWidget(self.point_size_slider)
        
        self.layout.addLayout(button_layout)

        # Sección de capas con scroll
        self.layer_scroll = QScrollArea()
        self.layer_widget = QWidget()
        self.layer_layout = QVBoxLayout(self.layer_widget)
        self.layer_scroll.setWidgetResizable(True)
        self.layer_scroll.setWidget(self.layer_widget)
        self.layer_scroll.setFixedHeight(150)
        self.layer_scroll.setStyleSheet("border: 1px solid #ccc; background-color: #ffffff; margin: 5px; padding: 5px;")
        self.layout.addWidget(self.layer_scroll)

        # Variables para manejo de datos
        self.files_data = []
        self.meshes = {}
        self.current_point_size = 5
        self.point_pairs = []

        # Etiqueta para mostrar distancia y diferencia de alturas
        self.distance_label = QLabel("Distancia: N/A | Diferencia de Altura: N/A")
        self.distance_label.setStyleSheet("font-size: 14px; padding: 10px;")
        self.layout.addWidget(self.distance_label)

        # Habilitar la funcionalidad de selección de puntos
        self.pv_widget.enable_point_picking(callback=self.on_point_picked, show_message=False)

    def load_las_files(self):
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Seleccione archivos LAS", "", "LAS Files (*.las)")
        if file_paths:
            self.progress_bar.setValue(0)
            total_files = len(file_paths)
            for i, file_path in enumerate(file_paths):
                try:
                    las_data = laspy.read(file_path)
                    self.files_data.append({"file_path": file_path, "las_data": las_data})
                    layer_widget = QWidget()
                    layer_layout = QHBoxLayout(layer_widget)
                    layer_checkbox = QCheckBox(os.path.basename(file_path))
                    layer_checkbox.setChecked(True)
                    layer_checkbox.stateChanged.connect(self.update_visualization)
                    layer_layout.addWidget(layer_checkbox)
                    delete_button = QPushButton("Eliminar")
                    delete_button.clicked.connect(lambda _, cb=layer_checkbox: self.remove_layer(cb))
                    delete_button.setFixedSize(60, 25)
                    layer_layout.addWidget(delete_button)
                    self.layer_layout.addWidget(layer_widget)
                    self.meshes[layer_checkbox.text()] = self.create_pyvista_mesh(las_data)
                    self.start_button.setEnabled(True)
                    progress_percentage = ((i + 1) / total_files) * 100
                    self.progress_bar.setValue(int(progress_percentage))
                    QApplication.processEvents()
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"No se pudo cargar el archivo: {e}")
            self.update_visualization()

    def create_pyvista_mesh(self, las_data):
        points = np.vstack((las_data.x, las_data.y, las_data.z)).T
        point_cloud = pv.PolyData(points)
        z_min, z_max = points[:, 2].min(), points[:, 2].max()
        z_range = z_max - z_min
        colors = np.zeros((len(points), 3))
        normalized_z = (points[:, 2] - z_min) / z_range
        colors[:, 0] = np.clip(2 * (normalized_z - 0.5), 0, 1)
        colors[:, 1] = np.clip(2 * (0.5 - np.abs(normalized_z - 0.5)), 0, 1)
        colors[:, 2] = np.clip(2 * (0.5 - normalized_z), 0, 1)
        point_cloud['RGB'] = colors
        return point_cloud

    def update_visualization(self):
        self.pv_widget.clear()
        for i in range(self.layer_layout.count()):
            layer_widget = self.layer_layout.itemAt(i).widget()
            checkbox = layer_widget.layout().itemAt(0).widget()
            if checkbox.isChecked():
                mesh = self.meshes.get(checkbox.text())
                if mesh:
                    self.pv_widget.add_mesh(mesh, scalars="RGB", rgb=True, point_size=self.current_point_size)
        self.pv_widget.reset_camera()
        self.pv_widget.render()

    def remove_layer(self, checkbox):
        text = checkbox.text()
        if text in self.meshes:
            del self.meshes[text]
        checkbox.parentWidget().deleteLater()
        self.update_visualization()

    def start_classification(self):
        QMessageBox.information(self, "Información", "La clasificación de puntos por cota ya está aplicada en la visualización.")

    def update_point_size(self):
        self.current_point_size = self.point_size_slider.value()
        self.update_visualization()

    def on_point_picked(self, point):
        self.point_pairs.append(point)
        if len(self.point_pairs) == 2:
            self.calculate_distance_and_height_diff()
            self.show_measurement_summary()
            self.draw_measurement_line()
            self.point_pairs = []

    def calculate_distance_and_height_diff(self):
        point1, point2 = self.point_pairs
        distance = np.linalg.norm(np.array(point2) - np.array(point1))
        height_diff = abs(point2[2] - point1[2])
        self.distance_label.setText(f"Distancia: {distance:.2f} m | Diferencia de Altura: {height_diff:.2f} m")
        return distance, height_diff

    def draw_measurement_line(self):
        line = pv.Line(self.point_pairs[0], self.point_pairs[1])
        self.pv_widget.add_mesh(line, color='yellow', line_width=3)
        self.pv_widget.render()

    def show_measurement_summary(self):
        point1, point2 = self.point_pairs
        distance, height_diff = self.calculate_distance_and_height_diff()
        summary = (f"Coordenadas del Punto 1: {point1}\n"
                   f"Coordenadas del Punto 2: {point2}\n"
                   f"Distancia: {distance:.2f} m\n"
                   f"Diferencia de Altura: {height_diff:.2f} m")
        QMessageBox.information(self, "Resumen de Medición", summary)

def main():
    app = QApplication(sys.argv)
    viewer = LASClassificationViewer()
    viewer.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

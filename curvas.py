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
        self.initialize_ui()
        self.initialize_variables()
        self.show()

    def initialize_ui(self):
        """Inicializa la interfaz de usuario"""
        self.setWindowTitle("Suriyaco Valley - Clasificación y Medición de Puntos LAS")
        self.setGeometry(100, 100, 1600, 1000)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Etiqueta principal
        self.main_label = QLabel("Haz clic en 'Cargar Archivos LAS' para iniciar...")
        self.main_label.setStyleSheet("font-size: 18px; padding: 10px; font-weight: bold; color: #333;")
        self.layout.addWidget(self.main_label)

        # Visualizador PyVista
        self.pv_widget = QtInteractor(self)
        self.layout.addWidget(self.pv_widget.interactor)

        # Barra de progreso
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)
        self.layout.addWidget(self.progress_bar)

        # Configurar botones y controles
        self.setup_controls()

        # Sección de capas con scroll
        self.setup_layer_section()

        # Etiqueta para distancia y altura
        self.distance_label = QLabel("Distancia: N/A | Diferencia de Altura: N/A")
        self.distance_label.setStyleSheet("font-size: 14px; padding: 10px;")
        self.layout.addWidget(self.distance_label)

    def setup_controls(self):
        """Configura los controles y botones"""
        button_layout = QHBoxLayout()
        
        # Botón para cargar archivos LAS
        self.load_button = self.create_button("Cargar Archivos LAS", "#007bff", self.load_las_files)
        button_layout.addWidget(self.load_button)

        # Botón para guardar puntos seleccionados
        self.save_button = self.create_button("Guardar Puntos Seleccionados", "#17a2b8", self.save_selected_points, enabled=False)
        button_layout.addWidget(self.save_button)

        # Botón para dibujar curvas de nivel
        self.contour_button = self.create_button("Dibujar Curvas de Nivel", "#6f42c1", self.draw_contour_lines, enabled=False)
        button_layout.addWidget(self.contour_button)

        # Control para ajustar el tamaño del punto
        self.point_size_slider = QSlider(Qt.Horizontal)
        self.point_size_slider.setMinimum(1)
        self.point_size_slider.setMaximum(10)
        self.point_size_slider.setValue(5)
        self.point_size_slider.valueChanged.connect(self.update_point_size)
        button_layout.addWidget(QLabel("Tamaño del Punto:"))
        button_layout.addWidget(self.point_size_slider)

        self.layout.addLayout(button_layout)

    def setup_layer_section(self):
        """Configura la sección de capas"""
        self.layer_scroll = QScrollArea()
        self.layer_widget = QWidget()
        self.layer_layout = QVBoxLayout(self.layer_widget)
        self.layer_scroll.setWidgetResizable(True)
        self.layer_scroll.setWidget(self.layer_widget)
        self.layer_scroll.setFixedHeight(150)
        self.layer_scroll.setStyleSheet("border: 1px solid #ccc; background-color: #ffffff; margin: 5px; padding: 5px;")
        self.layout.addWidget(self.layer_scroll)

    def create_button(self, text, color, function, enabled=True):
        """Crea un botón estilizado"""
        button = QPushButton(text)
        button.clicked.connect(function)
        button.setEnabled(enabled)
        button.setStyleSheet(f"background-color: {color}; color: white; padding: 8px; border-radius: 4px; font-size: 14px;")
        return button

    def initialize_variables(self):
        """Inicializa las variables de la aplicación"""
        self.files_data = []
        self.meshes = {}
        self.current_point_size = 5
        self.selected_points = []
        self.area_selection = None
        self.pv_widget.enable_rubber_band_style()  # Habilitar la selección de área rectangular

    def update_point_size(self, value):
        """Actualiza el tamaño de los puntos en el visor"""
        self.current_point_size = value
        self.update_visualization()

    def load_las_files(self):
        """Carga archivos LAS y muestra la información"""
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Seleccione archivos LAS", "", "LAS Files (*.las)")
        if file_paths:
            self.progress_bar.setValue(0)
            total_files = len(file_paths)
            for i, file_path in enumerate(file_paths):
                try:
                    las_data = laspy.read(file_path)
                    self.files_data.append({"file_path": file_path, "las_data": las_data})
                    self.add_layer_widget(file_path, las_data)
                    self.save_button.setEnabled(True)
                    self.contour_button.setEnabled(True)
                    self.progress_bar.setValue(int(((i + 1) / total_files) * 100))
                    QApplication.processEvents()
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"No se pudo cargar el archivo: {e}")
            self.update_visualization()

    def add_layer_widget(self, file_path, las_data):
        """Añade un widget de capa para cada archivo LAS"""
        layer_widget = QWidget()
        layer_layout = QHBoxLayout(layer_widget)
        
        layer_checkbox = QCheckBox(os.path.basename(file_path))
        layer_checkbox.setChecked(True)
        layer_checkbox.stateChanged.connect(self.update_visualization)
        layer_layout.addWidget(layer_checkbox)

        delete_button = QPushButton("Eliminar")
        delete_button.clicked.connect(lambda _, cb=layer_checkbox: self.remove_layer(cb))
        delete_button.setFixedSize(70, 60)
        layer_layout.addWidget(delete_button)

        self.layer_layout.addWidget(layer_widget)
        self.meshes[layer_checkbox.text()] = self.create_pyvista_mesh(las_data)

    def create_pyvista_mesh(self, las_data):
        """Crea una malla PyVista a partir de los datos LAS"""
        points = np.vstack((las_data.x, las_data.y, las_data.z)).T
        if len(points) == 0:
            return None
        point_cloud = pv.PolyData(points)
        point_cloud['RGB'] = self.calculate_colors(points)
        return point_cloud

    def calculate_colors(self, points):
        """Calcula los colores RGB para los puntos en función de Z"""
        z_min, z_max = points[:, 2].min(), points[:, 2].max()
        z_range = z_max - z_min
        normalized_z = (points[:, 2] - z_min) / z_range
        colors = np.zeros((len(points), 3))
        colors[:, 0] = np.clip(2 * (normalized_z - 0.5), 0, 1)
        colors[:, 1] = np.clip(2 * (0.5 - np.abs(normalized_z - 0.5)), 0, 1)
        colors[:, 2] = np.clip(2 * (0.5 - normalized_z), 0, 1)
        return colors

    def update_visualization(self):
        """Actualiza la visualización de la nube de puntos"""
        self.pv_widget.clear()
        for i in range(self.layer_layout.count()):
            layer_widget = self.layer_layout.itemAt(i).widget()
            checkbox = layer_widget.layout().itemAt(0).widget()
            if checkbox.isChecked():
                mesh = self.meshes.get(checkbox.text())
                if mesh:
                    self.pv_widget.add_mesh(mesh, scalars="RGB", rgb=True, point_size=self.current_point_size)
        if len(self.selected_points) > 0:
            selected_point_cloud = pv.PolyData(np.array(self.selected_points))
            self.pv_widget.add_mesh(selected_point_cloud, color='red', point_size=self.current_point_size * 2)
        self.pv_widget.render()

    def remove_layer(self, checkbox):
        """Elimina una capa seleccionada"""
        text = checkbox.text()
        if text in self.meshes:
            del self.meshes[text]
        checkbox.parentWidget().deleteLater()
        self.update_visualization()

    def save_selected_points(self):
        """Guarda los puntos seleccionados en un archivo LAS"""
        if self.selected_points:
            save_path, _ = QFileDialog.getSaveFileName(self, "Guardar archivo LAS", "", "LAS Files (*.las)")
            if save_path:
                header = laspy.LasHeader(point_format=3, version="1.2")
                las = laspy.LasData(header)
                selected_points = np.array(self.selected_points)
                las.x = selected_points[:, 0]
                las.y = selected_points[:, 1]
                las.z = selected_points[:, 2]
                las.write(save_path)
                QMessageBox.information(self, "Éxito", f"Archivo guardado correctamente en: {save_path}")
                self.selected_points.clear()
                self.update_visualization()

    def draw_contour_lines(self):
        """Dibuja las curvas de nivel en el área seleccionada"""
        if self.area_selection:
            points = self.area_selection.points
            grid = pv.PolyData(points)
            grid["Elevation"] = points[:, 2]
            contours = grid.contour(isosurfaces=15, scalars="Elevation")
            self.pv_widget.add_mesh(contours, color='blue', line_width=1)
            self.pv_widget.render()
            QMessageBox.information(self, "Información", "Curvas de nivel dibujadas correctamente.")

        else:
            QMessageBox.warning(self, "Advertencia", "No se ha seleccionado un área para generar curvas de nivel.")

    def enable_area_selection(self):
        """Habilita la selección de un área en forma de rectángulo"""
        self.pv_widget.enable_rubber_band_style()
        self.pv_widget.iren.AddObserver("EndPickEvent", self.extract_selected_area)

    def extract_selected_area(self, *args):
        """Extrae los puntos dentro del área seleccionada"""
        self.area_selection = self.pv_widget.get_picked_polydata()
        self.update_visualization()
        QMessageBox.information(self, "Área Seleccionada", "Área seleccionada correctamente.")

def main():
    app = QApplication(sys.argv)
    viewer = LASClassificationViewer()
    viewer.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

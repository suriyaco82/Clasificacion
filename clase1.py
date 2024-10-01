import sys
import os
import laspy
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, 
                             QPushButton, QProgressBar, QMessageBox, QFileDialog, 
                             QScrollArea, QCheckBox, QHBoxLayout, QSlider, QFrame)
from PyQt5.QtCore import Qt
import pyvista as pv
from pyvistaqt import QtInteractor
import qdarkstyle

class LASClassificationViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Suriyaco Valley - Clasificación de Puntos LAS")
        self.setGeometry(100, 100, 1600, 1000)

        # Configuración de la interfaz
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # Etiqueta principal
        self.label = QLabel("Haz clic en 'Cargar Archivos LAS' para iniciar...")
        self.label.setStyleSheet("font-size: 16px; padding: 10px;")
        self.layout.addWidget(self.label)

        # Visualizador PyVista
        self.pv_widget = QtInteractor(self)
        self.layout.addWidget(self.pv_widget.interactor)

        # Barra de progreso
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMaximum(100)
        self.layout.addWidget(self.progress_bar)

        # Botones y controles
        button_layout = QHBoxLayout()
        self.load_button = QPushButton("Cargar Archivos LAS")
        self.load_button.clicked.connect(self.load_las_files)
        button_layout.addWidget(self.load_button)

        self.start_button = QPushButton("Iniciar Clasificación")
        self.start_button.clicked.connect(self.start_classification)
        self.start_button.setEnabled(False)
        button_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Parar Proceso")
        self.stop_button.clicked.connect(self.stop_classification)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)

        self.point_size_slider = QSlider(Qt.Horizontal)
        self.point_size_slider.setMinimum(1)
        self.point_size_slider.setMaximum(10)
        self.point_size_slider.setValue(5)
        self.point_size_slider.valueChanged.connect(self.update_point_size)
        button_layout.addWidget(QLabel("Tamaño del Punto:"))
        button_layout.addWidget(self.point_size_slider)

        self.layout.addLayout(button_layout)

        # Sección de Capas
        self.layer_title = QLabel("Capas Clasificadas")
        self.layer_title.setStyleSheet("font-size: 14px; font-weight: bold; margin: 5px; color: #fff;")
        self.layout.addWidget(self.layer_title)

        self.layer_scroll = QScrollArea()
        self.layer_widget = QWidget()
        self.layer_layout = QVBoxLayout()
        self.layer_widget.setLayout(self.layer_layout)
        self.layer_scroll.setWidgetResizable(True)
        self.layer_scroll.setWidget(self.layer_widget)
        self.layer_scroll.setFixedHeight(150)
        self.layout.addWidget(self.layer_scroll)

        # Variables para manejo de datos
        self.files_data = []
        self.meshes = {}
        self.class_labels = {0: 'Árboles', 1: 'Construcciones', 2: 'Piscinas'}
        self.stop_requested = False
        self.current_point_size = 5

    def load_las_files(self):
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Seleccione archivos LAS", "", "LAS Files (*.las)")
        if file_paths:
            for file_path in file_paths:
                try:
                    las_data = laspy.read(file_path)
                    self.files_data.append({"file_path": file_path, "las_data": las_data})
                    layer_item = self.create_layer_item(os.path.basename(file_path))
                    self.layer_layout.addWidget(layer_item)
                    self.meshes[layer_item.objectName()] = self.create_pyvista_mesh(las_data)
                    self.label.setText(f"{len(self.files_data)} archivo(s) LAS cargado(s).")
                    self.start_button.setEnabled(True)
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"No se pudo cargar el archivo: {e}")

            self.update_visualization()

    def create_layer_item(self, layer_name):
        frame = QFrame()
        frame.setObjectName(layer_name)
        frame.setStyleSheet("background-color: #f9f9f9; padding: 5px; border: 1px solid #ddd;")
        
        layout = QHBoxLayout(frame)
        
        checkbox = QCheckBox(layer_name)
        checkbox.setChecked(True)
        checkbox.setStyleSheet("color: black;")  # Cambiar el color de la fuente a negro
        checkbox.stateChanged.connect(self.update_visualization)
        layout.addWidget(checkbox)
        
        delete_button = QPushButton("Eliminar")
        delete_button.setFixedSize(54, 25)
        delete_button.setStyleSheet("background-color: #d9534f; color: white; border: none; border-radius: 4px;")
        delete_button.clicked.connect(lambda: self.delete_layer(frame))
        layout.addWidget(delete_button)
        
        return frame

    def delete_layer(self, frame):
        layer_name = frame.objectName()
        if layer_name in self.meshes:
            del self.meshes[layer_name]
        self.layer_layout.removeWidget(frame)
        frame.deleteLater()
        self.update_visualization()

    def create_pyvista_mesh(self, las_data):
        points = np.vstack((las_data.x, las_data.y, las_data.z)).T
        point_cloud = pv.PolyData(points)
        colors = np.column_stack((las_data.red / 65535, las_data.green / 65535, las_data.blue / 65535))
        point_cloud['RGB'] = colors
        return point_cloud

    def update_visualization(self):
        self.pv_widget.clear()
        for i in range(self.layer_layout.count()):
            frame = self.layer_layout.itemAt(i).widget()
            checkbox = frame.findChild(QCheckBox)
            if checkbox.isChecked():
                mesh = self.meshes.get(frame.objectName())
                if mesh:
                    self.pv_widget.add_mesh(mesh, scalars="RGB", rgb=True, point_size=self.current_point_size)
        
        self.pv_widget.reset_camera()
        self.pv_widget.render()

    def start_classification(self):
        if not self.files_data:
            QMessageBox.warning(self, "Advertencia", "Cargue al menos un archivo LAS antes de iniciar la clasificación.")
            return

        self.stop_requested = False
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        QApplication.processEvents()

        try:
            for file_data in self.files_data:
                las = file_data["las_data"]
                total_points = len(las.x)
                chunk_size = 1000000  # Optimización: usar un tamaño de 1 millón de puntos por segmento
                num_chunks = int(np.ceil(total_points / chunk_size))

                self.label.setText(f"Clasificando archivo {os.path.basename(file_data['file_path'])} en {num_chunks} segmentos...")

                for chunk_index in range(num_chunks):
                    if self.stop_requested:
                        self.label.setText("Clasificación detenida por el usuario.")
                        break

                    start_idx = chunk_index * chunk_size
                    end_idx = min((chunk_index + 1) * chunk_size, total_points)
                    points_chunk = np.vstack((las.x[start_idx:end_idx], las.y[start_idx:end_idx], las.z[start_idx:end_idx],
                                              las.red[start_idx:end_idx], las.green[start_idx:end_idx], las.blue[start_idx:end_idx])).T
                    
                    df_chunk = pd.DataFrame(points_chunk, columns=["X", "Y", "Z", "R", "G", "B"])

                    # Entrenar el modelo solo en el primer segmento
                    if chunk_index == 0:
                        self.label.setText(f"Entrenando modelo en el primer segmento de {chunk_size} puntos...")
                        X_train = df_chunk[["X", "Y", "Z", "R", "G", "B"]]
                        threshold_z = np.percentile(df_chunk['Z'], 80)
                        is_pool = ((df_chunk['R'] < 18000) & (df_chunk['G'] > 20000) & (df_chunk['B'] > 20000))
                        df_chunk["label"] = np.where(is_pool, 2, np.where(df_chunk['Z'] > threshold_z, 0, 1))
                        y_train = df_chunk["label"]
                        
                        self.model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42, max_depth=25, min_samples_split=5)
                        self.model.fit(X_train, y_train)
                        self.label.setText("Modelo entrenado con éxito.")

                    # Clasificar el segmento actual
                    df_chunk["predicted_label"] = self.model.predict(df_chunk[["X", "Y", "Z", "R", "G", "B"]])
                    self.save_classified_las(df_chunk, chunk_index)

                    progress_percentage = ((chunk_index + 1) / num_chunks) * 100
                    self.progress_bar.setValue(int(progress_percentage))
                    self.label.setText(f"Procesando segmento {chunk_index + 1}/{num_chunks} - {progress_percentage:.2f}% completado.")
                    QApplication.processEvents()

            self.label.setText("Clasificación completada y guardada.")
            self.progress_bar.setValue(100)
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)

        except Exception as e:
            self.label.setText(f"Error durante la clasificación: {e}")

    def save_classified_las(self, df_chunk, chunk_index):
        file_prefix = f"chunk_{chunk_index}_"
        
        # Guardar capas clasificadas
        for label, name in self.class_labels.items():
            df_filtered = df_chunk[df_chunk["predicted_label"] == label]
            if not df_filtered.empty:
                las_layer = laspy.create(file_version="1.4", point_format=7)
                las_layer.x, las_layer.y, las_layer.z = df_filtered["X"], df_filtered["Y"], df_filtered["Z"]
                las_layer.red, las_layer.green, las_layer.blue = df_filtered["R"].astype(np.uint16), df_filtered["G"].astype(np.uint16), df_filtered["B"].astype(np.uint16)
                las_layer.write(f"{file_prefix}{name}.las")

    def stop_classification(self):
        reply = QMessageBox.question(self, "Parar Proceso", "¿Deseas detener el proceso?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.stop_requested = True
            self.label.setText("Proceso detenido por el usuario...")

    def update_point_size(self):
        self.current_point_size = self.point_size_slider.value()
        self.update_visualization()

def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())  # Aplicando estilos CSS y tema oscuro
    viewer = LASClassificationViewer()
    viewer.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

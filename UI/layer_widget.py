from PyQt5.QtWidgets import QWidget, QHBoxLayout, QSpinBox, QComboBox, QPushButton, QLabel

class LayerWidget(QWidget):
    def __init__(self, layer_index, on_remove_callback):
        super().__init__()
        self.layer_index = layer_index
        self.on_remove_callback = on_remove_callback

        layout = QHBoxLayout()

        self.label = QLabel(f"Шар {layer_index + 1}:")
        layout.addWidget(self.label)

        self.neurons = QSpinBox()
        self.neurons.setRange(1, 2048)
        self.neurons.setValue(128)
        layout.addWidget(self.neurons)

        self.activation = QComboBox()
        self.activation.addItems(["relu", "sigmoid", "tanh", "softmax", "linear"])
        layout.addWidget(self.activation)

        self.remove_btn = QPushButton("Видалити")
        self.remove_btn.clicked.connect(self.remove_self)
        layout.addWidget(self.remove_btn)

        self.setLayout(layout)

    def remove_self(self):
        self.on_remove_callback(self)
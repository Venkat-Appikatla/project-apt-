
from yolox.exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self):
        super().__init__()

        self.num_classes = 13
        self.depth = 1.0
        self.width = 1.0

        self.input_size = (640, 640)
        self.test_size = (640, 640)

        self.data_num_workers = 2

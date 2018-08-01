import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import ctypes
import tensorrt as trt
import cv2
from torchvision import datasets, transforms
import os

CHANNEL = 3
HEIGHT = 224
WIDTH = 224


class PythonEntropyCalibrator(trt.infer.EntropyCalibrator):
    def __init__(self, input_layers, stream):
        trt.infer.EntropyCalibrator.__init__(self)
        self.input_layers = input_layers
        self.stream = stream

        self.d_input = cuda.mem_alloc(self.stream.calibration_data.nbytes)
        stream.reset()

    def get_batch_size(self):
        return self.stream.batch_size

    def get_batch(self, bindings, names):
        batch = self.stream.next_batch()
        if not batch.size:
            return None

        cuda.memcpy_htod(self.d_input, batch)
        for i in self.input_layers[0]:
            assert names[0] != i

        bindings[0] = int(self.d_input)
        return bindings

    def read_calibration_cache(self, length):
        return None

    def write_calibration_cache(self, ptr, size):
        cache = ctypes.c_char_p(int(ptr))
        with open('calibration_cache.bin', 'wb') as f:
            f.write(cache.value)
        return None


class ImageBatchStream():
    def __init__(self, batch_size, filename, prefix):
        self.prefix = prefix
        self.batch_size = batch_size
        lines = open(filename).readlines()
        calibration_files = [s.split()[0] for s in lines]
        self.max_batches = (len(calibration_files) // batch_size) + \
                           (1 if (len(calibration_files) % batch_size)
                            else 0)
        self.files = calibration_files
        self.calibration_data = np.zeros((batch_size, CHANNEL, HEIGHT, WIDTH),
                                         dtype=np.float32)
        self.batch = 0
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),

        ])
        # self.preprocessor = preprocessor

    def read_image_chw(self, path):
        img = cv2.imread(os.path.join(self.prefix, path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (WIDTH, HEIGHT))

        im = self.transform(img)
        im = im.numpy()
        return im

    def reset(self):
        self.batch = 0

    def next_batch(self):
        if self.batch < self.max_batches:
            imgs = []
            files_for_batch = self.files[self.batch_size * self.batch: \
                                         self.batch_size * (self.batch + 1)]
            for f in files_for_batch:
                # print("[ImageBatchStream] Processing ", f)
                img = self.read_image_chw(f)
                imgs.append(img)
            for i in range(len(imgs)):
                self.calibration_data[i] = imgs[i]
            self.batch += 1
            return np.ascontiguousarray(self.calibration_data, dtype=np.float32)
        else:
            return np.array([])

import tensorrt as trt
from tensorrt.parsers import onnxparser
import calib as calibrator
import glob
from random import shuffle
import numpy as np

prefix = './data/final-vision-test-b-0726/data'
anno = './data/test_b.txt'


#
# MEAN = (0.485, 0.456, 0.406)
# STD = (0.229, 0.224, 0.225)
#
#
# # def create_calibration_dataset():
# #     # Create list of calibration images (filename)
# #     # This sample code picks 100 images at random from training set
# #     calibration_files = glob.glob(CALIBRATION_DATASET_LOC)
# #     shuffle(calibration_files)
# #     return calibration_files[:100]
#
#
# # def sub_mean_chw(data):
# #     # data = data.transpose((1, 2, 0))  # CHW -> HWC
# #     data -= np.array(MEAN)  # Broadcast subtract
# #     data /= np.array(STD)
# #     # data = data.transpose((2, 0, 1))  # HWC -> CHW
# #     return data

def onnx_to_int8(anno, prefix):
    apex = onnxparser.create_onnxconfig()

    apex.set_model_file_name("model.onnx")
    apex.set_model_dtype(trt.infer.DataType.FLOAT)
    apex.set_print_layer_info(False)
    trt_parser = onnxparser.create_onnxparser(apex)
    data_type = apex.get_model_dtype()
    onnx_filename = apex.get_model_file_name()
    trt_parser.parse(onnx_filename, data_type)

    trt_parser.convert_to_trtnetwork()
    trt_network = trt_parser.get_trtnetwork()

    # calibration_files = create_calibration_dataset()
    batchstream = calibrator.ImageBatchStream(5, anno, prefix)
    int8_calibrator = calibrator.PythonEntropyCalibrator(["data"], batchstream)

    G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)

    builder = trt.infer.create_infer_builder(G_LOGGER)
    builder.set_max_batch_size(16)
    builder.set_max_workspace_size(1 << 20)
    builder.set_int8_calibrator(int8_calibrator)
    builder.set_int8_mode(True)
    engine = builder.build_cuda_engine(trt_network)
    modelstream = engine.serialize()
    trt.utils.write_engine_to_file("engin.bin", modelstream)
    engine.destroy()
    builder.destroy()

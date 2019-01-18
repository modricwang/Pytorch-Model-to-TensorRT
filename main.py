import torch
from torch.autograd import Variable
import tensorrt as trt
from tensorrt.parsers import onnxparser
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.autoinit

import os
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm, tqdm_notebook

from resnet import resnet50
from image_reader import read_image_chw
import calib as calibrator

args = ArgumentParser().parse_args()
args.input_size = 224
args.input_channel = 3
args.fc_num = 5
args.batch_size = 4

args.pretrained = "checkpoint/model_best.pth"
args.onnx_model_name = "model.onnx"
args.trt_model_name = "engine.bin"

args.fake_test_file = "data/fake_test.txt"
args.test_file = "data/test.txt"
args.img_dir = "data"


def infer(context, input_img, output_size, batch_size):
    # Load engine
    engine = context.get_engine()
    assert (engine.get_nb_bindings() == 2)
    # Convert input data to Float32
    input_img = input_img.astype(np.float32)
    # Create output array to receive data
    output = np.empty(output_size, dtype=np.float32)

    # Allocate device memory
    d_input = cuda.mem_alloc(batch_size * input_img.nbytes)
    d_output = cuda.mem_alloc(batch_size * output.nbytes)

    bindings = [int(d_input), int(d_output)]

    stream = cuda.Stream()

    # Transfer input data to device
    cuda.memcpy_htod_async(d_input, input_img, stream)
    # Execute model
    context.enqueue(batch_size, bindings, stream.handle, None)
    # Transfer predictions back
    cuda.memcpy_dtoh_async(output, d_output, stream)

    # Return predictions
    return output


def do_test(context):
    lines = open(args.test_file).readlines()
    total = 0
    correct = 0
    for line in tqdm(lines):
        path, gt = line.strip().split(',')
        gt = int(gt)
        img = read_image_chw(os.path.join(args.img_dir, path),
                args.input_size, args.input_size)
        output = infer(context, img, 5, 1)
        conf, pred = torch.Tensor(output).topk(1, dim=0)
        pred = int(pred.data[0])
        if pred == gt:
            correct += 1
        total += 1
    return correct, total


def onnx_infer():
    apex = onnxparser.create_onnxconfig()
    apex.set_model_file_name(args.onnx_model_name)
    apex.set_model_dtype(trt.infer.DataType.FLOAT)
    apex.set_print_layer_info(False)
    trt_parser = onnxparser.create_onnxparser(apex)

    data_type = apex.get_model_dtype()
    onnx_filename = apex.get_model_file_name()
    trt_parser.parse(onnx_filename, data_type)
    trt_parser.convert_to_trtnetwork()
    trt_network = trt_parser.get_trtnetwork()

    G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)
    builder = trt.infer.create_infer_builder(G_LOGGER)
    builder.set_max_batch_size(16)
    engine = builder.build_cuda_engine(trt_network)
    context = engine.create_execution_context()

    print ("Start ONNX Test...")
    correct, total = do_test(context)
    print ("ONNX Acc: {}".format(correct / total))


def onnx_to_int8():
    apex = onnxparser.create_onnxconfig()

    apex.set_model_file_name(args.onnx_model_name)
    apex.set_model_dtype(trt.infer.DataType.FLOAT)
    apex.set_print_layer_info(False)
    trt_parser = onnxparser.create_onnxparser(apex)
    data_type = apex.get_model_dtype()
    onnx_filename = apex.get_model_file_name()
    trt_parser.parse(onnx_filename, data_type)

    trt_parser.convert_to_trtnetwork()
    trt_network = trt_parser.get_trtnetwork()

    # calibration_files = create_calibration_dataset()
    batchstream = calibrator.ImageBatchStream(args)
    int8_calibrator = calibrator.PythonEntropyCalibrator(["data"], batchstream)

    G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)

    builder = trt.infer.create_infer_builder(G_LOGGER)
    builder.set_max_batch_size(16)
    builder.set_max_workspace_size(1 << 20)
    builder.set_int8_calibrator(int8_calibrator)
    builder.set_int8_mode(True)
    engine = builder.build_cuda_engine(trt_network)
    modelstream = engine.serialize()
    trt.utils.write_engine_to_file(args.trt_model_name, modelstream)
    engine.destroy()
    builder.destroy()


def trt_infer():
    G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)
    engine = trt.utils.load_engine(G_LOGGER, args.trt_model_name)
    context = engine.create_execution_context()

    print ("Start TensorRT Test...")
    correct, total = do_test(context)
    print('INT8 acc: {}'.format(correct / total))


if __name__ == '__main__':
    if not os.path.exists(args.onnx_model_name):
        # Create your model
        model = resnet50(args).cuda()

        # Translate Pytorch Model into Onnx Model
        dummy_input = Variable(torch.randn(args.batch_size, args.input_channel, \
                args.input_size, args.input_size, device='cuda'))
        output_names = ["output"]
        torch.onnx.export(model, dummy_input, args.onnx_model_name, verbose=False,
                          output_names=output_names)

    onnx_infer()

    if not os.path.exists(args.trt_model_name):
        onnx_to_int8()

    trt_infer()

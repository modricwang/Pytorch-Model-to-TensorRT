import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
import cv2
from tensorrt.parsers import onnxparser
from torchvision import transforms
from torch.autograd import Variable
from torch import Tensor
import os
import argparse


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


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),

])


def read_image_chw(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (224, 224))

    im = transform(img)
    im = im.numpy()
    return im


def onnx_infer(anno, prefix):
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
    G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)
    builder = trt.infer.create_infer_builder(G_LOGGER)
    builder.set_max_batch_size(16)
    engine = builder.build_cuda_engine(trt_network)
    context = engine.create_execution_context()
    lines = open(anno).readlines()
    total = 0
    correct = 0
    for line in lines:
        path, gt = line.strip().split(' ')
        gt = int(gt)
        img = read_image_chw(os.path.join(prefix, path))
        output = infer(context, img, 10, 1)
        conf, pred = Tensor(output).topk(1, dim=0)
        pred = int(pred.data[0])
        if pred == gt:
            correct += 1
        total += 1
    print(correct / total)


def trt_infer(anno, prefix, model='engin.bin'):
    G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)
    engine = trt.utils.load_engine(G_LOGGER, 'engin.bin')
    context = engine.create_execution_context()

    resultf = open('results.txt', 'w')
    print('FILE_ID,CATEGORY_ID0,CATEGORY_ID1,CATEGORY_ID2', file=resultf)

    lines = open(anno).readlines()
    total = 0
    correct = 0
    for line in lines:
        path, gt = line.strip().split(' ')
        gt = int(gt)
        img = read_image_chw(os.path.join(prefix, path))
        output = infer(context, img, 10, 1)
        conf, pred = Tensor(output).topk(1, dim=-1)
        pred = int(pred.data[0])
        print('{},{},{},{}'.format(os.path.basename(path)[:-4], pred, 0, 0), file=resultf)
        if pred == gt:
            correct += 1
        total += 1
    print('int8 acc = %.3f' % (correct / total))
    resultf.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', required=True, help='Model type when we create a new one')
    parser.add_argument('-data_dir', required=True, help='Path to data directory')
    parser.add_argument('-test_list', required=True, help='Path to data directory')
    args = parser.parse_args()
    trt_infer(args.test_list, args.data_dir, args.model)

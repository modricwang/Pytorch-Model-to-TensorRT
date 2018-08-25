import torch
from torch.autograd import Variable
from argparse import ArgumentParser
from resnet import resnet50

from onnx_to_tensorrt import onnx_to_int8

from rt_infer import trt_infer

args = ArgumentParser().parse_args()
args.pretrained = None
args.color_classes = 10
args.type_classes = 4

test_list = '/media/modric/28F4310FF430E12C/non_motor/test_.txt'
data_prefix = '/media/modric/28F4310FF430E12C/non_motor/test/'
weight_path = '/media/modric/28F4310FF430E12C/non_motor/model_76.pth'

if __name__ == '__main__':
    # model = resnet50(args)
    # weights = torch.load(weight_path)['model']
    # model.load_state_dict(weights)
    # dummy_input = Variable(torch.randn(64, 3, 224, 224))
    # output_names = ["output1"]
    # torch.onnx.export(model, dummy_input, "model.onnx", verbose=False,
    #                   output_names=output_names)

    # onnx_to_int8(test_list, data_prefix)

    trt_infer(test_list, data_prefix)

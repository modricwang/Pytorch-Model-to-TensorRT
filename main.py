import torch
from torch.autograd import Variable
from torchvision.models.resnet import resnet50

from onnx_to_tensorrt import onnx_to_int8

from rt_infer import trt_infer

test_list = ''
data_prefix = ''

if __name__ == '__main__':
    model = resnet50(pretrained=True)
    dummy_input = Variable(torch.randn(10, 3, 224, 224)).cuda()
    output_names = ["output1"]
    torch.onnx.export(model, dummy_input, "model.onnx", verbose=False,
                      output_names=output_names)

    onnx_to_int8(test_list, data_prefix)

    trt_infer(test_list, data_prefix)

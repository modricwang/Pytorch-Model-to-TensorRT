# Pytorch-Model-to-TensorRT

## Convert a Pytorch model to TensorRT engin (optional: int8 engin)

### Required:

Python packages: in `requirements.txt` 

External: 

- CUDA >= 8.0
- CUDNN >= 5.0
- TensorRT >= 3.0
- Test dataset and list

The list of test dataset should follow this format:
```text
file0 label0
file1 label1
... ...
``` 

### Convert step:

step0: load your model in main.py 

step1: modify `test_list` and `data_prefix` in main.py

step2:
```bash
python main.py
```

If convert success, `model.onnx` and `engin.bin` will appear in your folder.

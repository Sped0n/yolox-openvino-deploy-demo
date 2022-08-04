# yolox-openvino-deploy-demo
yolox openvino deployment demo
* 使用方法

  我的demo文件可以运行带和不带decoder的onnx和openvino模型，只需要在初始化ovnet类时指定参数即可。

  ```python
  class ovnet:
      def __init__(self, model_path, onnx=False, onnx_input_size=0, need_decoder=False):
        self.enable_onnx = onnx
          self.enable_decoder = need_decoder
          if onnx and onnx_input_size:
              from openvino.runtime import Core
              ie = Core()
              self.net = ie.compile_model(model=model_path, device_name="MYRIAD")  # CPU or MYRIAD
              self.input_layer = None
              self.w = self.h = onnx_input_size
          elif onnx or onnx_input_size:
              print("You need to specify onnx input size while using onnx")
              exit(1)
          else:
              from openvino.inference_engine import IECore
              ie = IECore()
              IR = ie.read_network(model=model_path[0], weights=model_path[1])
              self.net = ie.load_network(network=IR, device_name="MYRIAD")  # CPU or MYRIAD
              self.input_layer = next(iter(self.net.input_info))
              _, _, self.h, self.w = self.net.input_info[self.input_layer].input_data.shape
          self.output_layer = next(iter(self.net.outputs))
          print("model loaded")
  ```

  不需要使用onnx的时候`onnx`和`onnx_input_size`选项不指定即可

  但使用onnx时必须指定`onnx_input_size`（int数据）

* 测试
  1. 先在文件开头指定模型输入路径`model_path`。
  2. 需要检测图片的将`test_img`改为True，再输入`img_path`即可。
  3. 检测视频的输入摄像头的id或者视频文件的路径`vid_path`，`test_img`保持False不动

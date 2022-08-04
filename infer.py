import numpy as np
from process import multiclass_nms, resize, decoder


class ovnet:
    def __init__(self, model_path, onnx=False, onnx_input_size=0, need_decoder=False):
        self.enable_onnx = onnx
        self.enable_decoder = need_decoder
        if onnx and onnx_input_size:
            from openvino.runtime import Core
            ie = Core()
            self.net = ie.compile_model(model=model_path, device_name="MYRIAD")  # CPU MYRIAD
            self.input_layer = None
            self.w = self.h = onnx_input_size
        elif onnx or onnx_input_size:
            print("You need to specify onnx input size while using onnx")
            exit(1)
        else:
            from openvino.inference_engine import IECore
            ie = IECore()
            IR = ie.read_network(model=model_path[0], weights=model_path[1])
            self.net = ie.load_network(network=IR, device_name="MYRIAD")  # CPU MYRIAD
            self.input_layer = next(iter(self.net.input_info))
            _, _, self.h, self.w = self.net.input_info[self.input_layer].input_data.shape
        self.output_layer = next(iter(self.net.outputs))
        print("model loaded")
        # print(self.size)

    def image_preprocess(self, image):
        image = resize(image, (self.h, self.w))
        image = np.ascontiguousarray(image)
        return image

    def forward(self, img, normalize=None):
        img = self.image_preprocess(img)
        if normalize is not None:
            img = (img - normalize[0]) / normalize[1]
        img = np.transpose(np.expand_dims(img, axis=0), (0, 3, 1, 2)).astype(np.float16)
        if self.input_layer is None:
            output = self.net([img])[self.output_layer]
        else:
            output = self.net.infer(inputs={self.input_layer: img})[self.output_layer]
        return output

    def result(self, img, nms_thr=0.45, score_thr=0.1, class_agnostic=True):
        net_output = self.forward(img)
        if self.enable_decoder:
            net_output = decoder(net_output, (self.w, self.h))[0]
        else:
            net_output = net_output[0]
        boxes = net_output[:, :4]
        scores = net_output[:, 4, None] * net_output[:, 5:]
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
        boxes_xyxy /= 1
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=nms_thr, score_thr=score_thr, class_agnostic=class_agnostic)
        return dets

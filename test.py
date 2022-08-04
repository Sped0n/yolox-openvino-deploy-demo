import cv2
import time
from process import vis, resize
from infer import ovnet

cls = ['R3', 'B3', 'R0', 'B0', 'R4', 'B4', 'land']
# model_path = 'example/nets/decode-sim.onnx'
model_path = ('example/nets/yolox_fp16.xml', 'example/nets/yolox_fp16.bin')

if __name__ == '__main__':
    # openvino_model = OpenVINO_Forward(('nets/nanodet_fp16.xml', 'nets/nanodet_fp16.bin'))
    # openvino_model = OpenVINO_Forward('example/nets/decoderv2-sim.onnx')
    # net = ovnet(model_path, onnx=True, onnx_input_size=224, need_decoder=False)
    net = ovnet(model_path, need_decoder=False)
    test_img = False
    # vid_path = 'IMG_6952.mp4'
    vid_path = 0
    if test_img:
        frame = cv2.imread('test.jpg', 1)
        dets = net.result(frame)
        print(dets)
        if dets is not None:
            final_boxes = dets[:, :4]
            final_scores, final_cls_inds = dets[:, 4], dets[:, 5]
            origin_img = vis(frame, final_boxes, final_scores, final_cls_inds,
                             conf=0.3, class_names=cls)
            cv2.imshow('test', origin_img)
            cv2.waitKey()
    else:
        stream = cv2.VideoCapture(vid_path)
        if stream.isOpened():
            ret, frame = stream.read()
            cv2.namedWindow('output', cv2.WINDOW_AUTOSIZE)
        timestamp = time.time()
        fps = 0
        while ret:
            now = time.time()
            fps += 1
            if now - timestamp > 1:
                print(fps)
                fps=0
                timestamp = now
            frame = resize(frame, (224, 224))
            dets = net.result(frame)
            # print(dets)
            if dets is not None:
                final_boxes = dets[:, :4]
                final_scores, final_cls_inds = dets[:, 4], dets[:, 5]
                origin_img = vis(frame, final_boxes, final_scores, final_cls_inds,
                                 conf=0.3, class_names=cls)
            cv2.imshow('test', frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:  # ESC out
                cv2.destroyAllWindows()
                stream.release()
                break
            else:
                ret, frame = stream.read()

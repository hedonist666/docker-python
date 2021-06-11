import torch
import face_detection as fd

detector = fd.build_detector('RetinaNetResNet50',
                                 confidence_threshold=.5,
                                 nms_iou_threshold=.3,
                                 max_resolution=640)

detector = fd.build_detector('RetinaNetMobileNetV1',
                                 confidence_threshold=.5,
                                 nms_iou_threshold=.3)

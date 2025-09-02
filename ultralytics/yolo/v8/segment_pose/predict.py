# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch

from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops
from ultralytics.yolo.v8.detect.predict import DetectionPredictor

class SegmentationPosePredictor(DetectionPredictor):

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = 'segment_pose'

    def _get_kpt_shape(self):
        """安全获取 (nkpt, ndim)，不存在则回退 (17,3)。"""
        try:
            head = self.model.model[-1]
            if hasattr(head, "kpt_shape"):
                nkpt, ndim = head.kpt_shape
                return int(nkpt), int(ndim)
        except Exception:
            pass
        return 17, 3  # fallback

    def postprocess(self, preds, img, orig_imgs):
        """后处理：NMS、mask 生成、关键点/框尺度还原。"""
        # 动态关键点形状
        nkpt, ndim = self._get_kpt_shape()
        nk = nkpt * ndim

        # preds[0] 形状: (B, 4+nc+nm+nk, S)  -> NMS 需要 (B, S, C)
        p = ops.non_max_suppression(preds[0],
                                    self.args.conf,
                                    self.args.iou,
                                    agnostic=self.args.agnostic_nms,
                                    max_det=self.args.max_det,
                                    nc=len(self.model.names),
                                    classes=self.args.classes)

        results = []
        # proto: (B, npr, Hp, Wp)
        proto = preds[1][-2] if len(preds[1]) == 4 else preds[1]

        for i, pred in enumerate(p):
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            path = self.batch[0]
            img_path = path[i] if isinstance(path, list) else path

            if not len(pred):
                # 空结果
                results.append(Results(orig_img=orig_img,
                                       path=img_path,
                                       names=self.model.names,
                                       boxes=pred[:, :6]))
                continue

            # 切片索引说明:
            # pred 列布局: [x1,y1,x2,y2,conf,cls, mask_coeff(nm), keypoints(nk)]
            # mask 系数区间: 6 : -nk (若 nk>0)；若 nk=0 则 6: 末尾
            if nk > 0:
                mask_coeff_cols = pred[:, 6:-nk] if pred.shape[1] > 6 + nk else pred[:, 6:6]  # 兼容极端情况
            else:
                mask_coeff_cols = pred[:, 6:]

            # 生成 mask
            if self.args.retina_masks:
                if not isinstance(orig_imgs, torch.Tensor):
                    pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                masks = ops.process_mask_native(proto[i], mask_coeff_cols, pred[:, :4], orig_img.shape[:2])
            else:
                masks = ops.process_mask(proto[i], mask_coeff_cols, pred[:, :4], img.shape[2:], upsample=True)
                if not isinstance(orig_imgs, torch.Tensor):
                    pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)

            # 关键点处理
            if nk > 0:
                kpt_slice = pred[:, -nk:]
                pred_kpts = kpt_slice.view(len(pred), nkpt, ndim)
                pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, orig_img.shape)
            else:
                pred_kpts = torch.zeros((len(pred), 0, 0), device=pred.device)

            results.append(
                Results(orig_img=orig_img,
                        path=img_path,
                        names=self.model.names,
                        boxes=pred[:, :6],
                        masks=masks,
                        keypoints=pred_kpts))
        return results


def predict(cfg=DEFAULT_CFG, use_python=False):
    """Runs YOLO object detection on an image or video source."""
    model = cfg.model or 'yolov8n-seg.pt'
    source = cfg.source if cfg.source is not None else ROOT / 'assets' if (ROOT / 'assets').exists() \
        else 'https://ultralytics.com/images/bus.jpg'

    args = dict(model=model, source=source)
    if use_python:
        from ultralytics import YOLO
        YOLO(model)(**args)
    else:
        predictor = SegmentationPosePredictor(overrides=args)
        predictor.predict_cli()


if __name__ == '__main__':
    predict()

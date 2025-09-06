# Ultralytics YOLO 🚀, AGPL-3.0 license
from copy import copy

from ultralytics.nn.tasks import SegmentationPoseModel
from ultralytics.yolo import v8
from ultralytics.yolo.utils import DEFAULT_CFG, RANK
from ultralytics.yolo.utils.plotting import plot_images, plot_results


# BaseTrainer python usage
class SegmentationPoseTrainer(v8.detect.DetectionTrainer):

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a SegmentationTrainer object with given arguments."""
        if overrides is None:
            overrides = {}
        overrides['task'] = 'segment_pose'
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True): # 构建对应任务的模型类并按需加载权重
        """Return SegmentationPoseModel initialized with specified config and weights."""
        model = SegmentationPoseModel(cfg, ch=3, nc=self.data['nc'], data_kpt_shape=self.data['kpt_shape'], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)        
        return model

    def get_validator(self):
        """Return an instance of SegmentationPoseValidator for validation of YOLO model."""
        self.loss_names = 'box_loss', 'seg_loss', 'cls_loss', 'dfl_loss', 'pose_loss', 'kobj_loss'
        return v8.segment_pose.SegmentationPoseValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

    def plot_training_samples(self, batch, ni):
        """Creates a plot of training sample images with labels and box coordinates."""
        
        images = batch['img']
        masks = batch['masks']
        cls = batch['cls'].squeeze(-1)
        bboxes = batch['bboxes']
        paths = batch['im_file']
        batch_idx = batch['batch_idx']
        keypoints = batch['keypoints'] # 从batch中解析出'keypoints'
        plot_images(images, batch_idx, cls, bboxes, masks, kpts=keypoints, paths=paths, fname=self.save_dir / f'train_batch{ni}.jpg') # 在train batch中新增keypoints的绘制

    def plot_metrics(self):
        """Plots training/val metrics."""
        plot_results(file=self.csv, segment=True)  # save results.png


def train(cfg=DEFAULT_CFG, use_python=False):
    """Train a YOLO segmentation-pose model based on passed arguments."""
    model = cfg.model or 'yolov8n-segpose.pt'
    data = cfg.data or 'coco-segment_pose.yaml'  # or yolo.ClassificationDataset("mnist")
    device = cfg.device if cfg.device is not None else ''

    args = dict(model=model, data=data, device=device)
    if use_python:
        from ultralytics import YOLO
        YOLO(model).train(**args)
    else:
        trainer = SegmentationPoseTrainer(overrides=args)
        trainer.train()


"""代码改进建议：
- 增加 set_model_attributes 覆写，确保 resume/切换数据安全。
- plot_training_samples 添加关键点显示。
- plot_metrics 传递 pose=True。
- 验证可在 loss_names 中排序分类（可读性：box, cls, dfl | seg | pose, kobj）。
- 引入多任务损失自适应加权（可选）。
- 在 validator 或日志里明确多任务指标列名 (mAP_box, mAP_mask, mAP_kpt)


# ...existing code...
class SegmentationPoseTrainer(v8.detect.DetectionTrainer):
    # ...existing code...

    def set_model_attributes(self):
        super().set_model_attributes()
        # 确保关键点形状同步
        if hasattr(self.data, 'kpt_shape'):
            self.model.kpt_shape = self.data['kpt_shape']

    def plot_training_samples(self, batch, ni):
        images = batch['img']
        masks = batch.get('masks')
        kpts = batch.get('keypoints')
        cls = batch['cls'].squeeze(-1)
        bboxes = batch['bboxes']
        paths = batch['im_file']
        batch_idx = batch['batch_idx']
        plot_images(images, batch_idx, cls, bboxes, masks, kpts=kpts,
                    paths=paths, fname=self.save_dir / f'train_batch{ni}.jpg')

    def plot_metrics(self):
        plot_results(file=self.csv, segment=True, pose=True)
# ...existing code...
"""


if __name__ == '__main__':
    # 入口 train()：统一解析 cfg 中 model/data/device → 组装 overrides → 初始化各自 Trainer → 调用 trainer.train()
    train()

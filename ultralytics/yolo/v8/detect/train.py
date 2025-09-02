# Ultralytics YOLO 🚀, AGPL-3.0 license
# 浅拷贝，用于复制 args（避免原对象被下游修改）
from copy import copy

# 仅用于 plot_training_labels 聚合标签
import numpy as np

# 任务级模型构建类（封装 backbone+neck+head）
from ultralytics.nn.tasks import DetectionModel
# 访问子模块 register（如 v8.detect.DetectionValidator）
from ultralytics.yolo import v8
# 数据相关导入：提供官方新数据管线与兼容旧版 v5loader 的入口
from ultralytics.yolo.data import build_dataloader, build_yolo_dataset
from ultralytics.yolo.data.dataloaders.v5loader import create_dataloader
# 通用训练循环基类BaseTrainer（优化器、调度器、日志、DDP、EMA 等）
from ultralytics.yolo.engine.trainer import BaseTrainer
# 默认配置对象；日志记录器；分布式 rank；彩色字符串工具
from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, RANK, colorstr
from ultralytics.yolo.utils.plotting import plot_images, plot_labels, plot_results
# 处理并行模型（拿真实 module）；分布式环境中只在 rank0 初始化缓存
from ultralytics.yolo.utils.torch_utils import de_parallel, torch_distributed_zero_first


# BaseTrainer python usage
class DetectionTrainer(BaseTrainer):

    def build_dataset(self, img_path, mode='train', batch=None):
        """Build YOLO Dataset

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        # 获取 stride：模型有若干输出层，对应步幅；取最大 stride 保障 mosaic/rect 裁剪符合网格；若模型未构建默认 0 → 用 32 作保底
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        # 依据 args（含增强、imgsz、缓存等）、数据集配置 self.data（解析 data.yaml）、模式 train/val、rect（验证阶段用长边等比矩形填充）创建 Dataset 对象。batch 仅在 rect 策略下影响 aspect ratio bucketing。
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == 'val', stride=gs)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
        """TODO: manage splits differently."""
        # Calculate stride - check if model is initialized
        if self.args.v5loader: # 分支使用旧式 create_dataloader（兼容 YOLOv5 风格）。发出 deprecated 警告。
            LOGGER.warning("WARNING ⚠️ 'v5loader' feature is deprecated and will be removed soon. You can train using "
                           'the default YOLOv8 dataloader instead, no argument is needed.')
            gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
            return create_dataloader(path=dataset_path,
                                     imgsz=self.args.imgsz,
                                     batch_size=batch_size,
                                     stride=gs,
                                     hyp=vars(self.args),
                                     augment=mode == 'train',
                                     cache=self.args.cache,
                                     pad=0 if mode == 'train' else 0.5,
                                     rect=self.args.rect or mode == 'val',
                                     rank=rank,
                                     workers=self.args.workers,
                                     close_mosaic=self.args.close_mosaic != 0,
                                     prefix=colorstr(f'{mode}: '),
                                     shuffle=mode == 'train',
                                     seed=self.args.seed)[0]
        assert mode in ['train', 'val']
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if ，只有 rank0 首次构建（写缓存 .cache），其余等待 → 降低多进程竞争
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == 'train'
        if getattr(dataset, 'rect', False) and shuffle:
            LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == 'train' else self.args.workers * 2
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)  # return dataloader

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images by scaling and converting to float."""
        batch['img'] = batch['img'].to(self.device, non_blocking=True).float() / 255 # 把 uint8 或 half 等转 float32，并归一化到 [0,1]。不在这里做均值方差标准化（YOLO 系列通常直接 0-1）
        return batch

    def set_model_attributes(self):
        """nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)."""
        # self.args.box *= 3 / nl  # scale to layers
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.model.nc = self.data['nc']  # attach number of classes to model
        self.model.names = self.data['names']  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model
        # TODO: self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        # verbose 只在单机主进程打印网络结构（RANK==-1 表示非分布式或主）
        # weights: 若指定预训练权重再 load（内部执行 shape 匹配并忽略不兼容层）
        model = DetectionModel(cfg, nc=self.data['nc'], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model # 返回实例供训练

    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss' # 设置损失名称：与 forward 返回的 loss 张量顺序一致（box 回归、分类、dfl=Distribution Focal Loss）
        return v8.detect.DetectionValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args)) # DetectionValidator：封装评估 (mAP50-95, precision, recall, speed metrics)；接收 test_loader 与配置；使用 copy(self.args) 避免下游修改影响原 args

    def label_loss_items(self, loss_items=None, prefix='train'):
        """
        Returns a loss dict with labelled training loss items tensor
        """
        # Not needed for classification but necessary for segmentation & detection
        keys = [f'{prefix}/{x}' for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        """Returns a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        # 格式化进度显示字符串头部：包含 Epoch, GPU_mem, 各损失名, Instances, Size。用于tqdm动态进度条输出。
        return ('\n' + '%11s' *
                (4 + len(self.loss_names))) % ('Epoch', 'GPU_mem', *self.loss_names, 'Instances', 'Size')

    def plot_training_samples(self, batch, ni):
        """Plots training samples with their annotations.
        images: batch['img']
        batch_idx: 每个目标实例对应的图像索引，用于把 bboxes/cls 映射回图像。
        cls: squeeze(-1) 去掉末尾一维（类别 shape 通常 [N,1]）。
        bboxes: 训练标签的归一化或绝对坐标（函数内部会处理）。
        paths: 原始图路径（叠加标题）。
        fname: 保存路径 train_batch{ni}.jpg
        on_plot: 回调（可能用于自定义画板或 GUI）
        """
        plot_images(images=batch['img'],
                    batch_idx=batch['batch_idx'],
                    cls=batch['cls'].squeeze(-1),
                    bboxes=batch['bboxes'],
                    paths=batch['im_file'],
                    fname=self.save_dir / f'train_batch{ni}.jpg',
                    on_plot=self.on_plot)

    def plot_metrics(self):
        """Plots metrics from a CSV file."""
        # 生成 results.png（常见包含 box_loss/cls_loss/dfl_loss, lr, mAP, precision, recall 等曲线）。在纯检测任务默认无需标识 segment/pose
        plot_results(file=self.csv, on_plot=self.on_plot)  # save results.png

    def plot_training_labels(self):
        """Create a labeled training plot of the YOLO model."""
        boxes = np.concatenate([lb['bboxes'] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb['cls'] for lb in self.train_loader.dataset.labels], 0)
        # plot_labels：绘制（1）类别直方图，（2）宽高分布，（3）宽高比，（4）面积分布等 → 用于数据质量与分布检查
        plot_labels(boxes, cls.squeeze(), names=self.data['names'], save_dir=self.save_dir, on_plot=self.on_plot)


def train(cfg=DEFAULT_CFG, use_python=False):
    """Train and optimize YOLO model given training data and device."""
    model = cfg.model or 'yolov8n.pt'
    data = cfg.data or 'coco128.yaml'  # or yolo.ClassificationDataset("mnist")
    device = cfg.device if cfg.device is not None else ''

    args = dict(model=model, data=data, device=device)
    if use_python:
        from ultralytics import YOLO
        YOLO(model).train(**args)
    else:
        trainer = DetectionTrainer(overrides=args)
        trainer.train()

"""
train() 入口宏观顺序：

1. 收集参数 (model 配置/权重路径, data 数据配置, device)
2. 构造 DetectionTrainer(overrides=args)
3. BaseTrainer.init 内部做：读取 data.yaml → self.data；解析超参；设置保存目录；构建模型 (调用子类 get_model)；构建优化器；构建 dataloader (调用 get_dataloader)；准备 EMA、AMP、回调等
4. 调用 trainer.train():
    - for epoch in range(epochs):
        - 训练循环：迭代 train dataloader
            预处理 batch (preprocess_batch)
            前向 + 得到损失向量
            反向传播 + 更新优化器 + EMA
            累积统计（loss_names 指导映射）
        - 验证：调用 get_validator() 返回的 validator，对 test_loader 做推理、计算 mAP 等
        - 记录/绘图：plot_training_samples（偶尔）、plot_metrics（最终或间隔）、plot_training_labels（首轮）
        - 保存最优/最近权重
5. 结束：results.png + weights + 日志 CSV
"""

if __name__ == '__main__':
    train()

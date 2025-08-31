# Ultralytics YOLO 🚀, AGPL-3.0 license

import contextlib
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

# 从 ultralytics.nn.modules 导入各种模块（Conv, Detect, Segment, Pose 等）
# 这些模块在 parse_model 中根据 yaml 动态实例化为网络层
from ultralytics.nn.modules import (AIFI, C1, C2, C3, C3TR, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x,
                                    Classify, Concat, Conv, Conv2, ConvTranspose, Detect, DWConv, DWConvTranspose2d,
                                    Focus, GhostBottleneck, GhostConv, HGBlock, HGStem, Pose, RepC3, RepConv,
                                    RTDETRDecoder, Segment, SegmentPose)

# 工具函数、默认配置、日志、损失等
from ultralytics.yolo.utils import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, colorstr, emojis, yaml_load
from ultralytics.yolo.utils.checks import check_requirements, check_suffix, check_yaml
from ultralytics.yolo.utils.loss import v8ClassificationLoss, v8DetectionLoss, v8PoseLoss, v8SegmentationLoss, v8SegmentationPoseLoss
from ultralytics.yolo.utils.plotting import feature_visualization
from ultralytics.yolo.utils.torch_utils import (fuse_conv_and_bn, fuse_deconv_and_bn, initialize_weights,
                                                intersect_dicts, make_divisible, model_info, scale_img, time_sync)

try:
    import thop
except ImportError:
    thop = None


class BaseModel(nn.Module):
    """
    BaseModel：所有模型的基类，封装通用的前向、推理、融合、加载、以及损失接口等，定义了通用的前向流程、推理/训练接口、模型加载/融合等基础能力。
    通过继承，子类只需实现任务相关的 init_criterion（返回损失计算器）即可。
    """

    def forward(self, x, *args, **kwargs):
        """
        Forward pass of the model on a single scale.
        Wrapper for `_forward_once` method.

        Args:
            x (torch.Tensor | dict): The input image tensor or a dict including image tensor and gt labels.

        Returns:
            (torch.Tensor): The output of the network.
        """
        """
        通用前向接口：
        - 如果输入 x 是 dict（训练时 dataloader 返回的 batch），则调用 loss(batch) 进入训练逻辑。
        - 否则将进入 predict 流程（推理/评估）。
        训练流程需要同时返回损失/训练项，推理只需要预测输出，所以统一根据输入类型区分。这样将训练与推理分流，调用者只需统一调用 model(batch_or_tensor)。
        """
        if isinstance(x, dict):  # for cases of training and validating while training.
            return self.loss(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)

    def predict(self, x, profile=False, visualize=False, augment=False):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool):  Print the computation time of each layer if True, defaults to False.
            visualize (bool): Save the feature maps of the model if True, defaults to False.
            augment (bool): Augment image during prediction, defaults to False.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        """
        推理入口：
        - augment=True 时使用多尺度/翻转增强（若子类实现）。
        - 否则调用 _predict_once 做单次前向。
        profile 控制是否对每层计时，visualize 控制是否保存特征图。
        """
        if augment:
            return self._predict_augment(x)
        return self._predict_once(x, profile, visualize)

    def _predict_once(self, x, profile=False, visualize=False):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool):  Print the computation time of each layer if True, defaults to False.
            visualize (bool): Save the feature maps of the model if True, defaults to False.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        """
        单次前向实现（通用执行器）：
        - 遍历 self.model（由 parse_model 返回的 nn.Sequential，每个元素 m 包含属性 m.f (from)、m.i (index)、m.type）
        - 根据 m.f 决定当前层的输入来源（支持单输入、多个输入 concat、跳跃连接）
        - profile 时统计层时间与 FLOPs（若 thop 可用）
        - 将层输出按需保存到 y 列表（仅保存 m.i 在 self.save 中的中间结果，节省内存）
        - visualize 时保存特征图
        返回最后一层输出 x（对于多分支 head，子类可能进一步处理）
        """
        y, dt = [], []  # y 保存中间层输出（按索引），dt 保存计时信息
        for m in self.model:
            # m.f != -1 表示当前层的输入不是来自上一层，而是来自指定的 earlier 层（支持 int 或 list）
            if m.f != -1:  # m.f（from）是构建网络时记录的“从哪一层取输入”的索引，支持复杂拓扑（not just sequential）。这是将 YAML 描述的网络（包含 concat、skip 等）运行起来的核心。
                # m.f != -1 表示当前层的输入不是来自上一层，而是来自指定的 earlier 层（支持 int 或 list）
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            # 执行当前模块（调用模块的 forward）
            x = m(x)  # run
            # 若 m.i 在 self.save (parse_model 计算得到的 savelist)，则把输出保存到 y；否则保存 None 占位节省内存。根据 m.i 是否在 self.save 决定是否保存该层输出（供后续层索引使用或 post-process）
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                # 将特征图可视化（保存到 visualize 指定目录）
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _predict_augment(self, x):
        """ 默认不支持增强推理：打印警告并退回单尺度推理（BaseModel 不实现增强）。DetectionModel 会重写为有效实现。"""
        LOGGER.warning(
            f'WARNING ⚠️ {self.__class__.__name__} has not supported augment inference yet! Now using single-scale inference instead.'
        )
        return self._predict_once(x)

    def _profile_one_layer(self, m, x, dt):
        """
        Profile the computation time and FLOPs of a single layer of the model on a given input.
        Appends the results to the provided list.

        Args:
            m (nn.Module): The layer to be profiled.
            x (torch.Tensor): The input data to the layer.
            dt (list): A list to store the computation time of the layer.

        Returns:
            None
        """
        """
        对单层进行性能分析（计时、FLOPs）：
        - 使用 thop 估算 FLOPs（如果可用）
        - 调用多次 forward 来测量耗时（cloning 特殊处理用于最后一层）
        - 打印每层的时间/GFLOPs/参数量信息，最后打印总时间
        """
        c = m == self.model[-1]  # is final layer, copy input as inplace fix
        # 计算 FLOPs（若 thop 未安装则为 0）
        o = thop.profile(m, inputs=[x.clone() if c else x], verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            # 多次运行以稳定计时结果
            m(x.clone() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self, verbose=True):
        """
        Fuse the `Conv2d()` and `BatchNorm2d()` layers of the model into a single layer, in order to improve the
        computation efficiency.

        Returns:
            (nn.Module): The fused model is returned.
        """
        """
        推理时的算子融合（Conv + BN -> single Conv）：
        行为：遍历模型模块，将 Conv/BN 合并成一个 conv（调用 fuse_conv_and_bn），同理对转置卷积合并 BN，RepConv 合并多分支；然后更新 forward 为 fuse 版本。
        原理：推理优化。BN 与前置 conv 合并，能减少运行时开销并稍微提高吞吐。
        目的：减少算子数、内存读取、提高推理速度（仅适用于 eval 模式）
        """
        if not self.is_fused():
            for m in self.model.modules():
                # 对常见的卷积模块做 BN 合并
                if isinstance(m, (Conv, Conv2, DWConv)) and hasattr(m, 'bn'):
                    if isinstance(m, Conv2):
                        # Conv2（可能是带分支或组合卷积）内部还有 fuse_convs 的自定义合并逻辑
                        m.fuse_convs()
                    # 用 util 中的 fuse_conv_and_bn 生成新的 conv（合并后的权重/偏置）
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                    delattr(m, 'bn')  # 删除 batchnorm 属性，减少推理开销
                    m.forward = m.forward_fuse  # update forward
                # 对转置卷积也类似
                if isinstance(m, ConvTranspose) and hasattr(m, 'bn'):
                    m.conv_transpose = fuse_deconv_and_bn(m.conv_transpose, m.bn)
                    delattr(m, 'bn')  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
                # RepConv 是一种训练时可重参数化的模块，fuse_convs 会合并多分支到单一 conv
                if isinstance(m, RepConv):
                    m.fuse_convs()
                    m.forward = m.forward_fuse  # update forward
            # 打印合并后信息
            self.info(verbose=verbose)

        return self

    def is_fused(self, thresh=10):
        """
        Check if the model has less than a certain threshold of BatchNorm layers.

        Args:
            thresh (int, optional): The threshold number of BatchNorm layers. Default is 10.

        Returns:
            (bool): True if the number of BatchNorm layers in the model is less than the threshold, False otherwise.
        """
        """
        检查模型是否已经 fuse（通过统计 Norm 层数量判断）：
        - 将 torch.nn 模块字典中包含 'Norm' 的类型视为归一化层（BatchNorm2d 等）
        - 若模型归一化层数小于阈值，则认为已经 fuse
        注：这个判定并非绝对，但通常用于检测是否已经合并 BN。
        """
        bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
        return sum(isinstance(v, bn) for v in self.modules()) < thresh  # True if < 'thresh' BatchNorm layers in model

    def info(self, detailed=False, verbose=True, imgsz=640):
        """
        Prints model information

        Args:
            verbose (bool): if True, prints out the model information. Defaults to False
            imgsz (int): the size of the image that the model will be trained on. Defaults to 640
        """
        """
        包装 model_info：打印模型总体信息（参数量、层结构、推理尺寸估计等）
        """
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)

    def _apply(self, fn):
        """
        `_apply()` is a function that applies a function to all the tensors in the model that are not
        parameters or registered buffers

        Args:
            fn: the function to apply to the model

        Returns:
            A model that is a Detect() object.
        """
        """
        在 model.to(device) 或 .cuda() 时会调用 _apply，将 module 的 Tensor/Buffer 转到对应 device。
        BaseModel 在父实现基础上，额外处理 Detect/Segment 类中存放的 stride/anchors/strides（这些不是 nn.Parameter / buffer）
        否则这些常量不会随 model.to(device) 移动，从而导致运行时 device mismatch。
        
        行为：重载 nn.Module._apply，调用父实现后，额外对 Detect/Segment 类的某些张量（stride、anchors、strides）应用 fn（例如在 model.to(device) 时把这些常量转到对应 device）。
        原因：Detect/Segment 有内部不是 Parameter 的张量（stride、anchors），默认 _apply 不会移动这些到 GPU，需要手动处理。
        """
        self = super()._apply(fn)
        m = self.model[-1]  # Detect(), head layer
        if isinstance(m, (Detect, Segment)):
            # stride/anchors/strides 可能是 torch.tensor，需要手动 apply 到同一 device
            m.stride = fn(m.stride)
            m.anchors = fn(m.anchors)
            m.strides = fn(m.strides)
        return self

    def load(self, weights, verbose=True):
        """Load the weights into the model.

        Args:
            weights (dict | torch.nn.Module): The pre-trained weights to be loaded.
            verbose (bool, optional): Whether to log the transfer progress. Defaults to True.
        """
        """
        行为：把 checkpoint 的 state_dict 与当前 model 的 state_dict 做交集（intersect_dicts），并以 strict=False 加载。用于把预训练权重映射到当前模型（兼容性加载）。
        - 支持传入 dict（ckpt）或直接 nn.Module 对象
        - 使用 intersect_dicts 只取两者共有的键，避免因结构差异报错
        - 以 strict=False 加载，打印转移数量
        """
        model = weights['model'] if isinstance(weights, dict) else weights  # torchvision models are not dicts
        csd = model.float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, self.state_dict())  # intersect
        self.load_state_dict(csd, strict=False)  # load
        if verbose:
            LOGGER.info(f'Transferred {len(csd)}/{len(self.model.state_dict())} items from pretrained weights')

    def loss(self, batch, preds=None):
        """
        Compute loss

        Args:
            batch (dict): Batch to compute loss on
            preds (torch.Tensor | List[torch.Tensor]): Predictions.
        """
        """
        行为：如果没有 criterion（损失模块），调用 init_criterion 构建一个（子类实现）。拿到 preds（若 None，用 batch['img'] 调 forward），再调用 self.criterion(preds, batch) 计算损失。
        设计：BaseModel 不实现具体损失，交给子类（检测/分割/姿态）实现 init_criterion
        
        统一损失入口：
        - 如果 self.criterion 不存在则调用 init_criterion 构造任务专用损失函数
        - preds 可选（如果不传，自动调用 self.forward(batch['img']) 获取预测）
        - 返回 self.criterion(preds, batch)（通常返回 loss 和 logging 信息）
        """
        if not hasattr(self, 'criterion'):
            self.criterion = self.init_criterion()

        preds = self.forward(batch['img']) if preds is None else preds
        return self.criterion(preds, batch)

    def init_criterion(self):
        # 抽象方法，子类必须实现（返回一个 callable loss 计算器），例如 v8DetectionLoss
        raise NotImplementedError('compute_loss() needs to be implemented by task heads')

"""
BaseModel 的核心设计思路总结：

提供推理/训练入口切换（forward 判别 dict vs tensor）
提供逐层执行器并支持非线性拓扑（m.f 指向）
抽象出任务无关的工具（fuse、load、profile、_apply）
强制子类实现任务相关的损失/配置（init_criterion）
"""

class DetectionModel(BaseModel):
    """YOLOv8 detection model.
    DetectionModel 在 BaseModel 基础上添加了目标检测任务所需的构建、初始化和推理增强逻辑；主要在 init、_predict_augment、_descale_pred、_clip_augmented、init_criterion 中扩展。
    负责根据 yaml 构建模型（backbone + head），初始化 stride、bias，并提供 detection 专用的增强推理与损失构造。
    """

    def __init__(self, cfg='yolov8n.yaml', ch=3, nc=None, verbose=True):  # model, input channels, number of classes
        super().__init__()
         # 加载 YAML 配置（可为 dict 或文件路径）
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict

        # Define model
        # 将输入通道写回 yaml
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        # 如果外部传入 nc（类别数）且不同于 yaml 中的，覆盖 yaml 的 nc（方便重用 yaml）
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        # parse_model 根据 yaml 返回 nn.Sequential 层列表 和 需要保存的输出层索引 self.save
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        # 默认 names（类别名）占位
        self.names = {i: f'{i}' for i in range(self.yaml['nc'])}  # default names dict
        # inplace 激活等设置（是否使用原地操作）
        self.inplace = self.yaml.get('inplace', True)

        # Build strides （计算检测 head 的 stride 信息）detection head 依赖 anchors/stride/bias 的正确初始化。stride 决定如何把输出网格坐标映射到输入图像坐标，必须动态根据 backbone 的输出结构计算（因为 YAML 可以定义不同深度/宽度、输入大小）。
        m = self.model[-1]  # Detect(), 即这个模型的head部分
        if isinstance(m, (Detect, Segment, Pose, SegmentPose)): # 这说明模型尾部是一个 detection/segment/pose head。需要计算 anchor stride（网络下采样比例）：
            s = 256  # # 选取一个固定尺寸（2x min stride）来进行一次前向以测量各个 detection 层的下采样比例
            m.inplace = self.inplace
            # 对于 Segment/Pose/SegmentPose，forward 可能返回 (y, train_outputs) 之类结构，统一取 [0]
            forward = lambda x: self.forward(x)[0] if isinstance(m, (Segment, Pose, SegmentPose)) else self.forward(x)
            # 使用 zeros 输入做一次前向，读取每个输出特征图的 spatial size（x.shape[-2]），由此计算 stride = input_size / feat_size
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # 原理：通过对一个特定大小（s=256）的空白图像做一次前向，查看 head 不同输出特征图的空间尺寸（x.shape[-2]），据此计算 stride（输入像素/输出特征图尺寸）。例如如果输出 P3 尺寸是 32，stride = 256/32 = 8。
            self.stride = m.stride # 保存全局 stride 信息
            m.bias_init()  # only run once，# head 初始化偏置（常用于提高训练稳定性：设置 objectness/class bias 初始值）。给检测 head 的偏置用合适初始值可以加快训练收敛（减少一开始大量背景预测的影响）。
        else:
            # 如果不是常见的 YOLO head（例如 RTDETR），使用默认 stride
            self.stride = torch.Tensor([32])  # default stride for i.e. RTDETR

        # Init weights, biases
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info('')

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference and train outputs.
        实现检测任务的增强推理（多尺度 + 翻转）：
        - 对不同 scale / flip 组合做前向
        - 将预测结果反变换回原始尺度（_descale_pred）
        - 对不同尺度产生的结果进行“tails”裁剪（_clip_augmented）
        - 最终返回 concat 后的结果（用于后续 NMS 融合）
        
        实现了多尺度与翻转增强推理：对几种 scale/flip 组合进行前向，得到多个 yi 后将预测反向缩放（_descale_pred），最后把不同尺度的结果合并并进行特殊裁剪（_clip_augmented）。
        原理：测试时用多尺度翻转推理并融合结果可以提升 mAP，但需要对返回的 bbox 坐标反向变换回原图尺度，并去掉重复/边界不一致部分
        
        """
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales（不同缩放因子）
        f = [None, 3, None]  # flips（None, 3 表示左右翻转，2 表示上下翻转）
        y = []  # 用来保存每个增强结果
        for si, fi in zip(s, f):
            # scale_img 会按 grid size(gs) 对图像进行整形（保持与 stride 对齐）
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            # 调用父类 predict 做单次前向（super().predict -> BaseModel.predict -> _predict_once）
            yi = super().predict(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            # 将预测去尺度并反向翻转到原图坐标系
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        # 对增强产生的多个预测进行 tail 裁剪（源自 YOLOv5 的融合策略）
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, -1), None  # augmented inference, train（返回拼接的预测与 None 占位训练输出）

    @staticmethod
    def _descale_pred(p, flips, scale, img_size, dim=1):
        """De-scale predictions following augmented inference (inverse operation)."""
        """将增强推理的预测恢复到原始图像尺度并反向翻转：
        - p 的前 4 个通道为 x, y, w, h（中心坐标 + 尺寸）
        - 首先对坐标除以 scale（反缩放）
        - 然后根据 flips（2 上下翻转, 3 左右翻转）对 x 或 y 坐标做反变换
        - 剩余通道为类别等信息，直接拼接返回
        
        对预测进行去尺度（除以 scale）并反向翻转坐标。
        重要点：预测 tensor 组织通常为 [x, y, w, h, cls...] 或者其他维度布局，需要拆分并针对翻转做坐标变换。
        """
        p[:, :4] /= scale  # de-scale: 恢复到原始尺度
        # 将张量按维度分割为 x, y, wh, cls（注意 cls 包含置信度 + 类别 logits 等）
        x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
        if flips == 2:
            y = img_size[0] - y  # de-flip ud（上下翻转的反向）
        elif flips == 3:
            x = img_size[1] - x  # de-flip lr（左右翻转的反向）
        return torch.cat((x, y, wh, cls), dim)

    def _clip_augmented(self, y):
        """Clip YOLOv5 augmented inference tails."""
        """对增强推理结果做裁剪（去除不同尺度产生的 tails）：
        - 该逻辑来源于 YOLOv5 的多尺度融合实现，按 detection 层数 (nl) 计算网格点 g，
          并按层大小裁剪第一/最后一个增强结果的片段，避免重复计入边界处的预测。
        - 这是工程化的后处理，用于改善多尺度融合一致性。
        
        修剪不同尺度预测的“tails”（推理融合时需要裁剪不同大小特征层产生的多余部分，保证最终融合的网格一致性）。这部分比较工程化，来自 YOLOv5 的增强融合实现。
        """
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points（总的网格点数的度量）
        e = 1  # exclude layer count（排除层级的宽度）
        i = (y[0].shape[-1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][..., :-i]  # large: 剪掉大尺度预测的尾部
        i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][..., i:]  # small: 剪掉小尺度预测的头部
        return y

    def init_criterion(self):
        """为检测任务返回 v8DetectionLoss：检测特定的损失函数，包含图像尺度映射、anchor-matching、IoU、分类损失等逻辑"""
        return v8DetectionLoss(self)


class SegmentationModel(DetectionModel):
    """YOLOv8 segmentation model（继承 DetectionModel，仅替换损失与增强推理行为）。"""

    def __init__(self, cfg='yolov8n-seg.yaml', ch=3, nc=None, verbose=True):
        """初始化 segmentation model，只是调用 DetectionModel.init（默认 cfg='yolov8n-seg.yaml'）"""
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """返回分割任务专用的损失实现（处理 mask loss 等）分割损失会处理 mask 分支、像素级损失（BCE/IoU等）、以及与检测一起的多任务权重"""
        return v8SegmentationLoss(self)

    def _predict_augment(self, x):
        """覆盖成警告 + 单尺度（暂不支持增强）。说明作者暂时不实现分割的多尺度增强融合（因为 mask 对齐更复杂）"""
        LOGGER.warning(
            f'WARNING ⚠️ {self.__class__.__name__} has not supported augment inference yet! Now using single-scale inference instead.'
        )
        return self._predict_once(x)

class SegmentationPoseModel(DetectionModel):
    """YOLOv8 同时输出分割与姿态（keypoints）的模型。"""

    def __init__(self, cfg='yolov8n-segpose.yaml', ch=3, nc=None, data_kpt_shape=(None, None), verbose=True):
        """init：加载 yaml（若不是 dict）并允许覆盖 kpt_shape（data_kpt_shape），然后调用 DetectionModel.init。"""
        if not isinstance(cfg, dict):
            cfg = yaml_model_load(cfg)  # load model YAML
        if any(data_kpt_shape) and list(data_kpt_shape) != list(cfg['kpt_shape']):
            LOGGER.info(f"Overriding model.yaml kpt_shape={cfg['kpt_shape']} with kpt_shape={data_kpt_shape}")
            cfg['kpt_shape'] = data_kpt_shape
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """返回 v8SegmentationPoseLoss(self)：处理同时含有分割和姿态输出的联合损失（需要平衡两类任务）。"""
        return v8SegmentationPoseLoss(self)

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference."""
        """暂不支持该混合任务的增强推理"""
        LOGGER.warning(
            f'WARNING ⚠️ {self.__class__.__name__} has not supported augment inference yet! Now using single-scale inference instead.'
        )
        return self._predict_once(x)

class PoseModel(DetectionModel):
    """YOLOv8 pose model（关键点检测）。"""

    def __init__(self, cfg='yolov8n-pose.yaml', ch=3, nc=None, data_kpt_shape=(None, None), verbose=True):
        """init：支持传入 data_kpt_shape（数据集关键点数/布局），若与 yaml 中 kpt_shape 不同则覆盖 yaml。原因：不同数据集关键点数量不同（COCO vs MPII 等），head 的输出通道数需要对应。"""
        if not isinstance(cfg, dict):
            cfg = yaml_model_load(cfg)  # load model YAML
        if any(data_kpt_shape) and list(data_kpt_shape) != list(cfg['kpt_shape']):
            LOGGER.info(f"Overriding model.yaml kpt_shape={cfg['kpt_shape']} with kpt_shape={data_kpt_shape}")
            cfg['kpt_shape'] = data_kpt_shape
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """返回 v8PoseLoss(self)：姿态损失处理关键点热图、关键点坐标回归、可见性权重等。"""
        return v8PoseLoss(self)

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference."""
        """暂不支持增强推理"""
        LOGGER.warning(
            f'WARNING ⚠️ {self.__class__.__name__} has not supported augment inference yet! Now using single-scale inference instead.'
        )
        return self._predict_once(x)


class ClassificationModel(BaseModel):
    """YOLOv8 classification model（从 detection model 或 yaml 构建分类模型）。"""

    def __init__(self,
                 cfg=None,
                 model=None,
                 ch=3,
                 nc=None,
                 cutoff=10,
                 verbose=True):  # yaml, model, channels, number of classes, cutoff index, verbose flag
        super().__init__()
        # 若提供了 detection model，则从 detection model 中裁剪 backbone 并替换 head 为分类头
        self._from_detection_model(model, nc, cutoff) if model is not None else self._from_yaml(cfg, ch, nc, verbose)

    def _from_detection_model(self, model, nc=1000, cutoff=10):
        """Create a YOLOv5 classification model from a YOLOv5 detection model."""
        """基于 detection model 构建分类模型：
        - 解包 DetectMultiBackend（若有）
        - 截断 backbone（取到 cutoff 索引）
        - 构建 Classify() 头替换原先 head，确保输入通道一致
        """
        from ultralytics.nn.autobackend import AutoBackend
        if isinstance(model, AutoBackend):
            model = model.model  # unwrap DetectMultiBackend
        model.model = model.model[:cutoff]  # backbone
        m = model.model[-1]  # last layer
        ch = m.conv.in_channels if hasattr(m, 'conv') else m.cv1.conv.in_channels  # ch into module
        c = Classify(ch, nc)  # Classify()
        c.i, c.f, c.type = m.i, m.f, 'models.common.Classify'  # index, from, type
        model.model[-1] = c  # replace
        self.model = model.model
        self.stride = model.stride
        self.save = []
        self.nc = nc

    def _from_yaml(self, cfg, ch, nc, verbose):
        """Set YOLOv8 model configurations and define the model architecture."""
        """直接从 yaml 构建分类模型（类似 DetectionModel 的 parse 流程）"""
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        elif not nc and not self.yaml.get('nc', None):
            raise ValueError('nc not specified. Must specify nc in model.yaml or function arguments.')
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        # 分类模型不关注 stride 约束，故设为 1
        self.stride = torch.Tensor([1])  # no stride constraints
        self.names = {i: f'{i}' for i in range(self.yaml['nc'])}  # default names dict
        self.info()

    @staticmethod
    def reshape_outputs(model, nc):
        """Update a TorchVision classification model to class count 'n' if required."""
        """调整最后一层输出以匹配类别数（适配 torchvision 模型或 YOLO 的 Classify 头）"""
        name, m = list((model.model if hasattr(model, 'model') else model).named_children())[-1]  # last module
        if isinstance(m, Classify):  # YOLO Classify() head
            if m.linear.out_features != nc:
                m.linear = nn.Linear(m.linear.in_features, nc)
        elif isinstance(m, nn.Linear):  # ResNet, EfficientNet
            if m.out_features != nc:
                setattr(model, name, nn.Linear(m.in_features, nc))
        elif isinstance(m, nn.Sequential):
            types = [type(x) for x in m]
            if nn.Linear in types:
                i = types.index(nn.Linear)  # nn.Linear index
                if m[i].out_features != nc:
                    m[i] = nn.Linear(m[i].in_features, nc)
            elif nn.Conv2d in types:
                i = types.index(nn.Conv2d)  # nn.Conv2d index
                if m[i].out_channels != nc:
                    m[i] = nn.Conv2d(m[i].in_channels, nc, m[i].kernel_size, m[i].stride, bias=m[i].bias is not None)

    def init_criterion(self):
        """Compute the classification loss between predictions and true labels."""
        """返回分类损失实现"""
        return v8ClassificationLoss()


class RTDETRDetectionModel(DetectionModel):
    """针对 RT-DETR 的 DetectionModel 子类，覆盖损失和 predict 以适配 transformer-decoder 风格的 head。"""

    def __init__(self, cfg='rtdetr-l.yaml', ch=3, nc=None, verbose=True):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Compute the classification loss between predictions and true labels."""
        """引入 RTDETR 特定损失实现"""
        from ultralytics.vit.utils.loss import RTDETRDetectionLoss

        return RTDETRDetectionLoss(nc=self.nc, use_vfl=True)

    def loss(self, batch, preds=None):
        # 为 RTDETR 定制的 loss 入口：
        # - 需要把 batch 中的 gt boxes/labels 重排为 decoder / encoder 所需的格式
        if not hasattr(self, 'criterion'):
            self.criterion = self.init_criterion()

        img = batch['img']
        # NOTE: preprocess gt_bbox and gt_labels to list.
        bs = len(img)
        batch_idx = batch['batch_idx']
        gt_groups = [(batch_idx == i).sum().item() for i in range(bs)]
        targets = {
            'cls': batch['cls'].to(img.device, dtype=torch.long).view(-1),
            'bboxes': batch['bboxes'].to(device=img.device),
            'batch_idx': batch_idx.to(img.device, dtype=torch.long).view(-1),
            'gt_groups': gt_groups}

        preds = self.predict(img, batch=targets) if preds is None else preds
        # preds 在训练/推理时结构不同，这里解包对应输出
        dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = preds if self.training else preds[1]
        if dn_meta is None:
            dn_bboxes, dn_scores = None, None
        else:
            dn_bboxes, dec_bboxes = torch.split(dec_bboxes, dn_meta['dn_num_split'], dim=2)
            dn_scores, dec_scores = torch.split(dec_scores, dn_meta['dn_num_split'], dim=2)

        dec_bboxes = torch.cat([enc_bboxes.unsqueeze(0), dec_bboxes])  # (7, bs, 300, 4)
        dec_scores = torch.cat([enc_scores.unsqueeze(0), dec_scores])

        loss = self.criterion((dec_bboxes, dec_scores),
                              targets,
                              dn_bboxes=dn_bboxes,
                              dn_scores=dn_scores,
                              dn_meta=dn_meta)
        # NOTE: There are like 12 losses in RTDETR, backward with all losses but only show the main three losses.
        # 返回总 loss 与用于显示的三个主 loss（GIoU/class/bbox）
        return sum(loss.values()), torch.as_tensor([loss[k].detach() for k in ['loss_giou', 'loss_class', 'loss_bbox']],
                                                   device=img.device)

    def predict(self, x, profile=False, visualize=False, batch=None, augment=False):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model
            profile (bool):  Print the computation time of each layer if True, defaults to False.
            visualize (bool): Save the feature maps of the model if True, defaults to False
            batch (dict): A dict including gt boxes and labels from dataloader.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        """
        RTDETR 特殊的 predict：
        - backbone 和 neck 通过普通遍历 self.model[:-1] 执行（与 BaseModel 类似）
        - head 部分需要把中间特征列表与 batch 一起传入 head（head 定义了 decoder 的调用接口）
        - 因此这里将最后一个模块 head 单独处理，避免使用 _predict_once 的默认行为
        """
        y, dt = [], []  # outputs
        for m in self.model[:-1]:  # except the head part
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        head = self.model[-1]
        # head 接受多个输入（通过 head.f 指定的索引），并可能额外需要 batch（gt）信息
        x = head([y[j] for j in head.f], batch)  # head inference
        return x


class Ensemble(nn.ModuleList):
    """Ensemble of models."""
    """简单的模型集合（ensemble）工具：把多个模型的输出按第三维拼接，供外部进行 NMS 融合。"""

    def __init__(self):
        """Initialize an ensemble of models."""
        """初始化 ensemble（本质上是一个 ModuleList）"""
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        """Function generates the YOLOv5 network's final layer."""
        """对集合中的每个模型调用 forward，并将返回的预测拼接：
        - 返回的 y 形状为 (B, HW, C_total)，外部可运行 NMS 融合多个模型预测
        """
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 2)  # nms ensemble, y shape(B, HW, C)
        return y, None  # inference, train output


# Functions ------------------------------------------------------------------------------------------------------------
"""
- 检测（DetectionModel）：
    - 任务：预测边界框（x,y,w,h） + 类别 + 置信度
    - 需求：正确把不同尺度的特征图输出映射回输入图像坐标（stride）、初始化 bias/anchors、匹配 GT（loss 中处理）
- 代码支持：parse_model 构建 head、计算 stride、bias_init、v8DetectionLoss

- 分割（SegmentationModel）：
    - 任务：像素级 mask 预测，通常基于检测的 bbox 或直接语义分割 head
    - 需求：额外的 mask 分支通道、mask loss、后处理不同（mask 出力需要上采样到原图）
    - 代码支持：Segment head 在 parse_model 中被识别并据此配置，损失用 v8SegmentationLoss

- 姿态（PoseModel）：
    - 任务：关键点坐标/热力图预测
    - 需求：关键点数量（kpt_shape）是可变的；pose loss 处理 heatmap/visibility 等
    - 代码支持：构造函数允许覆盖 kpt_shape，parse_model 会把 kpt 参数传入 Pose head，损失用 v8PoseLoss

- 分割+姿态（SegmentationPoseModel）：
    - 任务同时包含 mask 与关键点，损失与输出更复杂
    - 代码支持：SegmentPose head 的 parse 逻辑（parse_model 中对 SegmentPose 的 args 处理）和对应联合损失 v8SegmentationPoseLoss
"""

def torch_safe_load(weight):
    """
    This function attempts to load a PyTorch model with the torch.load() function. If a ModuleNotFoundError is raised,
    it catches the error, logs a warning message, and attempts to install the missing module via the
    check_requirements() function. After installation, the function again attempts to load the model using torch.load().

    Args:
        weight (str): The file path of the PyTorch model.

    Returns:
        (dict): The loaded PyTorch model.
    """
    """
    安全加载 PyTorch checkpoint 的辅助函数：
    - 若加载时出现 ModuleNotFoundError（模型依赖的第三方模块缺失），尝试通过 check_requirements 自动安装该模块后再加载
    - 对早期 yolov5 ckpt 给出友好提示（不兼容 YOLOv8）
    返回 (ckpt_dict, filepath)
    """
    from ultralytics.yolo.utils.downloads import attempt_download_asset

    check_suffix(file=weight, suffix='.pt')
    file = attempt_download_asset(weight)  # search online if missing locally，若本地缺失则尝试下载
    try:
        return torch.load(file, map_location='cpu'), file  # load
    except ModuleNotFoundError as e:  # e.name is missing module name
        if e.name == 'models':
            # 如果误差来自于 yolov5 的老模型结构，给出明确错误提示
            raise TypeError(
                emojis(f'ERROR ❌️ {weight} appears to be an Ultralytics YOLOv5 model originally trained '
                       f'with https://github.com/ultralytics/yolov5.\nThis model is NOT forwards compatible with '
                       f'YOLOv8 at https://github.com/ultralytics/ultralytics.'
                       f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
                       f"run a command with an official YOLOv8 model, i.e. 'yolo predict model=yolov8n.pt'")) from e
        LOGGER.warning(f"WARNING ⚠️ {weight} appears to require '{e.name}', which is not in ultralytics requirements."
                       f"\nAutoInstall will run now for '{e.name}' but this feature will be removed in the future."
                       f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
                       f"run a command with an official YOLOv8 model, i.e. 'yolo predict model=yolov8n.pt'")
        check_requirements(e.name)  # install missing module

        return torch.load(file, map_location='cpu'), file  # load


def attempt_load_weights(weights, device=None, inplace=True, fuse=False):
    """Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a."""
    """加载一个或多个权重文件，返回模型或 Ensemble：
    - 支持权重列表 -> 构建 Ensemble
    - 对于每个 ckpt，解包 model（或 ema），设置 model.args/pt_path/task/stride，并可选 fuse
    """

    ensemble = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt, w = torch_safe_load(w)  # load ckpt
        args = {**DEFAULT_CFG_DICT, **ckpt['train_args']} if 'train_args' in ckpt else None  # combined args
        model = (ckpt.get('ema') or ckpt['model']).to(device).float()  # FP32 model

        # Model compatibility updates
        model.args = args  # attach args to model
        model.pt_path = w  # attach *.pt file path to model
        model.task = guess_model_task(model)
        if not hasattr(model, 'stride'):
            model.stride = torch.tensor([32.])

        # Append（可选 fuse）
        ensemble.append(model.fuse().eval() if fuse and hasattr(model, 'fuse') else model.eval())  # model in eval mode

    # Module compatibility updates（处理 PyTorch 版本兼容设置）
    for m in ensemble.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Segment):
            # 部分激活/模块需要设置 inplace 属性以兼容老版本 torch
            m.inplace = inplace  # torch 1.7.0 compatibility
        elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model
    if len(ensemble) == 1:
        return ensemble[-1]

    # Return ensemble（若为多个模型，设置 ensemble 的公共属性）
    LOGGER.info(f'Ensemble created with {weights}\n')
    for k in 'names', 'nc', 'yaml':
        setattr(ensemble, k, getattr(ensemble[0], k))
    ensemble.stride = ensemble[torch.argmax(torch.tensor([m.stride.max() for m in ensemble])).int()].stride
    assert all(ensemble[0].nc == m.nc for m in ensemble), f'Models differ in class counts {[m.nc for m in ensemble]}'
    return ensemble


def attempt_load_one_weight(weight, device=None, inplace=True, fuse=False):
    """Loads a single model weights."""
    """加载单个权重，返回 (model, ckpt)：
    - 与 attempt_load_weights 类似但只处理一个文件，且返回原始 ckpt
    """
    ckpt, weight = torch_safe_load(weight)  # load ckpt
    args = {**DEFAULT_CFG_DICT, **(ckpt.get('train_args', {}))}  # combine model and default args, preferring model args
    model = (ckpt.get('ema') or ckpt['model']).to(device).float()  # FP32 model

    # Model compatibility updates
    model.args = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # attach args to model
    model.pt_path = weight  # attach *.pt file path to model
    model.task = guess_model_task(model)
    if not hasattr(model, 'stride'):
        model.stride = torch.tensor([32.])

    model = model.fuse().eval() if fuse and hasattr(model, 'fuse') else model.eval()  # model in eval mode

    # Module compatibility updates
    for m in model.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Segment):
            m.inplace = inplace  # torch 1.7.0 compatibility
        elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model and ckpt
    return model, ckpt


def parse_model(d, ch, verbose=True):  # model_dict, input_channels(3)
    # Parse a YOLO model.yaml dictionary into a PyTorch model
    import ast

    # Args: 读取 yaml 中的基础参数（nc, activation, scales 等）
    max_channels = float('inf')
    nc, act, scales = (d.get(x) for x in ('nc', 'activation', 'scales'))
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ('depth_multiple', 'width_multiple', 'kpt_shape'))
    if scales:
        scale = d.get('scale')
        if not scale:
            scale = tuple(scales.keys())[0]
            LOGGER.warning(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]

    if act:
        # 将 Conv.default_act 动态设置为 yaml 指定的激活函数（例如 SiLU）
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")  # print

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    # ch 列表用于追踪每层的输出通道数（parse 后续使用）
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    # 遍历 backbone + head 的配置项（每个元素为 (from, n, module, args)）
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        # m 可能是 'nn.Conv2d' 或自定义模块名，全局取对象，如 nn.Conv2d则取出Conv2d，即为m
        m = getattr(torch.nn, m[3:]) if 'nn.' in m else globals()[m]  # get module
        # 解析 args 中以字符串表示的字面量（如 '3' -> 3 或变量名 -> 查本地变量）
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)

        # 根据 depth_multiple 计算重复次数 n
        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        # 针对常见的卷积/结构化模块，重写输入输出通道参数（将 ch[f] 作为输入通道，
        # args[0] 为期望输出通道 c2，c2 需按 width_multiple 做缩放并向上取整满足硬件对齐）
        if m in (Classify, Conv, ConvTranspose, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, Focus,
                 BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x, RepC3):
            c1, c2 = ch[f], args[0]
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                # 按 width_multiple 缩放输出通道并对齐为 8 的倍数
                c2 = make_divisible(min(c2, max_channels) * width, 8)

            args = [c1, c2, *args[1:]]
            # 对于某些 block（CSP 等）需要在 args 中插入重复次数 n 作为参数
            if m in (BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, C3x, RepC3):
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is AIFI:
            args = [ch[f], *args]
        elif m in (HGStem, HGBlock):
            # HourGlass 专用的参数调整：输入、mid、输出通道
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)  # number of repeats
                n = 1

        elif m is nn.BatchNorm2d:
            # BatchNorm2d 的构造参数是通道数 ch[f]
            args = [ch[f]]
        elif m is Concat:
            # Concat 模块的输出通道为参与 concat 的通道总和
            c2 = sum(ch[x] for x in f)
        elif m in (Detect, Segment, Pose, RTDETRDecoder):
            # 对 detection/segment/pose 等 head，需要把前序层的输出通道列表传入 head
            args.append([ch[x] for x in f])
            if m is Segment:
                # 对 segment 特有的 mask 通道数进行按 width 缩放与对齐
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
        elif m is SegmentPose:
            # SegmentPose 同时需要传入来自哪些层的通道，并调整 kpt/seg 通道数
            args.append([ch[x] for x in f])
            args[3] = make_divisible(min(args[3], max_channels) * width, 8)            
        else:
            # 其他模块默认输出通道等于输入层的输出通道
            c2 = ch[f]

        # 实例化模块：若 n > 1 则用 nn.Sequential 包裹重复 n 次，否则直接实例化
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        # 获取模块类型字符串（用于打印）
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        # 计算该模块的参数量 m.np（用于后续 profile 打印）
        m.np = sum(x.numel() for x in m_.parameters())  # number params
        # 为模块附加索引/来源/类型信息（供 _predict_once 使用）
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        if verbose:
            LOGGER.info(f'{i:>3}{str(f):>20}{n_:>3}{m.np:10.0f}  {t:<45}{str(args):<30}')  # print
        # 将需要保存的层索引加入 save（parse_model 返回的 save 用于控制中间输出保留）
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    # 返回构建好的 nn.Sequential 层与排序后的保存索引
    return nn.Sequential(*layers), sorted(save)


def yaml_model_load(path):
    """从 yaml 文件加载模型配置，并为 legacy 名称做兼容处理（例如 -p6 后缀转换）。"""
    import re

    path = Path(path)
    if path.stem in (f'yolov{d}{x}6' for x in 'nsmlx' for d in (5, 8)):
        new_stem = re.sub(r'(\d+)([nslmx])6(.+)?$', r'\1\2-p6\3', path.stem)
        LOGGER.warning(f'WARNING ⚠️ Ultralytics YOLO P6 models now use -p6 suffix. Renaming {path.stem} to {new_stem}.')
        path = path.with_stem(new_stem)

    unified_path = re.sub(r'(\d+)([nslmx])(.+)?$', r'\1\3', str(path))  # i.e. yolov8x.yaml -> yolov8.yaml
    yaml_file = check_yaml(unified_path, hard=False) or check_yaml(path)
    d = yaml_load(yaml_file)  # model dict
    d['scale'] = guess_model_scale(path)
    d['yaml_file'] = str(path)
    return d


def guess_model_scale(model_path):
    """
    Takes a path to a YOLO model's YAML file as input and extracts the size character of the model's scale.
    The function uses regular expression matching to find the pattern of the model scale in the YAML file name,
    which is denoted by n, s, m, l, or x. The function returns the size character of the model scale as a string.

    Args:
        model_path (str | Path): The path to the YOLO model's YAML file.

    Returns:
        (str): The size character of the model's scale, which can be n, s, m, l, or x.
    """
    """
    从 yaml 文件名中猜测模型的 scale（n, s, m, l, x）。
    方便在 parse_model 中使用 scales 字段做自动选择。
    """
    with contextlib.suppress(AttributeError):
        import re
        return re.search(r'yolov\d+([nslmx])', Path(model_path).stem).group(1)  # n, s, m, l, or x
    return ''


def guess_model_task(model):
    """
    Guess the task of a PyTorch model from its architecture or configuration.

    Args:
        model (nn.Module | dict): PyTorch model or model configuration in YAML format.

    Returns:
        (str): Task of the model ('detect', 'segment', 'classify', 'pose').

    Raises:
        SyntaxError: If the task of the model could not be determined.
    """
    """
    猜测模型的任务（detect / segment / classify / pose）：
    - 优先从 yaml/config dict 判定
    - 否则通过遍历 model.modules() 查找 Detect/Segment/Classify/Pose 实例来判定
    - 再不行则根据文件名推断，最终默认 'detect'
    该函数在加载 checkpoint 时用于设置 model.task，便于上层分支处理。
    """

    def cfg2task(cfg):
        """从 yaml 字典猜测 task（读取 head 的模块名）"""
        m = cfg['head'][-1][-2].lower()  # output module name
        if m in ('classify', 'classifier', 'cls', 'fc'):
            return 'classify'
        if m == 'detect':
            return 'detect'
        if m == 'segment':
            return 'segment'
        if m == 'pose':
            return 'pose'

    # Guess from model cfg
    if isinstance(model, dict):
        with contextlib.suppress(Exception):
            return cfg2task(model)

    # Guess from PyTorch model
    if isinstance(model, nn.Module):  # PyTorch model
        for x in 'model.args', 'model.model.args', 'model.model.model.args':
            with contextlib.suppress(Exception):
                return eval(x)['task']
        for x in 'model.yaml', 'model.model.yaml', 'model.model.model.yaml':
            with contextlib.suppress(Exception):
                return cfg2task(eval(x))

        for m in model.modules():
            if isinstance(m, Detect):
                return 'detect'
            elif isinstance(m, Segment):
                return 'segment'
            elif isinstance(m, Classify):
                return 'classify'
            elif isinstance(m, Pose):
                return 'pose'

    # Guess from model filename
    if isinstance(model, (str, Path)):
        model = Path(model)
        if '-seg' in model.stem or 'segment' in model.parts:
            return 'segment'
        elif '-cls' in model.stem or 'classify' in model.parts:
            return 'classify'
        elif '-pose' in model.stem or 'pose' in model.parts:
            return 'pose'
        elif 'detect' in model.parts:
            return 'detect'

    # Unable to determine task from model
    LOGGER.warning("WARNING ⚠️ Unable to automatically guess model task, assuming 'task=detect'. "
                   "Explicitly define task for your model, i.e. 'task=detect', 'segment', 'classify', or 'pose'.")
    return 'detect'  # assume detect

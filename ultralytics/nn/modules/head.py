# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Model head modules

此文件实现多种 head（检测/分割/关键点/联合）：
- Detect: 基础检测头，输出 bbox 分布回归 + 类别 logits，支持 DFL 分布回归。
- Segment: 在 Detect 基础上增加 proto（原型 mask）与 mask coefficients，用于实例分割。
- Pose: 在 Detect 基础上增加 keypoint 分支并实现解码。
- SegmentPose: 同时包含 Segment 和 Pose 的分支，在单一前向过程中输出 bbox/class + mask coeff + keypoints。
此外还有简单的 Classify 和一个基于 Deformable Transformer 的 RTDETRDecoder（query-based decoder）。
"""

import math

import torch
import torch.nn as nn
from torch.nn.init import constant_, xavier_uniform_

# 工具函数：dist2bbox 用于把分布/偏移转换为 bbox，make_anchors 用于构造网格 anchor（中心坐标）
from ultralytics.yolo.utils.tal import dist2bbox, make_anchors
# make_anchors：根据输出特征图大小和 stride 生成网格锚点（anchor 中心与 stride）
# dist2bbox：将 DFL 的分布表示或相对量转成真正的框（xywh）

# 局部模块：DFL（distribution focal loss / distribution based regression）、Proto（mask 原型网络）
# DFL：distribution focal layer，将 reg_max 的 logits转为偏移量（通常先 softmax 再期望值）。传统 bbox 回归直接回归 4 个实数。DFL 用将每个坐标分成多个离散 bin（reg_max）学习一个概率分布，再通过期望或类似方法得到连续值，能提高回归精度与稳定性（尤其对小物体或高精度需求有帮助）
from .block import DFL, Proto
# Conv 是封装的卷积模块（含 BN/激活等）
from .conv import Conv
# Transformer 相关简化模块（用于 RTDETRDecoder）
from .transformer import MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer
# 初始化辅助
from .utils import bias_init_with_prob, linear_init_

__all__ = 'Detect', 'Segment', 'Pose', 'Classify', 'RTDETRDecoder'


class Detect(nn.Module):
    """YOLOv8 Detect head for detection models.

    功能总结：
    - 每个特征尺度通过两个分支输出：box 分布 (4 * reg_max) 和类别 logits (nc)。
    - 使用 DFL 将离散分布转换为连续偏移（如果 reg_max > 1）。
    - 在训练时直接返回各尺度的原始 logits（便于 loss 计算）。
    - 在推理时构造 anchors/strides、将输出拼接、对 box 部分做 dfl->bbox 解码并拼接 sigmoid 后的类别概率返回。
    """

    dynamic = False  # 是否强制在每次前向重建 grid（anchors），用于动态图场景
    export = False  # 导出模型（tflite/tfjs/saved_model）时的一些特殊处理标志
    shape = None
    anchors = torch.empty(0)  # anchors 占位（在第一次推理或调用 make_anchors 时填充）
    strides = torch.empty(0)  # strides 占位

    def __init__(self, nc=80, ch=()):  # detection layer
        super().__init__()
        self.nc = nc  # 类别数量
        self.nl = len(ch)  # detection 层数（通常等于特征金字塔层数，例：3）
        self.reg_max = 16  # DFL 的离散 bins 数（每个 bbox coordinate 被离散化为 reg_max 个 bin），用于更细粒度的回归表示（distribution focal loss 相关）
        # no: 每个 anchor 的输出通道数 = 类别数 + 4 个坐标 * reg_max（box 使用分布回归）
        self.no = nc + self.reg_max * 4
        # stride 暂时用 zeros 占位，会在 build/make_anchors 时被更新
        self.stride = torch.zeros(self.nl)

        # 计算中间通道数以保持合理容量（经验值）
        # c2 用于 box 分支的中间通道；c3 用于类别分支中间通道
        c2 = max((16, ch[0] // 4, self.reg_max * 4))
        c3 = max(ch[0], min(self.nc, 100))

        # 为每个尺度创建 box 分支序列：Conv -> Conv -> Conv(out_channels=4*reg_max)
        # 最后一层不带 activation，直接输出 raw logits / distribution logits
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch) # box 分支，每尺度最后 conv 产生 4*reg_max 通道

        # 为每个尺度创建 cls 分支序列：Conv -> Conv -> Conv(out_channels=nc)
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch) # 类别分支

        # 如果 reg_max>1 使用 DFL 将离散 logits 转为偏移值（期望），否则 Identity
        # DFL 的核心思想是让网络预测每个坐标的离散分布（比直接回归更稳定/精细）
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities.

        参数 x: list of feature maps，长度 = self.nl，每项形状为 (B, C_i, H_i, W_i)
        返回：
        - training: 返回 list，每个元素为该尺度拼接后的 logits tensor (B, no, H_i, W_i) 便于 loss 计算
        - eval: 返回 (y, x) 或 y，y 为解码后的预测 (B, 4+nc, sum(H_i*W_i))（包含 bbox(xywh) + class_prob）
        """

        shape = x[0].shape  # 取第一个尺度的 BCHW 用来检测 batch size 和形状变化
        # 对每个尺度，将 box 分支和 cls 分支的输出在 channel 维拼接
        # box: (B, 4*reg_max, H, W), cls: (B, nc, H, W) -> concat => (B, no, H, W)
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        # 训练模式直接返回原始 logits（loss 计算需要 raw logits/分布）
        if self.training:
            return x
        # 推理模式：如果 dynamic 或 当前输入 shape 变化，则重新生成 anchors 和 strides
        elif self.dynamic or self.shape != shape:
            # make_anchors 根据各尺度 feature map 尺寸以及 stride（占位）构建 anchor 网格
            # 返回 anchors 和 strides，每个是列表（按尺度）。YOLOv8对每个 grid cell 直接预测一套输出（距离/偏移、类别等），但是要把这些相对值映射回图像坐标，仍然需要一个参考点（grid center）和 stride，这个参考点在代码中常被称为 anchors 或 grids
            # 这里使用生成器表达式转置张量形状以匹配后边使用的格式
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5)) # anchors = 特征图上每个格点的参考坐标（通常是归一化的中心坐标 x, y，有时也带一个基准 wh），配套还有对应的 stride（下采样倍数）。它们只是解码网络“相对预测”到图像坐标的参考点，不是预设的多尺度、多纵横比锚框
            self.shape = shape

        # 把每个尺度展平并在空间维度上拼接（concatenate across scales）
        # xi.view(shape[0], self.no, -1) -> (B, no, H_i*W_i)
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)

        # export 到某些 TF 相关格式时需要避免某些算子，故对 box/cls 的切分方式做兼容处理
        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            # 标准切分：box 部分占前 reg_max*4 个通道，其余为类别 logits
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        # 将 box 的分布 logits 通过 DFL 转为连续值偏移（相对于 anchor 网格）
        # dist2bbox：把偏移与 anchors 结合并返回 xywh（或其他格式）；
        # 注意这里对 anchors 做了 unsqueeze(0) 以匹配 batch 维度
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        # 最终拼接：bbox(xywh, 已乘 stride => 像素尺度) + 类别概率(sigmoid)
        y = torch.cat((dbox, cls.sigmoid()), 1)
        # export 时返回 y，否则返回 (y, x) 其中 x 为原始 logits 列表，便于后续 loss 或进一步处理
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability.

        目的：初始化偏置项使得训练初期类别置信度和 bbox 回归稳定。
        - 对 box 分支偏置设为 1.0（经验值，使网络一开始能预测较大 bbox 值）
        - 对 cls 分支偏置设为一个与 stride 相关的 logit（估计图片中目标概率）
        注：需要事先知道每层的 stride（通常在 build 或第一次 forward 时通过 make_anchors 填充 self.stride）。
        """
        m = self  # 简写
        # 以下循环按尺度更新最后一层 conv 的 bias
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # a: box branch seq, b: cls branch seq, s: stride scalar
            a[-1].bias.data[:] = 1.0  # 将 box 分支最后 conv 的 bias 设置为 1
            # 类别偏置：使用经验公式 log(5 / nc / (640 / s)^2)
            # 目的是基于 stride 与图像大小估计初始物体密度，使得初始 sigmoid 概率接近合理值
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)


class Segment(Detect):
    """YOLOv8 Segment head for segmentation models.

    扩展 Detect：
    - 增加 Proto 网络生成原型 masks（p），通常分辨率较高；proto 提供基底 mask 图（通常通道数 npr，每个通道是一张基础 mask），每个实例的最终 mask 通过 mask coefficients 与 proto 做线性组合（后接 sigmoid/threshold）得到
    - 增加每个预测的 mask coefficients（mc），通过线性组合 proto 得到实例 mask。
    - 这种设计把 mask 的分辨率开销从每个预测的独立输出降到共享 proto + 低维 coeffs，更节省计算与参数
    设计亮点：通过共享 proto（高分辨率特征的多个通道）+ 每实例低维系数，避免为每个实例输出完整分辨率 mask，节省计算与内存。
    """

    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        """Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers."""
        super().__init__(nc, ch)
        self.nm = nm  # mask 数量（mask coefficients 的数量）
        self.npr = npr  # proto 的通道数（即 proto 基底数量）
        # Proto 网络：从第一个尺度的特征生成 proto 原型 masks，输出形状 (B, npr, H_p, W_p)
        self.proto = Proto(ch[0], self.npr, self.nm)  # proto 网络，通常从高分辨率特征（第一个尺度）生成 Prototype mask（p）
        # 保存 Detect.forward 的方法引用，方便复用父类的预测逻辑（不覆盖父类 forward）
        self.detect = Detect.forward

        # cv4 用来产生 mask coefficients（每个尺度一套分支）
        # c4 是分支的中间通道数，至少为 nm
        c4 = max(ch[0] // 4, self.nm) # 每个尺度生成 mask coefficients 的卷积分支（最后输出 nm 通道），用于给每个预测生成对应的 mask coefficients（线性组合 proto）
        # 每个尺度输出层为 Conv(...)->Conv(...)->Conv(out_channels=nm)
        self.cv4 = nn.ModuleList(
            nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)

    def forward(self, x):
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        # 1) 生成 proto 原型 masks（高分辨率），形状 p: (B, npr, Hp, Wp)
        p = self.proto(x[0])  # mask protos，从第一个尺度生成 proto（通常是一个较高分辨率的特征图，最终用线性组合生成 masks）
        bs = p.shape[0]  # batch size

        # 2) mask coefficients: 对每个尺度的特征跑 cv4 分支，然后 view 到 (B, nm, H_i*W_i)，在尺度维度 concat => (B, nm, sum(H_i*W_i))
        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients，reshape为 (bs, nm, -1) 把每个预测的 mask coefficients 按空间拼接

        # 3) 使用 detect 的 forward 得到检测输出（原始 logits 或解码后的 y）
        x = self.detect(self, x) # 调用 Detect.forward 来获取检测部分输出
        if self.training:
            # 训练时返回：检测输出 x（logits 列表）、mask coefficients mc、proto p
            # loss 端会使用这些量计算 mask losses 等
            return x, mc, p
        # 推理/导出时，返回格式不同以配合后续推理流水线：
        # - export 时通常要把 mc 并到检测输出通道维，便于导出为单张 tensor
        # - 非 export 时返回 (concatenated, (x[1], mc, p)) 其中 x[1] 是原始 logits 列表
        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))


class SegmentPose(Detect):
    """YOLOv8 SegmentPose head for segmentation-pose models.
    关系：SegmentPose 是对 Detect 的扩展，兼顾 segmentation 和 pose 两个任务；相当于在同一个检测结构上并行增加两类额外输出 branch（mask coeff 与 keypoint logits），使模型可在单次前向里同时预测 bbox/class、mask、keypoint

    综合 Segment 与 Pose：
    - 集成了 Segment 的 proto + mask coefficient 分支（cv4 + proto）和 Pose 的关键点分支（cv5）
    - cv4：输出 nm 个 mask coefficients（和 Segment 一样）；cv5：输出 nk 个 keypoint 通道（和 Pose 的 cv4 类似，但放在 cv5）
    - 具有 proto + mask coefficients 分支（用于实例分割）
    - 同时具有 keypoint 分支（用于人体/物体关键点）
    - 训练时返回原始 logits + mc + p + kpt（kpt 为原始未解码的 keypoint predictions）
    - 推理时对 keypoints 做解码，返回 decoded keypoints 以便后续可视化或 NMS 合并
    设计动机：在一次前向中同时预测 bbox/class、mask、keypoints，共享 backbone/neck 特征，节省计算并保证任务输出对齐（anchor/grid 对齐）。
    """

    def __init__(self, nc=80, kpt_shape=(17, 3), nm=32, npr=256, ch=()):
        """Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers."""
        super().__init__(nc, ch)
        # Segment 部分参数
        self.nm = nm  # mask coefficients 数量
        self.npr = npr  # proto 通道数
        self.proto = Proto(ch[0], self.npr, self.nm)  # proto 网络
        self.detect = Detect.forward

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(
            nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)

        # Pose 部分参数
        self.kpt_shape = kpt_shape  # (num_kpts, dims) dims 通常为 2(x,y) 或 3(x,y,visibility/score)
        self.nk = kpt_shape[0] * kpt_shape[1]  # 总输出通道数（每个 keypoint 有 ndim 个通道）
        c5 = max(ch[0] // 4, self.nk)
        # cv5 为 keypoint 分支，为每个尺度产生 nk 通道的 map
        self.cv5 = nn.ModuleList(
            nn.Sequential(Conv(x, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, self.nk, 1)) for x in ch)

    def forward(self, x):
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        # 1) proto
        p = self.proto(x[0])  # mask protos (B, npr, Hp, Wp)
        bs = p.shape[0]  # batch size

        # 2) mask coefficients，同 Segment：(B, nm, sum(H_i*W_i))
        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients

        # 3) keypoints：对每个尺度取 cv5 输出 -> reshape concat -> (B, nk, sum(H_i*W_i))
        kpt = torch.cat([self.cv5[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)  # (bs, nk, h*w)

        # 4) 调用 Detect 的 forward 得到检测输出
        x = self.detect(self, x)
        if self.training:
            # 训练：返回原始检测 logits 列表 x、mc、proto p、以及原始 keypoint logits kpt（后续 loss 用）
            return x, mc, p, kpt

        # 推理：需要对 keypoint 做解码（从网格/anchor 相对值 -> 图像尺度坐标）
        pred_kpt = self.kpts_decode(bs, kpt)
        # 推理时输出格式：
        # - export: 把 x 和 mc 以及 pred_kpt 在 channel 维拼接，然后返回 proto p（导出需求）
        # - 正常推理: 返回 (concat(x[0], mc, pred_kpt), (x[1], mc, p, kpt))，其中 x[1] 为原始 logits 列表
        return (torch.cat([x, mc, pred_kpt], 1), p) if self.export else (torch.cat([x[0], mc, pred_kpt], 1), (x[1], mc, p, kpt))
    
    def kpts_decode(self, bs, kpts):
        """Decodes keypoints from raw head output to image-scale coordinates.

        输入 kpts: (B, nk, N) 其中 nk = num_kpts * ndim, N = sum(H_i*W_i)
        返回: 与 prediction 对齐的 decoded keypoints，仍然以通道优先 (B, nk, N)

        解码逻辑要点：
        - 当 ndim == 3 时，第 3 个通道通常表示可见性/置信度 -> 对该通道做 sigmoid。
        - 对 x,y 通道按网格 anchors 与 strides 做线性变换：
            y_x = (raw * 2.0 + (anchor_coord - 0.5)) * stride
          这里 raw * 2.0 扩展了网络输出范围（常见技巧，配合 sigmoid/原始值使用），anchor_coord 是网格坐标（0..W-1 / H-1）
          anchor_coord - 0.5 调整中心偏移，使预测相对于像素坐标正确。
        - self.anchors, self.strides 是在 Detect.forward 中由 make_anchors 填充的，全局共享。
        """
        ndim = self.kpt_shape[1]
        # 直接复制 tensor 避免覆盖原始 kpts（后续训练时仍需原始 kpt）
        y = kpts.clone()
        if ndim == 3:
            # 对每个关键点的 visibility/score 通道做 inplace sigmoid，索引为 2::3（假设 ndim==3）
            y[:, 2::3].sigmoid_()  # inplace sigmoid
        # x 通道索引为 0::ndim，y 通道索引为 1::ndim
        # anchors 和 strides 的形状与 (2, N) 或 (N,) 等有关，具体在 make_anchors 中生成
        # 将网络输出变换到图像尺度
        y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
        y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
        return y


class Pose(Detect):
    """YOLOv8 Pose head for keypoints models.

    在 Detect 基础上增加关键点分支 cv4，并实现 kpts 解码。与 SegmentPose 区别在于 Pose 只负责 keypoints（不包含 mask/proto）。
    """

    def __init__(self, nc=80, kpt_shape=(17, 3), ch=()):
        """Initialize YOLO network with default parameters and Convolutional Layers."""
        super().__init__(nc, ch)
        self.kpt_shape = kpt_shape  # (num_kpts, ndim)
        self.nk = kpt_shape[0] * kpt_shape[1]  # 输出通道数，最终每尺度最后 conv 输出 nk 通道
        self.detect = Detect.forward  # 复用父类 Detect 的检测逻辑（bbox + cls 部分）

        # 中间通道数 c4 至少为 nk
        c4 = max(ch[0] // 4, self.nk)
        # cv4 为每个尺度的 keypoint 分支：Conv -> Conv -> Conv(out_channels=nk)
        self.cv4 = nn.ModuleList(
            nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)

    def forward(self, x):
        """Perform forward pass through YOLO model and return predictions.

        x: list of feature maps
        返回：
        - training: (x, kpt) 其中 x 为 detect 的 logits 列表，kpt 为原始 keypoint logits
        - eval: 返回 (concatenated preds, (x[1], kpt)) 或导出时 self.export 兼容格式
        """
        bs = x[0].shape[0]  # batch size
        # 将每个尺度 cv4 输出 reshape 后 concat -> (bs, nk, sum(H_i*W_i))
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)  # (bs, nk, h*w)
        # 调用 Detect.forward 得到检测输出
        x = self.detect(self, x)
        if self.training:
            # 训练：返回 detect logits 列表 与 keypoint 原始 logits（用于 loss）
            return x, kpt
        # 推理时对 keypoint 做解码（得到图像尺度坐标）
        pred_kpt = self.kpts_decode(bs, kpt)
        # export 时返回不同格式以兼容导出工具，否则返回 (concat(x[0], pred_kpt), (x[1], kpt))
        return torch.cat([x, pred_kpt], 1) if self.export else (torch.cat([x[0], pred_kpt], 1), (x[1], kpt))

    def kpts_decode(self, bs, kpts):
        """Decodes keypoints.

        与 SegmentPose.kpts_decode 相似，但在 export 路径中 shape 处理略有不同以规避导出 bug。
        """
        ndim = self.kpt_shape[1]
        if self.export:  # required for TFLite export to avoid 'PLACEHOLDER_FOR_GREATER_OP_CODES' bug
            # export 情况下，将通道重组为 (bs, num_kpts, ndim, N)
            y = kpts.view(bs, *self.kpt_shape, -1)
            # 计算 x,y: (y[:, :, :2] * 2.0 + (anchors - 0.5)) * strides
            # 注意此处 self.anchors 的形状是针对 keypoint 解码 export 特殊路径的预期格式
            a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            # 若有可见性通道则拼接 sigmoid 后的可见性
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            # 重新 view 为 (bs, nk, N)
            return a.view(bs, self.nk, -1)
        else:
            # 非导出路径：使用 inplace sigmoid + 基于 anchors/strides 的线性映射（与 SegmentPose 一致）
            y = kpts.clone()
            if ndim == 3:
                y[:, 2::3].sigmoid_()  # inplace sigmoid for visibility
            y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
            return y


class Classify(nn.Module):
    """YOLOv8 classification head, i.e. x(b,c1,20,20) to x(b,c2).

    简单的分类头：先 conv -> 全局池化 -> dropout -> linear 输出类别概率（训练返回 logits，推理返回 softmax）。
    常用于 image-level 分类而非检测分支。
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        c_ = 1280  # 按 EfficientNet-B0 的中间通道规模选择
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # 全局池化到 (B, c_, 1, 1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)  # fc -> c2

    def forward(self, x):
        """Performs a forward pass of the YOLO model on input image data."""
        if isinstance(x, list):
            # 若传入是 list（多尺度特征），则在 channel 维度拼接（非常少见的用法）
            x = torch.cat(x, 1)
        # conv -> pool -> flatten -> dropout -> linear
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        # 训练返回 logits（供 loss），推理返回 softmax 概率分布
        return x if self.training else x.softmax(1)


class RTDETRDecoder(nn.Module):
    """基于 Deformable Transformer 的简化 DETR 解码器模块（query-based detection）。

    该类与上面的 YOLO-style heads 不同，属于 query-based 检测器的 decoder 实现示例。
    主要包含：
    - 特征投影 input_proj，将 backbone 多尺度特征投影到相同 hidden dim
    - Deformable Transformer decoder（使用 DeformableTransformerDecoderLayer）
    - Encoder-side head（给 encoder 输出做初步候选并选 topk 作为 queries）
    - Denoising training 支持（query 前添加噪声样例以稳定训练）
    注意：此实现为简化版本，不完全与官方 checkpoint 权重直接对应，但用于展示 QUERY 解码器逻辑。
    """
    export = False  # export mode

    def __init__(
            self,
            nc=80,
            ch=(512, 1024, 2048),
            hd=256,  # hidden dim
            nq=300,  # num queries
            ndp=4,  # num decoder points
            nh=8,  # num head
            ndl=6,  # num decoder layers
            d_ffn=1024,  # dim of feedforward
            dropout=0.,
            act=nn.ReLU(),
            eval_idx=-1,
            # training args
            nd=100,  # num denoising
            label_noise_ratio=0.5,
            box_noise_scale=1.0,
            learnt_init_query=False):
        super().__init__()
        self.hidden_dim = hd
        self.nhead = nh
        self.nl = len(ch)  # num levels (backbone feature levels)
        self.nc = nc
        self.num_queries = nq
        self.num_decoder_layers = ndl

        # backbone feature projection -> 将每个尺度的通道投影到相同 hidden dim（便于 transformer 处理）
        self.input_proj = nn.ModuleList(
            nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch)
        # NOTE: 注释中提到的 Conv 版本被简化为上述实现以兼容部分权重

        # Transformer decoder 层与整体 decoder（deformable attention）
        decoder_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        self.decoder = DeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx)

        # denoising embedding，用于 denoising training 框架（CDN）
        self.denoising_class_embed = nn.Embedding(nc, hd)
        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # decoder query 初始化（可以选择学习型或动态从 encoder 输出选取）
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)
        # 根据 bbox (4-dim) 生成 query position embedding 的 head（MLP）
        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)

        # encoder 侧 head：用于从 encoder 输出中选取 top-K 候选 query
        self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))
        self.enc_score_head = nn.Linear(hd, nc)
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)

        # decoder 侧 head：每层 decoder 都有独立的分类/回归 head（用于 deep supervision）
        self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])

        self._reset_parameters()

    def forward(self, x, batch=None):
        from ultralytics.vit.utils.ops import get_cdn_group

        # 1) input projection + embedding
        feats, shapes = self._get_encoder_input(x)

        # 2) denoising training 准备（如果训练且使用 CDN）
        dn_embed, dn_bbox, attn_mask, dn_meta = \
            get_cdn_group(batch,
                          self.nc,
                          self.num_queries,
                          self.denoising_class_embed.weight,
                          self.num_denoising,
                          self.label_noise_ratio,
                          self.box_noise_scale,
                          self.training)

        # 3) 获取 decoder input（embeddings, reference bboxes 等）
        embed, refer_bbox, enc_bboxes, enc_scores = \
            self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)

        # 4) decoder forward
        dec_bboxes, dec_scores = self.decoder(embed,
                                              refer_bbox,
                                              feats,
                                              shapes,
                                              self.dec_bbox_head,
                                              self.dec_score_head,
                                              self.query_pos_head,
                                              attn_mask=attn_mask)
        # 返回结构：decoder 输出的 bbox/score 和 encoder-side 的 bbox/score 以及 denoising 元信息
        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        if self.training:
            return x
        # 推理：拼接 decoder 最后一层 bbox 与 sigmoid 后的 score
        # (bs, 300, 4+nc)
        y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1)
        return y if self.export else (y, x)

    def _generate_anchors(self, shapes, grid_size=0.05, dtype=torch.float32, device='cpu', eps=1e-2):
        """为 Deformable DETR 风格生成 anchors（参考框），返回 anchors（logit 空间）和 valid_mask.
        基于 Deformable Transformer 的检测解码器（与 YOLO 系列不同风格），包含 encoder->decoder 流程、anchor 生成、denoising training 机制、decoder heads。文件中此类是一个完整的 transformer decoder head 的简化实现，非 YOLO 的主干部分，但放在同文件以便复用/替代分析。主要用于 query-based 检测（如 DETR 变体）的任务

        shapes: list of [h, w] 每个尺度的空间尺寸
        grid_size: 基础尺度（anchor 的 wh）
        主要流程：
        - 生成网格坐标 grid_xy（0..w-1, 0..h-1），归一化到 (0,1) 并加 0.5 偏移使中心对齐
        - wh 根据尺度以倍数增长
        - anchors shape (1, h*w*nl, 4)，最后对数化 (logit 空间) 以便后续与网络输出相加
        - valid_mask: 标记那些 anchors 在 (eps, 1-eps) 区间内的合法性（避免边界值）
        """
        anchors = []
        for i, (h, w) in enumerate(shapes):
            # meshgrid 产生 y,x 网格
            grid_y, grid_x = torch.meshgrid(torch.arange(end=h, dtype=dtype, device=device),
                                            torch.arange(end=w, dtype=dtype, device=device),
                                            indexing='ij')
            grid_xy = torch.stack([grid_x, grid_y], -1)  # (h, w, 2) 按 (x, y)

            valid_WH = torch.tensor([h, w], dtype=dtype, device=device)
            # 归一化为 (0..1)，并加上 0.5/valid_WH 用于中心对齐
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH  # (1, h, w, 2)
            # wh 按尺度缩放，基础为 grid_size * (2^i)
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0 ** i)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))  # (1, h*w, 4)

        anchors = torch.cat(anchors, 1)  # (1, h*w*nl, 4)
        # valid mask: 如果 anchors 的所有维度都在 (eps, 1-eps) 内则为有效
        valid_mask = ((anchors > eps) * (anchors < 1 - eps)).all(-1, keepdim=True)  # 1, h*w*nl, 1
        # 将 anchors 转为 logit 空间（log(x/(1-x)))，方便后续与网络输出相加/回归
        anchors = torch.log(anchors / (1 - anchors))
        # 非有效位置填充为 inf（后续会被屏蔽）
        anchors = anchors.masked_fill(~valid_mask, float('inf'))
        return anchors, valid_mask

    def _get_encoder_input(self, x):
        # get projection features
        # 将每个尺度的特征投影到 hidden dim 并展开为 [b, h*w, c]
        x = [self.input_proj[i](feat) for i, feat in enumerate(x)]
        feats = []
        shapes = []
        for feat in x:
            h, w = feat.shape[2:]
            # [b, c, h, w] -> [b, h*w, c]
            feats.append(feat.flatten(2).permute(0, 2, 1))
            # 记录每个尺度的 (h,w)
            shapes.append([h, w])

        # 将不同尺度的 flattened features 在空间维度拼接 -> [b, sum(h*w), c]
        feats = torch.cat(feats, 1)
        return feats, shapes

    def _get_decoder_input(self, feats, shapes, dn_embed=None, dn_bbox=None):
        """从 encoder 输出中选取 topk anchors 作为 decoder 的 query reference（参考框），并准备 embeddings。

        返回：
        - embeddings: decoder 的初始输入（可包含 denoising 的 embed）
        - refer_bbox: 用于 decoder cross-attention 的参考 bbox（未 sigmoid）
        - enc_bboxes: encoder side 的 bbox（sigmoid 后）
        - enc_scores: encoder 侧的分类 logits（未 sigmoid）
        """
        bs = len(feats)
        # 生成 anchors（logit 空间）和 valid_mask
        anchors, valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
        # 对 encoder features 做一个线性 layer + layernorm（enc_output），并乘以 valid_mask 避免无效位置影响
        features = self.enc_output(valid_mask * feats)  # bs, h*w, 256

        # encoder-side 的分类和回归输出
        enc_outputs_scores = self.enc_score_head(features)  # (bs, h*w, nc)
        enc_outputs_bboxes = self.enc_bbox_head(features) + anchors  # (bs, h*w, 4)  (net输出+anchors(logit))

        # query selection: 按 encoder 预测的最大 class score 选 topk positions
        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)

        # 从 enc_outputs_bboxes 中按选中的 index 取出 refer_bbox（未 sigmoid）
        refer_bbox = enc_outputs_bboxes[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        enc_bboxes = refer_bbox.sigmoid()  # encoder 输出的 bbox（可用于训练监督）
        if dn_bbox is not None:
            # 若使用 denoising，则把 dn_bbox 拼接在 refer_bbox 前面
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)
        if self.training:
            # 在训练中 detach refer_bbox 以阻断梯度流回 encoder（经验做法）
            refer_bbox = refer_bbox.detach()
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        # 构造 decoder 的 embeddings：可以使用 learnable query 或 encoder 中对应 positions 的 features
        if self.learnt_init_query:
            embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
        else:
            embeddings = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)
            if self.training:
                # 同样 detach embeddings 在训练阶段以避免 encoder 更新时产生不稳定
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)

        return embeddings, refer_bbox, enc_bboxes, enc_scores

    # TODO
    def _reset_parameters(self):
        """初始化各 head 的权重与偏置，部分采用常数初始化以便训练稳定。"""
        # 初始化类别偏置使得初始 sigmoid 概率接近某个小值（bias_init_with_prob 生成一个建议值）
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        # encoder head 初始化
        # 使用常数初始化分类 bias
        constant_(self.enc_score_head.bias, bias_cls)
        # bbox head 最后一层 weight/bias 初始化为 0
        constant_(self.enc_bbox_head.layers[-1].weight, 0.)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.)
        # decoder heads 初始化
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.)
            constant_(reg_.layers[-1].bias, 0.)

        # 线性层与 embedding 初始化（xavier）
        linear_init_(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)
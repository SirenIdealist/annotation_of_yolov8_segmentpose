# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F

# OKS_SIGMA: 在姿态关键点损失中用来对不同关键点做不同尺度敏感性权重（COCO 评估中使用的 sigma）
from ultralytics.yolo.utils.metrics import OKS_SIGMA

# 若干工具函数：用于 mask 裁剪、坐标互换、assigner/anchor/距离计算等
from ultralytics.yolo.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.yolo.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors

from .metrics import bbox_iou
from .tal import bbox2dist


class VarifocalLoss(nn.Module):
    """
    Varifocal loss：
    - 论文：Varifocal Loss (Zhang et al.)，用于改进检测器中置信度与类别得分的训练目标。
    - 设计动机：让高质量（高 IoU）真阳性得到更高的训练权重，从而提升排序与置信估计质量。
    - 本实现：对 BCEWithLogits 结果乘以一个权重，权重由预测概率、gt_score（质量软标签）和 label 决定。
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        """
        pred_score: logits（未 sigmoid）
        gt_score: 软标签（例如 IoU）——用于对正样本加权
        label: 0/1 二值标签
        alpha/gamma: focal 样式的缩放参数
        计算步骤：
          weight = alpha * p^gamma * (1 - label) + gt_score * label
          loss = BCEWithLogits(pred_score, gt_score) * weight
        说明：对于正样本使用 gt_score 直接作为权重；对负样本使用由预测概率调制的权重（类似 focal）。
        """
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        # 在混合精度下确保 stable（BCEWithLogits 用 float32 更稳定）
        with torch.cuda.amp.autocast(enabled=False):
            loss = (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction='none') *
                    weight).mean(1).sum()
        return loss


class FocalLoss(nn.Module):
    """
    Focal loss wrapper（用于二分类 / 每-anchor 多类二选一场景）：
    - 在原始 BCE 上乘上 modulating factor (1 - p_t)^gamma 来降低易分类样本的权重。
    - alpha 用来做类别不平衡的 re-weight。
    """
    def __init__(self, ):
        super().__init__()

    def forward(self, pred, label, gamma=1.5, alpha=0.25):
        """
        pred: logits
        label: 0/1
        返回：对 batch 的 sum（或可按需要 mean）
        实现采用 TF Addons 实现的计算方式（数值稳定性好）。
        """
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction='none')
        pred_prob = pred.sigmoid()  # p
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class BboxLoss(nn.Module):
    """
    边界框损失容器（IoU loss + 可选 DFL）:
    - IoU loss: 使用 CIoU（考虑中心/尺度/纵横比），衡量预测框与 gt 框的重合与几何差异。
    - DFL (Distribution Focal Loss): 将边距回归视为离散分布并做分类式 loss，再用期望恢复连续值，提高定位精度。
    - reg_max: 控制离散分布的 bin 数（reg_max + 1 个类别）
    对于预测检测框中心点所属grid_cell距离该grid_cell左上角点的距离时，会根据reg_max这个参数将该grid_cell的边划分为(reg_max + 1)个bin，然后预测该中心点在这些bin区间的取值概率，最终通过求期望的形式最终确定检测框中心点距离grid_cell左上角点的offset
    
    DFL 的设计动机：解决离散化坐标的精度损失
    传统目标检测（如 YOLOv5）对边界框坐标（如中心点偏移量、宽高）的预测采用 “离散化锚点 + 回归修正” 策略：
        - 将坐标范围划分为若干离散区间（如 0~1 分为 10 个区间），模型预测每个区间的概率；
        - 最终坐标通过 “区间索引 × 步长 + 偏移量” 计算。
    这种方式存在缺陷：
        - 离散区间划分导致 “量化误差”，预测坐标难以精确匹配真实值；
        - 模型仅输出单一区间的概率，忽略了相邻区间的相关性（如真实值可能位于两个区间的交界处）。
    DFL 通过将坐标预测视为 “离散概率分布”，让模型学习坐标在各区间的概率分布，再通过分布的 “期望” 计算最终坐标，从而缓解量化误差。
    """
    def __init__(self, reg_max, use_dfl=False):
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """
        主要步骤：
        - 对前景（fg_mask）样本计算加权 IoU 损失：loss_iou = sum((1 - IoU) * weight) / normalizer
        - 如果启用 DFL: 计算 pred_dist 与 target 分布的交叉熵插值损失并归一化
        参数说明：
            pred_dist: 网络输出的回归分布（或直接回归值）
            pred_bboxes: 解码得到的预测框（xyxy）
            anchor_points: 每个预测对应的 grid 中心点（用于 bbox <-> dist 的转换）
            target_bboxes: 分配到正样本的 target（在 anchor/grid 参考系）
            target_scores: assigner 给出的分数（软标签）
            target_scores_sum: 正样本总分（用于归一化）
            fg_mask: 前景掩码（哪些预测被视为正样本）
        返回: loss_iou, loss_dfl
        """
        # weight: 对每个正样本按照 target_scores 的和来加权（使更高质量样本贡献更大）
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        # CIoU 距离（越大越差）
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # 若启用了 DFL，则计算 dfl 损失
        if self.use_dfl:
            # bbox2dist: 把 gt bbox 转为每个边界的离散分布目标
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        """
        Distribution Focal Loss (DFL):
        - 将 target 表示为实数位置（非整数 bin），用左右两个整数 bin 的交叉熵加权逼近（线性插值）。
        - tl = floor(target), tr = tl + 1, weight = 分别为距离比例
        """
        tl = target.long()
        tr = tl + 1
        wl = tr - target
        wr = 1 - wl
        return (F.cross_entropy(pred_dist, tl.view(-1), reduction='none').view(tl.shape) * wl +
                F.cross_entropy(pred_dist, tr.view(-1), reduction='none').view(tl.shape) * wr).mean(-1, keepdim=True)


class KeypointLoss(nn.Module):
    """
    关键点检测的特殊性决定了 KeypointLoss 不能直接复用普通回归任务的损失（如纯 MSE），需解决三个核心问题：
    - `坐标尺度一致性`：关键点坐标依赖图像分辨率（如 640×480 图像中，x 范围 0-640，y 范围 0-480），需避免尺度差异导致损失偏倚；
    - `可见性处理`：真实标注中常存在「关键点不可见」（如被遮挡的手肘），若将不可见关键点计入损失，会误导模型学习错误信号；
    - `关键点重要性差异`：不同关键点的语义重要性不同（如人脸中 “眼睛” 比 “脸颊” 更重要），需支持对关键点位赋予更高权重。
    
    基于以上问题，KeypointLoss 的核心设计原理可概括为：
    以「坐标回归损失」为基础，通过「可见性掩码」过滤无效标注，通过「权重机制」区分关键点重要性，最终实现精准且鲁棒的关键点定位优化。

    
    姿态关键点位置损失：
    - 使用欧式距离并结合 OKS 风格的 sigma 和目标面积做归一化，
      采用 (1 - exp(-e)) 形式降低极端距离影响（参考 COCO/OKS 公式）。
    """
    def __init__(self, sigmas) -> None:
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        """
        pred_kpts: (n, k, 2) 或 (n, k, 3)（若包含可见性/置信）
            - n：正样本数量（如前景锚点个数）
            - k：关键点数量（如 COCO数据集人体的17个关键点）
            - 2：(x, y) 坐标；3：额外包含可见性置信度
        gt_kpts: (n, k, 3) - 最后一个维度通常为 (x, y, v)
        kpt_mask: 用于可见性处理。(n, k) 非零表示该点存在/标注。二值掩码，非零值表示该关键点「有效 / 可见」，零值表示无效
        area: 每个正样本对应的目标面积（如边界框面积），用于保证坐标尺度一致性
        返回归一后的关键点损失（标量）
        """
        # 计算预测与真实关键点的「欧氏距离的平方」（x 坐标差的平方 + y 坐标差的平方）
        """
        ... 用于 省略张量前面的所有 “非目标维度”，仅保留最后一个维度用于索引。它等价于 “将前面所有维度直接传递，只对最后一维做切片”
        - pred_kpts[..., 0]：省略前两维（n, k），取最后一维的第 0 个元素 → 提取所有正样本、所有关键点的 x 坐标，最终形状为 (n, k)
        - pred_kpts[..., 1]：省略前两维（n, k），取最后一维的第 1 个元素 → 提取所有正样本、所有关键点的 y 坐标，最终形状为 (n, k)
        - gt_kpts[..., 0]：同理，从 (n, k, 3) 中提取所有正样本、所有关键点的 真实 x 坐标，形状 (n, k)
        - gt_kpts[..., 1]：提取所有正样本、所有关键点的 真实 y 坐标，形状 (n, k)
        """
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]) ** 2 + (pred_kpts[..., 1] - gt_kpts[..., 1]) ** 2 # ... 用于“将前面所有维度直接传递，只对最后一维做切片”。
        # loss factor 防止没有关键点时梯度为 0（保障数值稳定性）
        kpt_loss_factor = (torch.sum(kpt_mask != 0) + torch.sum(kpt_mask == 0)) / (torch.sum(kpt_mask != 0) + 1e-9)
        # e = d / (2 * (area * sigma)^2) 参考 COCO OKS（此处实现与 COCO 评估接近）
        e = d / (2 * self.sigmas) ** 2 / (area + 1e-9) / 2
        return kpt_loss_factor * ((1 - torch.exp(-e)) * kpt_mask).mean()


# ============================
# v8DetectionLoss：检测任务的核心损失实现（所有 task-specific loss 都基于此构建）
# ============================
class v8DetectionLoss:
    """
    设计理念概述：
    - 将检测共有的步骤集中实现：pred 解码（包括 DFL 支持）、生成 anchors、使用 assigner 分配正负样本、计算 cls/bbox/dfl 损失。
        - 多尺度特征拼接：把不同下采样率（stride）的预测合并成一个长列表，方便统一处理。
        - 解码：把网络输出的“回归表达”（可能是分布/离散 bin）转为实际 bbox 坐标（xyxy）。
        - 分配：训练时需要知道哪些预测点是正样本（要对它们计算 box/cat loss），哪些是负样本。传统方法用 anchor-IoU threshold，但现代方法（TaskAligned/SimOTA 等）会综合考虑 分类置信度 和 bbox 质量 来动态选正样本，以得到更可靠的监督信号。
    - 任务特异性（如 mask／kpt）由子类覆盖 __call__（或在 __call__ 中调用父类工具）。
    """

    def __init__(self, model):  # model must be de-paralleled
        device = next(model.parameters()).device  # 模型当前 device
        h = model.args  # 超参数（训练超参字典）

        m = model.model[-1]  # head（Detect 模块），需要读取 head 上定义的一些属性
        self.bce = nn.BCEWithLogitsLoss(reduction='none')  # 用于每-anchor 每-class 的 BCE（未 reduction）
        self.hyp = h
        self.stride = m.stride  # head 每个尺度对应的下采样率（tensor）
        self.nc = m.nc  # 类别数量
        self.no = m.no  # head 每个网格点输出的通道数（reg+dims）
        self.reg_max = m.reg_max  # 用于 DFL 的参数
        self.device = device

        # 是否启用 DFL（当 reg_max>1 时启用）
        self.use_dfl = m.reg_max > 1

        # TaskAlignedAssigner：一种现代的分配器，用于动态选取高质量预测作为正样本（结合 cls & iou）
        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        # bbox_loss 封装 IoU + DFL 计算
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)
        # 用于 DFL 将离散 bin 编号 [0, 1, ..., reg_max-1] 做 matmul 转为期望
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """
        把 dataset 提供的 targets（每行: [img_idx, class, x,y,w,h]）按 image 聚合成 (B, max_targets, 5)，并把 xywh 转为 xyxy 并乘 scale_tensor（恢复像素坐标）
        - 并把 xywh 转为 xyxy 并乘以 scale_tensor（把归一化坐标恢复到像素尺度）。为什么要转成 xyxy？ 因为模型预测的边界框输出就是 xyxy 格式，且计算边界框损失（如 CIoU）时，需要用 xyxy 来算两个框的交并比、中心距离等，格式一致才能直接计算。
        - 这样可以按 image 索引对 targets 做批处理，便于后续 assign/计算
        - preprocess的输出是一个按图片分组的列表，每个元素对应批次中一张图片的所有目标标注。preprocess会根据img_idx把标注拆分成 “每张图专属的标注列表”
        
        scale_tensor用于将归一化的标注数据转换成“原图像素坐标”。原始标注的x, y, w, h是 “归一化值”（相对于原图宽高），而模型输入的图片可能经过了缩放（比如原图 800x600→模型输入 640x480），因此需要把归一化坐标转换成 “输入图像的像素坐标”
        
        out shape: (Batch_size, Max_target, 5), 其中，max_targets = 当前 batch 中单张图片的最大标注数量；5 表示[class, x1, y1, x2, y2]，其中 x1,y1,x2,y2 是以像素为单位的左上/右下角坐标（xyxy 形式），没有真实 gt 的填充行，全为 0（因此可用 sum == 0 判断这一行是空填充）
        例如假设batch size = 8，这8张图片中每张图片的Bbox标记数量为3， 5，7， 9， 1， 2，2， 6.那么M应该是9。out 中每一行前 n=目标数量位置填真实值 [class, x1,y1,x2,y2]（像素坐标），剩余位置填 0
        """
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image idx
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            # targets[..., 1:5] 原为 xywh，转为 xyxy 并乘以 scale（恢复成像素）
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """
        将网络预测的回归输出解码为 bbox:
        - 若启用 DFL: 把 pred_dist reshape -> softmax on bins -> matmul proj -> 得到连续值
        - 最终调用 dist2bbox 将 ltrb distances 与 anchor_points 转为 xyxy bbox
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            # 这里把 channels -> 4 sides * (reg_max) 每侧的分布
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist shape -> (b, a, 4)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        """
        处理预测并返回损失（sum across components）：
        preds: 网络输出的 feats（list of feature maps）或 (feats, ...) 形式
        batch: 来自 dataloader 的 batch（包含 batch_idx, cls, bboxes 等）
        高层流程：
          - 将多尺度 feat 拼接成 (B, no, Npred) 并 split 成 pred_distri & pred_scores
          - 生成 anchor_points 与 stride_tensor
          - preprocess targets -> gt_labels / gt_bboxes
          - bbox_decode -> pred_bboxes
          - 使用 assigner 给出正样本 mask/target_scores/target_bboxes
          - 计算 cls loss (BCE) 与 bbox loss (IoU + DFL)
          - 按超参加权返回总 loss
        """
        loss = torch.zeros(3, device=self.device)  # [box, cls, dfl]
        feats = preds[1] if isinstance(preds, tuple) else preds  # 支持两种 preds 结构
        # 将 feats 每个尺度 reshape 并在最后维度拼接，然后 split 为 (reg_dist, cls_logits)
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        # 为后续处理调整维度 (B, Npred, C)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        # feats[0].shape[2:] 是最小尺度的特征图 spatial size，乘以 stride[0] 得到 image size
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        # 生成 anchor grid 中心点（相对于 feat grid）和对应 stride tensor
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # =========================
        # targets 预处理
        # =========================
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # 标签与坐标
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)  # 标注存在的 mask

        # =========================
        # decode predicted bboxes
        # =========================
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, shape (b, Npred, 4)

        # =========================
        # assigner: 分配正负样本
        # =========================
        # assigner 根据预测分数(sigmoid)与预测 bbox（乘回像素尺度）与 gt 去计算匹配
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        # =========================
        # cls loss（BCE）
        # =========================
        # 注：可替换为 VarifocalLoss（更复杂的 soft-label 设计）
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        # =========================
        # bbox loss（IoU [+ DFL]）
        # =========================
        if fg_mask.sum():
            target_bboxes /= stride_tensor  # 将 target box 也转换到网格尺度
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)

        # apply gains from hyperparameters
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl

        # 返回 scaled 总 loss 以及 detach 的各项 loss（便于日志记录）
        return loss.sum() * batch_size, loss.detach()


# ============================
# v8SegmentationLoss：在检测 loss 基础上增加 mask 分支的损失计算
# ============================
"""
v8SegmentationLoss（继承 v8DetectionLoss 并扩展） 功能与设计：

- 继承 v8DetectionLoss 的大部分逻辑（bbox 解码、assigner、cls loss、bbox dfl），新增处理 mask 分支的计算。
- 额外属性：
    - self.nm：mask 原型数量（proto channels，典型 YOLACT 风格：proto 是网络输出的一组原型 mask）。
    - self.overlap：是否通过 overlap_mask 处理重叠 mask 的逻辑。
- 输入 preds：由 network 返回 (feats, pred_masks, proto)（或包含在 tuple 的第二元素）。
    - proto 是 prototype maps（形如 [B, nm, Hm, Wm]）。
    - pred_masks 是每个预测对应的 mask coefficients（每个预测会线性组合 proto 得到最终 mask）。
- 对每个正样本：
    - 从 proto 与 pred_masks 通过线性组合得到 pred_mask，然后用 binary_cross_entropy_with_logits 与 gt_mask 比较。
    - 使用 crop_mask 将 mask 裁剪到 bbox 区域并除以区域 area，减小大目标对损失的主导影响（normalize by object area）。
- 为避免 DDP 中出现 unused gradients 的问题，保留一些零操作。
- 最终 loss 加权：box, seg, cls, dfl 分别乘以 self.hyp.box / box/batch / cls / dfl 等。

为什么这样设计（算法/工程原因）

- 分割（instance mask）通常用 prototype+coeffs（YOLACT 样式）能高效地产生任意数量的 mask，而不是为每个实例输出完整 HxW mask。
- 为了同时训练检测与分割，loss 需要既保证定位/分类，又保证 mask 质量；因此把 mask loss 添加到检测 loss 管线中并做合适的规范化（如按面积）。
- overlap 选项用于处理同一像素可能属于多个实例的情况（部分数据集标签方式不同）。
"""
class v8SegmentationLoss(v8DetectionLoss):
    """
    继承关系与设计：
    - 继承 v8DetectionLoss，因此复用了 detection 的: 多尺度 feat 拼接、decode、assign、cls 与 bbox 损失计算
    - 在此基础上添加还原/计算 mask 的流程（Prototype mask + Coefficient 的方式），并把 mask 损失加入总损失中。
    设计动机：
    - 使用 prototype + coeffs（类似 YOLACT）能节省显存与计算：网络只输出固定数量的 proto 特征图和每个实例对应的系数，
      再线性组合得到实例 mask。
    - Proto原型掩码：先学习一组通用的“原型掩码”，再为每个目标预测一组“权重系数”，通过“系数加权原型”的方式生成该目标的独特掩码，用“少量基础模版”组合出“无限组目标形状”
    """
    def __init__(self, model):  # model must be de-paralleled
        super().__init__(model)
        self.nm = model.model[-1].nm  # head 中定义的mask 原型数量（proto channels，典型 YOLACT 风格：proto 是网络输出的一组原型 mask）。 
        self.overlap = model.args.overlap_mask  # 是否标签中同像素可能属于多个实例,是否通过 overlap_mask 处理重叠 mask 的逻辑

    def __call__(self, preds, batch):
        """
        preds: (feats, pred_masks, proto) 或 (something, (feats, pred_masks, proto))
        - feats: list of feature maps for detection head
        - pred_masks: 每个预测对应的 mask coefficients (B, nm, Npred) -> 经过 permute 后 (B, Npred, nm)
        - proto: prototype masks (B, nm, Hm, Wm) 用于线性组合生成每个实例的 mask
        总体流程：
          - 先复用 detection 的 cls/bbox 分支计算（利用 assigner 得到 fg_mask、target_idx 等）
          - 对每个正样本按索引用 proto 与 coeff 线性组合重建 pred_mask，并与 gt_mask 做 BCE 损失
          - mask 损失会按 bbox 面积归一（area normalization）并裁剪到 bbox（crop_mask）
        """
        loss = torch.zeros(4, device=self.device)  # box, seg, cls, dfl （注意索引对应）
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1] # 解析输入 preds——preds 为 (feats, pred_masks, proto) 或嵌套 tuple，需提取 feats（检测分支特征图）、pred_masks（mask 系数）、proto（原型掩码）
        batch_size, _, mask_h, mask_w = proto.shape  # proto 的空间分辨率（通常比原图低）
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1) # 处理多尺度 feats—— 将各尺度 feats 拼接并拆分得到 pred_distri（bbox 回归分布）和 pred_scores（分类置信度）

        # 将 (B, no, N) -> (B, N, C), 调整 pred_masks 的维度（从 (B, nm, Npred) 转为 (B, Npred, nm)，便于后续按实例计算）
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        # pred_masks: (B, nm, N) -> (B, N, nm)
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets: 类似 detection 的处理，但若非 segment 数据集会报错（保护性检查）
        try:
            batch_idx = batch['batch_idx'].view(-1, 1)
            targets = torch.cat((batch_idx, batch['cls'].view(-1, 1), batch['bboxes']), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        except RuntimeError as e:
            raise TypeError('ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n'
                            "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                            "i.e. 'yolo train model=yolov8n-seg.pt data=coco128.yaml'.\nVerify your dataset is a "
                            "correctly formatted 'segment' dataset using 'data=coco128-seg.yaml' "
                            'as an example.\nSee https://docs.ultralytics.com/tasks/segment/ for help.') from e

        # 生成 anchors 与 bbox 解码 —— 复用父类 make_anchors 生成 anchor_points（网格中心点）和 stride_tensor（各 anchor 的下采样率），调用 bbox_decode 将 pred_distri 解码为预测 bbox（xyxy 格式）
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, N, 4)

        # assigner: 返回 target_bboxes, target_scores, fg_mask, target_gt_idx（后者用于索引 gt mask）
        """
        正负样本分配（assigner）：
        调用 TaskAlignedAssigner，结合 pred_scores（sigmoid 后）、pred_bboxes（detach 后）与 gt_labels/gt_bboxes，输出 target_bboxes（分配的 gt bbox）、target_scores（分类软标签）、fg_mask（正样本掩码）、target_gt_idx（正样本对应的 gt 索引，用于后续索引 gt_mask）。
        """
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss（BCE）
        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        if fg_mask.sum():
            # bbox loss（IoU + DFL）
            loss[0], loss[3] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes / stride_tensor,
                                              target_scores, target_scores_sum, fg_mask)
            # masks loss 计算
            masks = batch['masks'].to(self.device).float()
            # proto 与 gt_mask 分辨率对齐：若 gt_mask（batch ['masks']）的分辨率与 proto 不一致，用最近邻插值下采样到 proto 的分辨率（mask_h, mask_w），确保计算维度匹配
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode='nearest')[0]

            # 对 batch 中每张图逐个计算正样本对应的 mask loss，遍历 batch 逐图处理正样本
            for i in range(batch_size):
                if fg_mask[i].sum():
                    # target_gt_idx 指示了每个正样本对应的哪个 gt mask 索引
                    mask_idx = target_gt_idx[i][fg_mask[i]]
                    if self.overlap: # 索引 gt_mask：根据 target_gt_idx 找到当前正样本对应的 gt_mask，若 overlap=True，用 torch.where 将 “像素值 = 实例 ID” 转为二值 mask；否则直接按索引取 gt_mask
                        # 若 dataset 用重叠 mask 表示不同实例值（像素值为 instance id），则用 where 构建二值 gt_mask
                        gt_mask = torch.where(masks[[i]] == (mask_idx + 1).view(-1, 1, 1), 1.0, 0.0)
                    else:
                        gt_mask = masks[batch_idx.view(-1) == i][mask_idx]
                    # xyxy coords normalized in [0,1] 用于 crop_mask，计算 mask 归一化参数：将 target_bboxes 转为归一化 xyxyn（相对于原图），再转为 proto 分辨率下的 mxyxy（用于 crop_mask），计算目标面积 marea（用于平衡大小目标的 mask 损失）
                    xyxyn = target_bboxes[i][fg_mask[i]] / imgsz[[1, 0, 1, 0]]
                    marea = xyxy2xywh(xyxyn)[:, 2:].prod(1)  # area normalization
                    mxyxy = xyxyn * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=self.device)
                    # 调用 single_mask_loss 计算单图正样本的 mask 损失，并累加到总 loss [1]（seg loss）。single_mask_loss: 将 pred coeff 与 proto 线性组合得到 pred_mask，再与 gt_mask 计算 BCE，并 crop/area normalization
                    loss[1] += self.single_mask_loss(gt_mask, pred_masks[i][fg_mask[i]], proto[i], mxyxy, marea)
                else:
                    # 防止 DDP 中某些进程没有正样本导致 unused-parameter 的问题（保持梯度图连通）。DDP 梯度图连通处理：若某图无正样本，需对 proto 和 pred_masks 做 “0 乘” 操作（proto0 + pred_masks0），避免 DDP 训练中因部分进程无梯度导致的 unused params 错误
                    loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()
        else:
            # 同上：若整个 batch 没有正样本，也保持梯度图连通以避免 DDP 报错
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()

        # loss 权重
        # 损失加权与返回：按超参（hyp）对各损失项加权（box -> hyp.box、seg -> hyp.box/batch_size、cls -> hyp.cls、dfl -> hyp.dfl），返回总损失（乘以 batch_size 保持尺度）和各分项损失（detach 后用于日志记录）
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.box / batch_size  # seg loss 通常按 batch_size 归一
        loss[2] *= self.hyp.cls
        loss[3] *= self.hyp.dfl

        # 返回总 loss（乘以 batch_size 以保持历史实现的 scale）以及 detach 的分项
        return loss.sum() * batch_size, loss.detach()

    def single_mask_loss(self, gt_mask, pred, proto, xyxy, area):
        """
        单实例 mask loss：
        - pred: 该实例的 mask coefficient 向量 (nm,)
        - proto: prototype maps (nm, Hm, Wm)
        - pred_mask = pred @ proto.reshape(nm, -1) -> reshape to (Hm, Wm)
        - 用 BCEWithLogits 与 gt_mask 计算像素级损失，然后 crop_mask（只计算 bbox 区域）并按 target 面积归一
        设计理由：按面积归一可以平衡大/小实例的影响（避免大实例主导 loss）
        """
        # 生成预测 mask：将 pred_masks（当前实例的系数，shape=(nm,)）与 proto（shape=(nm, Hm, Wm)）做矩阵乘法，即 pred_mask = pred @ proto.view (nm, -1)，再 reshape 为 (n, Hm, Wm)（n 为当前图正样本数）
        pred_mask = (pred @ proto.view(self.nm, -1)).view(-1, *proto.shape[1:])  # (n, Hm, Wm)
        # 像素级损失计算：用 F.binary_cross_entropy_with_logits 计算 pred_mask 与 gt_mask 的像素级 BCE 损失（reduction='none'，保留每个像素的损失）
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction='none')
        # 裁剪与归一：调用 crop_mask 裁剪掉 mask 中 “超出目标 bbox” 的区域（只保留目标区域的损失，减少背景干扰），计算裁剪后损失的均值，再除以目标面积 marea（平衡大小目标的损失贡献，避免大目标主导 loss）
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).mean()


# ============================
# v8SegmentationPoseLoss：在 Segmentation 基础上进一步加入姿态（关键点）损失（联合任务）
# ============================
"""
v8SegmentationPoseLoss
- 继承自 v8DetectionLoss，然后同时实现 segment 的 mask loss 和 pose 的关键点 loss。
- 输入 preds 包含 feats, pred_masks, proto, pred_kpts。
- 对每个正样本同时计算 mask loss（和裁剪/面积归一化）与关键点 loss（位置 + 可见性），再与检测 loss 权重相加。
- 设计目的是在单一网络中联合学习 detection + segmentation + pose，利用共享特征节约计算并让任务互相监督（例如关键点可能增强实例边界特征等）。

实现细节与工程注意点（为何要这样实现）
- Permute/reshape：多尺度 feat 的拼接和维度变换是为了把 (B, no, H, W) 形式转成 (B, Npred, C) 便于整体处理（Npred = sum(HW)）。
- make_anchors：返回 anchor_points（grid 的中心点相对于 feature），使 bbox 解码/关键点解码与网络网格对齐。
- targets preprocess：把 dataset 中的 GT 转成统一 batch 形状，并乘以 scale（由网络特征到像素尺度的映射）以便匹配 decoded predictions。
- loss 权重（self.hyp.xxx）：可调节不同任务重要性（例如你可以把 pose 的权重调高以提升关键点精度）。
- 返回格式：loss.sum() * batch_size（训练中常这样返回以保持和以前实现兼容），并返回 detach 的 loss components 以便在训练日志中显示。
"""
class v8SegmentationPoseLoss(v8DetectionLoss):
    """
    这个类演化自 v8DetectionLoss（即继承 detection 的核心流程），并在此基础上：
      - 引入 segmentation 所需的 proto/pred_masks/单实例 mask loss（复用 v8SegmentationLoss 的思想）
      - 引入 pose 所需的 pred_kpts 的解析与关键点 loss（KeypointLoss）、关键点可见性(kobj) loss
    设计动机：
      - 多任务联合训练（检测 + 分割 + 姿态）可以共享 backbone/neck 特征，互相监督有时能提升性能且节省推理成本。
      - 但实现较复杂：需要在 assigner 得到正样本索引后同时为每个正样本计算 mask loss 与 keypoint loss，并正确归一化权重。
    关键点（实现上需要注意）：
      - 继承 v8DetectionLoss 复用了以下功能：anchors 生成、pred 解码、assigner 调用、bbox/cls/dfl 基本项。
      - 为 segmentation/p pose 新增的字段与步骤需与 detection 部分对齐（例如 fg_mask, target_gt_idx 用于索引 gt mask / keypoints）。
      - 关键点预测 pred_kpts 格式预期为 (B, nm?, Npred, kpt_shape) 或 (B, Npred, nkpt, 3)，代码先 permute -> reshape -> decode。
    """

    def __init__(self, model, overlap=True):  # model must be de-paralleled
        super().__init__(model)
        # mask prototype channels
        self.nm = model.model[-1].nm
        self.overlap = overlap

        # keypoints shape 存在于 head 配置中（例如 [17, 3] 表示 17 个关键点，每个点预测 (x,y,vis)）
        self.kpt_shape = model.model[-1].kpt_shape
        # 二值交叉用于 kpt 可见性 / 置信
        self.bce_pose = nn.BCEWithLogitsLoss()
        # 若 kpt_shape == [17,3] 则认为是 COCO 风格 keypoints，使用 OKS_SIGMA 否则用均匀权重
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        # KeypointLoss 会使用 sigmas 和目标面积对关键点误差做加权与归一
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds, batch):
        """
        preds: (feats, pred_masks, proto, pred_kpts) 或 preds[1] 的形式
        返回: 总 loss，及 detach 的分项（box, seg, cls, dfl, pose, kobj）
        主要流程：
          1. 复用 detection 的 feat -> pred_distri / pred_scores 解码
          2. permute/reshape pred_masks 与 pred_kpts（使形状为 (B, Npred, nm) 与 (B, Npred, nkpt, kdim)）
          3. 生成 anchors，preprocess targets，decode pred_bboxes & pred_kpts（kpts_decode）
          4. 使用 assigner 分配，得到 fg_mask、target_gt_idx（后者用于索引 GT mask 与 GT keypoints）
          5. 计算 cls 与 bbox/dfl loss（复用父类）
          6. 对每个正样本：
             - 计算 mask loss（同 v8SegmentationLoss）
             - 提取对应 gt keypoints（按 anchor scale/stride 调整）并计算 keypoint_loss（位置）与 kobj（二值可见性）损失
          7. 按超参对各项 loss 加权并返回
        """
        loss = torch.zeros(6, device=self.device)  # box, seg, cls, dfl, pose, kobj
        feats, pred_masks, proto, pred_kpts = preds if len(preds) == 4 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape  # proto 的 (B, nm, Hm, Wm)

        # 将 feats 拼接为 pred_distri 与 pred_scores（相同于父类）
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)
        # (B, no, N) -> (B, N, C)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        # pred_masks: (B, nm, N) -> (B, N, nm)
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()
        # pred_kpts: (B, kdim, N) -> (B, N, kdim) -> 之后会 view 成 (B, N, nkpt, kpt_dim)
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size in pixels
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # =========================
        # targets preprocess（同 detection）
        # =========================
        try:
            batch_idx = batch['batch_idx'].view(-1, 1)
            targets = torch.cat((batch_idx, batch['cls'].view(-1, 1), batch['bboxes']), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        except RuntimeError as e:
            raise TypeError('ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n'
                            "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                            "i.e. 'yolo train model=yolov8n-seg.pt data=coco128.yaml'.\nVerify your dataset is a "
                            "correctly formatted 'segment' dataset using 'data=coco128-seg.yaml' "
                            'as an example.\nSee https://docs.ultralytics.com/tasks/segment/ for help.') from e

        # =========================
        # decode predicted bboxes & keypoints
        # =========================
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # (B, N, 4)
        # pred_kpts 需要重塑为 (B, N, nkpt, kpt_dim) 再 decode
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (B, N, nkpt, kpt_dim)

        # assigner 返回 target_bboxes, target_scores, fg_mask, target_gt_idx（target_gt_idx 用于索引 GT mask/kpts）
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        if fg_mask.sum():
            # bbox (IoU) 和 DFL loss（复用 BboxLoss）
            loss[0], loss[3] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes / stride_tensor,
                                              target_scores, target_scores_sum, fg_mask)
            # masks loss 部分（与 v8SegmentationLoss 相同）
            masks = batch['masks'].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode='nearest')[0]

            # keypoints 从 batch 中读入并放大为像素坐标（乘以 imgsz）
            keypoints = batch['keypoints'].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            # 遍历 batch 中每张图，对其正样本逐个计算 mask + keypoint loss
            for i in range(batch_size):
                if fg_mask[i].sum():
                    # 用 assigner 提供的 target_gt_idx 找到当前正样本对应的 GT 索引
                    mask_idx = target_gt_idx[i][fg_mask[i]]
                    if self.overlap:
                        gt_mask = torch.where(masks[[i]] == (mask_idx + 1).view(-1, 1, 1), 1.0, 0.0)
                    else:
                        gt_mask = masks[batch_idx.view(-1) == i][mask_idx]
                    # 归一化 bbox 到 [0,1] 以便 crop_mask
                    xyxyn = target_bboxes[i][fg_mask[i]] / imgsz[[1, 0, 1, 0]]
                    marea = xyxy2xywh(xyxyn)[:, 2:].prod(1)  # 面积
                    mxyxy = xyxyn * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=self.device)
                    # mask loss 累加
                    loss[1] += self.single_mask_loss(gt_mask, pred_masks[i][fg_mask[i]], proto[i], mxyxy, marea)

                    # ========== keypoint loss ==========
                    # 根据 assigner 返回的索引取出对应的 GT keypoints（形状 (n, nkpt, 3)）
                    idx = target_gt_idx[i][fg_mask[i]]
                    gt_kpt = keypoints[batch_idx.view(-1) == i][idx]  # (n, nkpt, 3)
                    # 把 gt_kpt 的 xy 换算成和 pred_kpts 相同的网格尺度（除以对应的 stride）
                    gt_kpt[..., 0] /= stride_tensor[fg_mask[i]]
                    gt_kpt[..., 1] /= stride_tensor[fg_mask[i]]
                    # area 用于 keypoint loss 归一（使用 target_bboxes / stride_tensor 得到网格尺度的 bbox）
                    area = xyxy2xywh((target_bboxes/stride_tensor)[i][fg_mask[i]])[:, 2:].prod(1, keepdim=True)
                    pred_kpt = pred_kpts[i][fg_mask[i]]  # predicted kpts for these positive anchors
                    # kpt_mask: 可见性掩码（gt_kpt[...,2] != 0）
                    kpt_mask = gt_kpt[..., 2] != 0
                    # 位置损失（KeypointLoss）
                    loss[4] += self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)
                    # 若 pred_kpt 同时输出了 kpt score（最后一维 == 3），则计算 kobj 可见性/置信损失（BCE）
                    if pred_kpt.shape[-1] == 3:
                        loss[5] += self.bce_pose(pred_kpt[..., 2], kpt_mask.float())

                else:
                    # 保持计算图连通，防止 DDP unused params
                    loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()

        else:
            # 同上
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()

        # 归一与超参加权：
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.box / batch_size  # seg 按 batch 归一
        loss[2] *= self.hyp.cls
        loss[3] *= self.hyp.dfl
        loss[4] *= self.hyp.pose / batch_size  # pose 位置损失按 batch 归一
        loss[5] *= self.hyp.kobj / batch_size  # kobj 按 batch 归一

        return loss.sum() * batch_size, loss.detach()  # 返回总 loss 及各分项（detach 用于日志）

    def single_mask_loss(self, gt_mask, pred, proto, xyxy, area):
        """
        与 v8SegmentationLoss.single_mask_loss 相同实现：
        - pred: (nm,) coefficient
        - proto: (nm, Hm, Wm)
        - pred_mask = pred @ proto.reshape(nm, -1) -> (Hm, Wm)
        - BCE + crop_mask + area 归一
        """
        pred_mask = (pred @ proto.view(self.nm, -1)).view(-1, *proto.shape[1:])
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction='none')
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).mean()

    def kpts_decode(self, anchor_points, pred_kpts):
        """
        把相对预测的关键点值解码回 anchor/grid 的坐标系：
        - pred_kpts 的前两维假设是相对 offset（实现上乘 2 再偏移 anchor-0.5，和 bbox offset 的编码一致）
        - 这一步保证 pred_kpts 与 gt_kpt 在同一坐标尺度下比较（用于 KeypointLoss）
        注意：该解码假设网络输出的 keypoint encoding 是按相同的编码规则（见网络 head 实现）。
        """
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y


# ============================
# v8PoseLoss：只包含姿态任务（继承自 v8DetectionLoss）
# v8SegmentationLoss 与 v8PoseLoss 都是基于统一的 detection 流程（multi-scale concat → decode → assign → cls/bbox loss）再分别附加任务专属模块（prototype+coeff mask 与 keypoint decode+OKS loss）
# ============================
"""
v8PoseLoss（继承 v8DetectionLoss 并扩展） 功能与设计要点总结：
- 在继承检测通用逻辑（multi-scale concat / decode / assign / cls & bbox loss）的基础上，新增关键点任务相关处理。
- 继承复用的项目（来自父类 v8DetectionLoss）：
    * 多尺度 preds 拼接、pred_distri/pred_scores 的 reshape/permute
    * anchor_points / stride_tensor 的生成（make_anchors）
    * targets preprocess（统一成 (B, M, 5) 格式）
    * bbox 解码（包含 DFL 支持）
    * assigner 调用（TaskAlignedAssigner）用于生成 fg_mask / target_scores / target_bboxes / target_gt_idx
    * bbox loss（IoU + DFL）计算（BboxLoss）
- 新增的 pose 特性（v8PoseLoss 特有）：
    * 解析 pred_kpts：把 head 输出的 pred_kpts reshape/permute 为 (B, Npred, nkpt, kpt_dim) 并 decode 到 grid 坐标（kpts_decode）
    * 从 batch 中读取 GT keypoints（像素坐标或归一化），将 GT 转换到 pred_kpts 的坐标尺度（通常除以 stride）以对齐
    * 计算 KeypointLoss（位置误差，使用 OKS sigma + area 归一）和可见性/置信 BCE（若模型同时预测可见性分量）
- 工程设计动因：
    * 多任务共享特征（backbone/neck）可节省计算并提供互补监督（例如关键点可增强定位特征）
    * 继承父类实现可避免重复代码且确保 assign / bbox loss 的一致性（同一 assign 定义用于 detection 与 pose）
    * 在正样本（fg_mask）上计算 pose loss，确保关键点 supervision 只作用于与 GT 匹配的预测点（即那些负责预测该实例的 anchors）
"""

class v8PoseLoss(v8DetectionLoss):
    """
    继承 v8DetectionLoss 的通用检测逻辑，并在其上添加 keypoint-specific loss：
    - keypoint 位置损失（KeypointLoss）
    - keypoint 可见性/置信损失（BCE）
    设计与 v8SegmentationPoseLoss 的区别在于：此类只处理 pose（不处理 mask/proto）
    
    - “继承”意味着 v8PoseLoss 并没有重写 detection 的前半部分（拼接/解码/assign），
      而是在那些步骤之后插入 pose 相关的处理并复用 bbox/cls 的损失结果。
    - 这样做的好处是：保证 detection 与 pose 使用一致的正样本分配（same assigner），
      使得 pose 学习与检测任务“对齐”（避免不同任务互相冲突）。
    """

    def __init__(self, model):  # model must be de-paralleled
        super().__init__(model) # 继承父类 v8DetectionLoss（由父类获得：self.bce、self.stride、self.nc、self.no、self.reg_max、self.assigner、self.bbox_loss、self.proj 等检测共用工具）
        self.kpt_shape = model.model[-1].kpt_shape # 从模型配置文件的head读取关键点参数
        self.bce_pose = nn.BCEWithLogitsLoss() # 用于关键点的可见性/置信预测损失（binary sigmoids）Q: 为什么用 BCE 做 keypoint 可见性而不是 MSE？A: 可见性/置信是二分类/概率性质，用 BCE（带 logits）更合适；位置用专门归一化的欧式/OKS 风格损失。
        is_pose = self.kpt_shape == [17, 3] # COCO 人体关键点任务的判断开关。is_pose=True：使用 COCO 预定义的 OKS_SIGMA；is_pose=False：使用默认的 “均匀权重”（torch.ones(nkpt, device=self.device) / nkpt），即所有关键点的误差敏感度相同
        nkpt = self.kpt_shape[0]
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt # OKS（Object Keypoint Similarity）在 COCO 中定义了不同关键点对误差的尺度敏感度（例如鼻子比脚更精确重要），该 sigma 用于按关键点类型加权位置误差，使评价与训练更契合 COCO 的实际要求
        self.keypoint_loss = KeypointLoss(sigmas=sigmas) # KeypointLoss 使用 sigmas 与 area 做归一

    def __call__(self, preds, batch):
        """
        preds: (feats, pred_kpts) 或 preds[1]
        主要流程：
        1) 多尺度 preds 拼接 -> 得到 pred_distri（回归）与 pred_scores（分类）
        2) 生成 anchor_points & stride_tensor，用于 bbox 与 kpts 的坐标解码与尺度转换
        3) preprocess batch targets -> gt_labels, gt_bboxes（像素尺度）
        4) bbox_decode 把 pred_distri 解码为 pred_bboxes（grid 单位或接近）
        5) kpts_decode 把 pred_kpts 解码为 grid 单位的关键点预测（以 anchor 为中心的偏移）
        6) assigner 使用 pred_scores & pred_bboxes 与 GT 做匹配，得到 fg_mask 与 target_gt_idx（后者用来索引 GT keypoints）
        7) 计算 cls 与 bbox loss（复用父类实现）
        8) 对每个正样本，取对应的 gt_kpt（像素），把它转为 grid 单位（除以 stride），计算 keypoint loss（位置 + 可见性）
        """
        # 预分配 loss 向量：索引含义在代码中注释明确
        # loss indices mapping used here:
        # loss[0] = box (IoU)
        # loss[1] = kpt location loss
        # loss[2] = kpt visibility (kobj) BCE loss
        # loss[3] = cls loss
        # loss[4] = dfl loss (bbox distribution focal)   <-- 注意：此处的索引顺序为实现细节，阅读时以代码赋值为准
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1] # preds 解析：支持两种 preds 结构（直接 (feats, pred_kpts) 或 nested）
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        # reshape 为 (B, Npred, C) 以方便后续计算
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        # pred_kpts: (B, kdim, N) -> (B, N, kdim)
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        # 计算 image size（像素）用于把归一化 GT 恢复为像素坐标
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        # 生成 anchors（grid centers）与 stride tensor（每个预测点对应的下采样率）
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # =========================
        # targets preprocess（同 detection）
        # =========================
        batch_size = pred_scores.shape[0]
        batch_idx = batch['batch_idx'].view(-1, 1)
        targets = torch.cat((batch_idx, batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        
        # =========================
        # decode boxes and kpts
        # =========================
        # pred_bboxes: (B, Npred, 4)（grid 单位 / 可乘 stride 还原到像素）
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # (B, N, 4)
        # pred_kpts 需要 view 成 (B, Npred, nkpt, kpt_dim) 再 decode（kpts_decode 会把相对 offset 转为 grid 坐标）
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (B,N,nkpt,kpt_dim)

        # =========================
        # assigner: 使用 pred_scores(sigmoid) 与 pred_bboxes*stride (像素尺度) 去匹配 GT
        # 返回 (_, target_bboxes, target_scores, fg_mask, target_gt_idx)
        # target_gt_idx 是每个正样本对应的 GT 索引，用于后续从 batch 中取出该实例的 keypoints
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        # =========================
        # cls loss（复用父类 BCE）
        # =========================
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        # =========================
        # bbox + dfl + keypoint loss
        # =========================
        if fg_mask.sum():
            target_bboxes /= stride_tensor # 将 target_bboxes 转为 grid 单位（与 pred_distri 的坐标系一致）
            loss[0], loss[4] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask) # bbox loss（IoU, DFL）复用 BboxLoss
            # keypoints 从 batch 中读取并变为像素坐标（如果原始是归一化）
            keypoints = batch['keypoints'].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1] # x -> width
            keypoints[..., 1] *= imgsz[0] # y -> height
            
            # 对每张图片处理其正样本
            for i in range(batch_size):
                if fg_mask[i].sum():
                    idx = target_gt_idx[i][fg_mask[i]] # target_gt_idx[i][fg_mask[i]] 表示当前图中每个正样本对应的 GT 索引
                    gt_kpt = keypoints[batch_idx.view(-1) == i][idx]  # 按 idx 提取该图对应 GT keypoints（shape (n_pos, nkpt, 3)）
                    # 归一化 gt_kpt 到网格尺度，把 GT 除以 stride：预测是以 grid 单位或相对 anchor 的偏移编码，必须把 GT 转换到相同尺度才能比较
                    # 把 gt_kpt 从像素坐标转换为与 pred_kpts 一致的 grid 单位（除以 stride）
                    # 注意 stride_tensor[fg_mask[i]] 是每个正样本对应的下采样率（像素->grid）
                    gt_kpt[..., 0] /= stride_tensor[fg_mask[i]]
                    gt_kpt[..., 1] /= stride_tensor[fg_mask[i]]
                    # area: 使用 target_bboxes（已在 grid 单位）计算目标面积，用于 keypoint loss 的归一
                    area = xyxy2xywh(target_bboxes[i][fg_mask[i]])[:, 2:].prod(1, keepdim=True)
                    # pred_kpt: (n_pos, nkpt, kpt_dim)，已经过 kpts_decode，属于 grid 单位
                    pred_kpt = pred_kpts[i][fg_mask[i]]
                    kpt_mask = gt_kpt[..., 2] != 0 # 过滤掉不可见/未标注的关键点，避免给不可见点施加错误监督
                    loss[1] += self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # kpt位置损失
                    # kpt 可见性损失,如果模型同时输出关键点置信分量（最后一维 == 3），用 BCE 对可见性/置信做监督
                    if pred_kpt.shape[-1] == 3:
                        loss[2] += self.bce_pose(pred_kpt[..., 2], kpt_mask.float())

        # apply gains
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.pose / batch_size
        loss[2] *= self.hyp.kobj / batch_size
        loss[3] *= self.hyp.cls
        loss[4] *= self.hyp.dfl

        # 返回 total loss（乘以 batch_size 保持原实现比例）和 detach 的 loss items（便于日志）
        return loss.sum() * batch_size, loss.detach()

    def kpts_decode(self, anchor_points, pred_kpts):
        """
        把网络输出的相对 kpt 值解码到 grid 坐标。
        细节说明（小白友好）：
        - 网络通常预测相对于 anchor 的偏移（offset），常见编码为先缩放（乘 2）再平移 anchor-0.5：
            pred_decoded = pred_offset * 2 + anchor - 0.5
          这个编码在训练与推理时常被用于稳定数值与约束范围（把偏移限定在 [-0.5, 1.5] 等可控区间）。
        - kpts_decode 会把该偏移转换为与 anchor grid 对齐的坐标（grid 单位），便于与把 GT 除以 stride 后的坐标直接比较。
        """
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y


class v8ClassificationLoss:
    """
    简单分类损失（cross_entropy），保留与其他 loss 风格的一致接口（返回 loss 与 detach 显示项）
    """
    def __call__(self, preds, batch):
        loss = torch.nn.functional.cross_entropy(preds, batch['cls'], reduction='sum') / 64
        loss_items = loss.detach()
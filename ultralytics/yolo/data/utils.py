# Ultralytics YOLO 🚀, AGPL-3.0 license

import contextlib
import hashlib
import json
import os
import subprocess
import time
import zipfile
from multiprocessing.pool import ThreadPool
from pathlib import Path
from tarfile import is_tarfile

import cv2
import numpy as np
from PIL import ExifTags, Image, ImageOps
from tqdm import tqdm

from ultralytics.nn.autobackend import check_class_names
from ultralytics.yolo.utils import (DATASETS_DIR, LOGGER, NUM_THREADS, ROOT, SETTINGS_YAML, clean_url, colorstr, emojis,
                                    yaml_load)
from ultralytics.yolo.utils.checks import check_file, check_font, is_ascii
from ultralytics.yolo.utils.downloads import download, safe_download, unzip_file
from ultralytics.yolo.utils.ops import segments2boxes

HELP_URL = 'See https://docs.ultralytics.com/yolov5/tutorials/train_custom_data'
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # image suffixes
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv', 'webm'  # video suffixes
PIN_MEMORY = str(os.getenv('PIN_MEMORY', True)).lower() == 'true'  # global pin_memory for dataloaders
IMAGENET_MEAN = 0.485, 0.456, 0.406  # RGB mean
IMAGENET_STD = 0.229, 0.224, 0.225  # RGB standard deviation

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def img2label_paths(img_paths):
    """Define label paths as a function of image paths."""
    sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]


def get_hash(paths):
    """Returns a single hash value of a list of paths (files or dirs)."""
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.sha256(str(size).encode())  # hash sizes
    h.update(''.join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


def exif_size(img):
    """Returns exif-corrected PIL size."""
    s = img.size  # (width, height)
    with contextlib.suppress(Exception):
        rotation = dict(img._getexif().items())[orientation]
        if rotation in [6, 8]:  # rotation 270 or 90
            s = (s[1], s[0])
    return s


def verify_image_label(args):
    """Verify one image-label pair."""
    im_file, lb_file, prefix, keypoint, num_cls, nkpt, ndim = args
    # Number (missing, found, empty, corrupt), message, segments, keypoints
    nm, nf, ne, nc, msg, segments, keypoints = 0, 0, 0, 0, '', [], None
    try:
        # Verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        shape = (shape[1], shape[0])  # hw
        assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
        assert im.format.lower() in IMG_FORMATS, f'invalid image format {im.format}'
        if im.format.lower() in ('jpg', 'jpeg'):
            with open(im_file, 'rb') as f:
                f.seek(-2, 2)
                if f.read() != b'\xff\xd9':  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, 'JPEG', subsampling=0, quality=100)
                    msg = f'{prefix}WARNING ⚠️ {im_file}: corrupt JPEG restored and saved'

        # Verify labels
        if os.path.isfile(lb_file):
            nf = 1  # label found
            with open(lb_file) as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if any(len(x) > 6 for x in lb) and (not keypoint):  # is segment
                    classes = np.array([x[0] for x in lb], dtype=np.float32)
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)
                    lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                lb = np.array(lb, dtype=np.float32)
            nl = len(lb)
            if nl:
                if keypoint:
                    assert lb.shape[1] == (5 + nkpt * ndim), f'labels require {(5 + nkpt * ndim)} columns each'
                    assert (lb[:, 5::ndim] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
                    assert (lb[:, 6::ndim] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
                else:
                    assert lb.shape[1] == 5, f'labels require 5 columns, {lb.shape[1]} columns detected'
                    assert (lb[:, 1:] <= 1).all(), \
                        f'non-normalized or out of bounds coordinates {lb[:, 1:][lb[:, 1:] > 1]}'
                    assert (lb >= 0).all(), f'negative label values {lb[lb < 0]}'
                # All labels
                max_cls = int(lb[:, 0].max())  # max label count
                assert max_cls <= num_cls, \
                    f'Label class {max_cls} exceeds dataset class count {num_cls}. ' \
                    f'Possible class labels are 0-{num_cls - 1}'
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:  # duplicate row check
                    lb = lb[i]  # remove duplicates
                    if segments:
                        segments = [segments[x] for x in i]
                    msg = f'{prefix}WARNING ⚠️ {im_file}: {nl - len(i)} duplicate labels removed'
            else:
                ne = 1  # label empty
                lb = np.zeros((0, (5 + nkpt * ndim)), dtype=np.float32) if keypoint else np.zeros(
                    (0, 5), dtype=np.float32)
        else:
            nm = 1  # label missing
            lb = np.zeros((0, (5 + nkpt * ndim)), dtype=np.float32) if keypoint else np.zeros((0, 5), dtype=np.float32)
        if keypoint:
            keypoints = lb[:, 5:].reshape(-1, nkpt, ndim)
            if ndim == 2:
                kpt_mask = np.ones(keypoints.shape[:2], dtype=np.float32)
                kpt_mask = np.where(keypoints[..., 0] < 0, 0.0, kpt_mask)
                kpt_mask = np.where(keypoints[..., 1] < 0, 0.0, kpt_mask)
                keypoints = np.concatenate([keypoints, kpt_mask[..., None]], axis=-1)  # (nl, nkpt, 3)
        lb = lb[:, :5]
        return im_file, lb, shape, segments, keypoints, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f'{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}'
        return [None, None, None, None, None, nm, nf, ne, nc, msg]


# def verify_image_label_seg_pose(args):
#     """Verify one image-label pair.
#     args：外部打包传入的元组。展开为：
#         - im_file：图像文件路径（字符串）
#         - lb_file：与该图像对应的标签(.txt)文件路径
#         - prefix：日志输出前缀（用于打印 WARNING 等信息）
#         - keypoint：布尔标志，表示数据集中是否有关键点任务（本函数内部其实没用它做条件判断）
#         - num_cls：类别总数（本函数也未使用做校验）
#         - nkpt：关键点个数（期望值，比如 17）
#         - ndim：每个关键点的维度（典型为 3：x,y,visibility）
        
#     各变量作用速览：
#         - im_file：当前图片路径
#         - lb_file：对应标签 txt 路径
#         - prefix：日志前缀（打印 Warning 时前置）
#         - keypoint：布尔开关（这里没用上）
#         - num_cls：总类别数（没用）
#         - nkpt：关键点数量（只在 reshape 用）
#         - ndim：每个关键点维度（reshape 用）
#         - nm：标签缺失标志（未被正确使用）
#         - nf：标签找到标志（存在标签文件则 =1）
#         - ne：空标签标志（未使用）
#         - nc：损坏标志（异常捕获设为 1）
#         - msg：警告消息
#         - segments：list，每个元素是 (Pi,2) 的分割顶点数组
#         - keypoints：shape (N,nkpt,ndim) 的关键点张量
#         - lb：最终返回的仅含 (cls,cx,cy,w,h) 的二维数组
#     """
#     im_file, lb_file, prefix, keypoint, num_cls, nkpt, ndim = args
#     # Number (missing, found, empty, corrupt), message, segments, keypoints
#     """
#     nm, nf, ne, nc：四个计数/标记
#         - nm (no label / missing)：标签文件缺失标记 (1=缺失)
#         - nf (found)：标签文件找到标记 (1=找到)
#         - ne (empty)：标签文件存在但内容为空 (1=空) —— 本函数没有真正设置它
#         - nc (corrupt)：图像或标签解析异常 (1=损坏或错误)
#     msg：警告/提示信息字符串
#     segments：存放分割多边形点序列的列表，每个元素是一个 (N,2) 的 numpy 数组
#     keypoints：关键点张量 (对象数, nkpt, ndim)；初始为 None
#     """
#     nm, nf, ne, nc, msg, segments, keypoints = 0, 0, 0, 0, '', [], None
#     try:
#         # Verify images
#         im = Image.open(im_file) # 用 PIL 打开图像，不读取完整像素到内存（延迟读取）
#         im.verify()  # PIL verify,PIL 的快速一致性校验，查看文件头是否损坏。成功后该 Image 对象不能再直接用来读取像素（verify 会丢流），但这里后面不再用它做像素处理，只取尺寸，所以 OK
#         shape = exif_size(im)  # image size, 调用项目里的 exif_size，读取 EXIF 方向信息，返回“正确方向”下的 (宽, 高)
#         shape = (shape[1], shape[0])  # hw, 转成 (高, 宽) 顺序，后续代码习惯使用 (h,w)
#         assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
#         assert im.format.lower() in IMG_FORMATS, f'invalid image format {im.format}'
#         """
#         专门处理 JPEG 结尾标记：
#             - f.seek(-2, 2) 跳到文件倒数第二个字节位置
#             - f.read() 读取最后两个字节，正常 JPEG 应为 b'\xff\xd9'
#             - 若不同，视为损坏，尝试用 ImageOps.exif_transpose(...).save 重新保存成标准 JPEG，并记录 msg 警告
#         """
#         if im.format.lower() in ('jpg', 'jpeg'):
#             with open(im_file, 'rb') as f:
#                 f.seek(-2, 2)
#                 if f.read() != b'\xff\xd9':  # corrupt JPEG
#                     ImageOps.exif_transpose(Image.open(im_file)).save(im_file, 'JPEG', subsampling=0, quality=100)
#                     msg = f'{prefix}WARNING ⚠️ {im_file}: corrupt JPEG restored and saved'        
#         # 标签解析部分
#         if os.path.isfile(lb_file): # 判断标签文件是否存在。存在则            
#             nf = 1  # label found,nf = 1 表示找到标签
#             with open(lb_file) as f:                
#                 lb = [x.split() for x in f.read().strip().splitlines() if len(x)] # 打开文件读取全部行，strip 去掉首尾空白，splitlines 分割行，再逐行 split() 按空白分隔成 token 列表               
#                 """
#                 segments硬编码切片:
#                     - 1+4 = 5 ：表示前 5 个字段是 cls, cx, cy, w, h。
#                     - 3*17 = 51 ：表示有 17 个关键点，每个 3 个数 (x,y,vis)。
#                     - 所以 x[1+4+3*17:] 等价 x[56:]，即从第 57 个 token 开始都是分割多边形坐标。
#                     - 将剩余 token 转成 float32，再 reshape(-1,2) 把 (x1,y1,x2,y2,...) 变成二维数组 [[x1,y1],[x2,y2],...]。
#                     - 得到的 segments 是一个列表，其长度 = 这一张图片中标注的实例（行）数。
#                 """
#                 segments = [np.array(x[1+4+3*17:], dtype=np.float32).reshape(-1, 2) for x in lb]                    
#                 """
#                 - 把每行截成前 56 个 token（类别 + bbox4 + 51 个关键点字段），丢弃多边形点部分。
#                 - 得到新的 lb（仍是列表的列表）
#                 """
#                 lb = [lbx[:1+4+3*17] for lbx in lb] 
#                 """
#                 把上面的二维列表转为 numpy 数组，形状 (对象数, 56)。
#                     - 列 0：类别
#                     - 列 1~4：bbox(cx,cy,w,h) 归一化
#                     - 列 5~55：17*3=51 个关键点字段顺序 (kpt1_x,kpt1_y,kpt1_vis,kpt2_x,...)
#                 注意：若某行 token 不够 56 个，会抛异常，但这里没有显式检查，会在后面 reshape 或索引时报错
                
#                 标签文件不存在的情况：
#                     - 代码里 if 块结束后没有 else，对“标签缺失”并没有设置 nm=1，也没有给 lb 一个默认空数组。
#                     - 这是一个缺陷：如果标签不存在，lb 变量未定义，后续会直接抛异常进入 except。
#                 """
#                 lb = np.array(lb, dtype=np.float32)
#         """
#         - 先切出列 5~55（共 51 个值）
#         - reshape 成 (对象数, nkpt, ndim)；这里依赖调用时传入的 nkpt=17, ndim=3 与硬编码 3*17 一致才安全。
#         - 如果你后面想换 nkpt=4，就会错，因为前面硬编码还是按 3*17 截
#         """
#         keypoints = lb[:, 5:5+3*17].reshape(-1, nkpt, ndim)
#         """
#         - 把 lb 减到只保留前 5 列 (cls + bbox)。
#         - 即函数最终返回的 lb 不再含关键点；关键点单独作为 keypoints 返回；分割点在 segments
#         """
#         lb = lb[:, :5]
#         # 返回一个 10 元组，供上游数据集加载器汇总
#         return im_file, lb, shape, segments, keypoints, nm, nf, ne, nc, msg
#     except Exception as e:
#         # 捕获任意异常，将 nc=1（标记这一组图像/标签损坏），构造警告字符串，然后返回全 None/计数标记的占位数组
#         nc = 1
#         msg = f'{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}'
#         return [None, None, None, None, None, nm, nf, ne, nc, msg]


def verify_image_label_seg_pose(args):
    """
    通用多任务标签解析 (检测 + 关键点 + 分割)。
    标签行格式:
        cls cx cy w h (kpt_x kpt_y kpt_vis) * nkpt  seg_x1 seg_y1 seg_x2 seg_y2 ...
    说明:
        - bbox 与关键点 (x,y) 建议为 0~1 归一化
        - kpt_vis 不强制范围，可自行约定 (0/1/2)，缺失可用负坐标表示（此处不过滤）
        - 分割点数量若 >0 必须为偶数
    返回:
        (im_file, lb(N,5), shape(h,w), segments(list[np.ndarray]), keypoints(N,nkpt,ndim),
         nm, nf, ne, nc, msg)
    """
    im_file, lb_file, prefix, keypoint_flag, num_cls, nkpt, ndim = args
    nm = nf = ne = nc = 0
    msg = ''
    segments, keypoints = [], None
    try:
        # -------- 图像校验 --------
        im = Image.open(im_file)
        im.verify()
        shape_wh = exif_size(im)              # (w,h) 方向修正
        shape = (shape_wh[1], shape_wh[0])    # -> (h,w)
        assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
        assert im.format.lower() in IMG_FORMATS, f'invalid image format {im.format}'
        if im.format.lower() in ('jpg', 'jpeg'):
            with open(im_file, 'rb') as f:
                f.seek(-2, 2)
                if f.read() != b'\xff\xd9':
                    # 尝试修复 JPEG 尾标记
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, 'JPEG', subsampling=0, quality=100)
                    msg = f'{prefix}WARNING ⚠️ {im_file}: corrupt JPEG restored and saved'

        # -------- 标签解析 --------
        if os.path.isfile(lb_file):
            nf = 1
            with open(lb_file) as f:
                rows = [r.split() for r in f.read().strip().splitlines() if r.strip()]

            kp_total = nkpt * ndim
            min_tokens = 5 + kp_total
            if len(rows) == 0:
                ne = 1
                lb = np.zeros((0, 5), dtype=np.float32)
                keypoints = np.zeros((0, nkpt, ndim), dtype=np.float32)
                segments = []
                return im_file, lb, shape, segments, keypoints, nm, nf, ne, nc, msg

            lb_records = []
            segments = []
            for li, tokens in enumerate(rows):
                tlen = len(tokens)
                if tlen < min_tokens:
                    raise ValueError(f'Row {li} token count {tlen} < required {min_tokens} '
                                     f'(cls + bbox4 + {nkpt}*{ndim})')

                # 基本字段
                cls = float(tokens[0])
                bbox = list(map(float, tokens[1:5]))
                kps_flat = list(map(float, tokens[5:5 + kp_total]))
                seg_tokens = tokens[5 + kp_total:]

                # 分割多边形
                if seg_tokens:
                    if len(seg_tokens) % 2 != 0:
                        raise ValueError(f'Row {li} segment token count {len(seg_tokens)} not even')
                    seg = np.array(list(map(float, seg_tokens)), dtype=np.float32).reshape(-1, 2)
                else:
                    seg = np.zeros((0, 2), dtype=np.float32)
                segments.append(seg)

                lb_records.append([cls, *bbox, *kps_flat])

            lb_full = np.array(lb_records, dtype=np.float32)  # (N, 5 + kp_total)

            # -------- 可选校验（放宽坐标允许出现 <0 以表示缺失） --------
            # 类别范围
            if lb_full.shape[0]:
                max_cls = int(lb_full[:, 0].max())
                if num_cls is not None:
                    assert max_cls <= num_cls, \
                        f'Label class {max_cls} exceeds dataset class count {num_cls} (0~{num_cls-1})'
            # bbox 0~1 校验（允许略超出可改成宽松模式）
            if lb_full.shape[0]:
                if not ((lb_full[:, 1:5] >= 0).all() and (lb_full[:, 1:5] <= 1).all()):
                    msg += f'{prefix}WARNING ⚠️ {im_file}: bbox out of [0,1] range detected; please confirm normalization. '

            # 关键点 reshape
            keypoints = lb_full[:, 5:5 + kp_total].reshape(-1, nkpt, ndim)

            # 仅保留前 5 列作为检测标签
            lb = lb_full[:, :5]

        else:
            # 标签缺失
            nm = 1
            lb = np.zeros((0, 5), dtype=np.float32)
            keypoints = np.zeros((0, nkpt, ndim), dtype=np.float32)
            segments = []

        return im_file, lb, shape, segments, keypoints, nm, nf, ne, nc, msg

    except Exception as e:
        nc = 1
        msg = f'{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}'
        return [None, None, None, None, None, nm, nf, ne, nc, msg]




def polygon2mask(imgsz, polygons, color=1, downsample_ratio=1):
    """
    Args:
        imgsz (tuple): The image size.
        polygons (list[np.ndarray]): [N, M], N is the number of polygons, M is the number of points(Be divided by 2).
        color (int): color
        downsample_ratio (int): downsample ratio
    """
    mask = np.zeros(imgsz, dtype=np.uint8)
    polygons = np.asarray(polygons)
    polygons = polygons.astype(np.int32)
    shape = polygons.shape
    polygons = polygons.reshape(shape[0], -1, 2)
    cv2.fillPoly(mask, polygons, color=color)
    nh, nw = (imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio)
    # NOTE: fillPoly firstly then resize is trying the keep the same way
    # of loss calculation when mask-ratio=1.
    mask = cv2.resize(mask, (nw, nh))
    return mask


def polygons2masks(imgsz, polygons, color, downsample_ratio=1):
    """
    Args:
        imgsz (tuple): The image size.
        polygons (list[np.ndarray]): each polygon is [N, M], N is number of polygons, M is number of points (M % 2 = 0)
        color (int): color
        downsample_ratio (int): downsample ratio
    """
    masks = []
    for si in range(len(polygons)):
        mask = polygon2mask(imgsz, [polygons[si].reshape(-1)], color, downsample_ratio)
        masks.append(mask)
    return np.array(masks)


def polygons2masks_overlap(imgsz, segments, downsample_ratio=1):
    """Return a (640, 640) overlap mask."""
    masks = np.zeros((imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio),
                     dtype=np.int32 if len(segments) > 255 else np.uint8)
    areas = []
    ms = []
    for si in range(len(segments)):
        mask = polygon2mask(imgsz, [segments[si].reshape(-1)], downsample_ratio=downsample_ratio, color=1)
        ms.append(mask)
        areas.append(mask.sum())
    areas = np.asarray(areas)
    index = np.argsort(-areas)
    ms = np.array(ms)[index]
    for i in range(len(segments)):
        mask = ms[i] * (i + 1)
        masks = masks + mask
        masks = np.clip(masks, a_min=0, a_max=i + 1)
    return masks, index


def check_det_dataset(dataset, autodownload=True):
    """Download, check and/or unzip dataset if not found locally."""
    data = check_file(dataset)

    # Download (optional)
    extract_dir = ''
    if isinstance(data, (str, Path)) and (zipfile.is_zipfile(data) or is_tarfile(data)):
        new_dir = safe_download(data, dir=DATASETS_DIR, unzip=True, delete=False, curl=False)
        data = next((DATASETS_DIR / new_dir).rglob('*.yaml'))
        extract_dir, autodownload = data.parent, False

    # Read yaml (optional)
    if isinstance(data, (str, Path)):
        data = yaml_load(data, append_filename=True)  # dictionary

    # Checks
    for k in 'train', 'val':
        if k not in data:
            raise SyntaxError(
                emojis(f"{dataset} '{k}:' key missing ❌.\n'train' and 'val' are required in all data YAMLs."))
    if 'names' not in data and 'nc' not in data:
        raise SyntaxError(emojis(f"{dataset} key missing ❌.\n either 'names' or 'nc' are required in all data YAMLs."))
    if 'names' in data and 'nc' in data and len(data['names']) != data['nc']:
        raise SyntaxError(emojis(f"{dataset} 'names' length {len(data['names'])} and 'nc: {data['nc']}' must match."))
    if 'names' not in data:
        data['names'] = [f'class_{i}' for i in range(data['nc'])]
    else:
        data['nc'] = len(data['names'])

    data['names'] = check_class_names(data['names'])

    # Resolve paths
    path = Path(extract_dir or data.get('path') or Path(data.get('yaml_file', '')).parent)  # dataset root

    if not path.is_absolute():
        path = (DATASETS_DIR / path).resolve()
    data['path'] = path  # download scripts
    for k in 'train', 'val', 'test':
        if data.get(k):  # prepend path
            if isinstance(data[k], str):
                x = (path / data[k]).resolve()
                if not x.exists() and data[k].startswith('../'):
                    x = (path / data[k][3:]).resolve()
                data[k] = str(x)
            else:
                data[k] = [str((path / x).resolve()) for x in data[k]]

    # Parse yaml
    train, val, test, s = (data.get(x) for x in ('train', 'val', 'test', 'download'))
    if val:
        val = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]  # val path
        if not all(x.exists() for x in val):
            name = clean_url(dataset)  # dataset name with URL auth stripped
            m = f"\nDataset '{name}' images not found ⚠️, missing paths %s" % [str(x) for x in val if not x.exists()]
            if s and autodownload:
                LOGGER.warning(m)
            else:
                m += f"\nNote dataset download directory is '{DATASETS_DIR}'. You can update this in '{SETTINGS_YAML}'"
                raise FileNotFoundError(m)
            t = time.time()
            if s.startswith('http') and s.endswith('.zip'):  # URL
                safe_download(url=s, dir=DATASETS_DIR, delete=True)
                r = None  # success
            elif s.startswith('bash '):  # bash script
                LOGGER.info(f'Running {s} ...')
                r = os.system(s)
            else:  # python script
                r = exec(s, {'yaml': data})  # return None
            dt = f'({round(time.time() - t, 1)}s)'
            s = f"success ✅ {dt}, saved to {colorstr('bold', DATASETS_DIR)}" if r in (0, None) else f'failure {dt} ❌'
            LOGGER.info(f'Dataset download {s}\n')
    check_font('Arial.ttf' if is_ascii(data['names']) else 'Arial.Unicode.ttf')  # download fonts

    return data  # dictionary


def check_cls_dataset(dataset: str, split=''):
    """
    Check a classification dataset such as Imagenet.

    This function takes a `dataset` name as input and returns a dictionary containing information about the dataset.
    If the dataset is not found, it attempts to download the dataset from the internet and save it locally.

    Args:
        dataset (str): Name of the dataset.
        split (str, optional): Dataset split, either 'val', 'test', or ''. Defaults to ''.

    Returns:
        data (dict): A dictionary containing the following keys and values:
            'train': Path object for the directory containing the training set of the dataset
            'val': Path object for the directory containing the validation set of the dataset
            'test': Path object for the directory containing the test set of the dataset
            'nc': Number of classes in the dataset
            'names': List of class names in the dataset
    """
    data_dir = (DATASETS_DIR / dataset).resolve()
    if not data_dir.is_dir():
        LOGGER.info(f'\nDataset not found ⚠️, missing path {data_dir}, attempting download...')
        t = time.time()
        if dataset == 'imagenet':
            subprocess.run(f"bash {ROOT / 'yolo/data/scripts/get_imagenet.sh'}", shell=True, check=True)
        else:
            url = f'https://github.com/ultralytics/yolov5/releases/download/v1.0/{dataset}.zip'
            download(url, dir=data_dir.parent)
        s = f"Dataset download success ✅ ({time.time() - t:.1f}s), saved to {colorstr('bold', data_dir)}\n"
        LOGGER.info(s)
    train_set = data_dir / 'train'
    val_set = data_dir / 'val' if (data_dir / 'val').exists() else None  # data/test or data/val
    test_set = data_dir / 'test' if (data_dir / 'test').exists() else None  # data/val or data/test
    if split == 'val' and not val_set:
        LOGGER.info("WARNING ⚠️ Dataset 'split=val' not found, using 'split=test' instead.")
    elif split == 'test' and not test_set:
        LOGGER.info("WARNING ⚠️ Dataset 'split=test' not found, using 'split=val' instead.")

    nc = len([x for x in (data_dir / 'train').glob('*') if x.is_dir()])  # number of classes
    names = [x.name for x in (data_dir / 'train').iterdir() if x.is_dir()]  # class names list
    names = dict(enumerate(sorted(names)))
    return {'train': train_set, 'val': val_set or test_set, 'test': test_set or val_set, 'nc': nc, 'names': names}


class HUBDatasetStats():
    """
    Class for generating HUB dataset JSON and `-hub` dataset directory

    Arguments
        path:           Path to data.yaml or data.zip (with data.yaml inside data.zip)
        task:           Dataset task. Options are 'detect', 'segment', 'pose', 'classify'.
        autodownload:   Attempt to download dataset if not found locally

    Usage
        from ultralytics.yolo.data.utils import HUBDatasetStats
        stats = HUBDatasetStats('/Users/glennjocher/Downloads/coco8.zip', task='detect')  # detect dataset
        stats = HUBDatasetStats('/Users/glennjocher/Downloads/coco8-seg.zip', task='segment')  # segment dataset
        stats = HUBDatasetStats('/Users/glennjocher/Downloads/coco8-pose.zip', task='pose')  # pose dataset
        stats.get_json(save=False)
        stats.process_images()
    """

    def __init__(self, path='coco128.yaml', task='detect', autodownload=False):
        """Initialize class."""
        LOGGER.info(f'Starting HUB dataset checks for {path}....')
        zipped, data_dir, yaml_path = self._unzip(Path(path))
        try:
            # data = yaml_load(check_yaml(yaml_path))  # data dict
            data = check_det_dataset(yaml_path, autodownload)  # data dict
            if zipped:
                data['path'] = data_dir
        except Exception as e:
            raise Exception('error/HUB/dataset_stats/yaml_load') from e

        self.hub_dir = Path(str(data['path']) + '-hub')
        self.im_dir = self.hub_dir / 'images'
        self.im_dir.mkdir(parents=True, exist_ok=True)  # makes /images
        self.stats = {'nc': len(data['names']), 'names': list(data['names'].values())}  # statistics dictionary
        self.data = data
        self.task = task  # detect, segment, pose, classify

    @staticmethod
    def _find_yaml(dir):
        """Return data.yaml file."""
        files = list(dir.glob('*.yaml')) or list(dir.rglob('*.yaml'))  # try root level first and then recursive
        assert files, f'No *.yaml file found in {dir}'
        if len(files) > 1:
            files = [f for f in files if f.stem == dir.stem]  # prefer *.yaml files that match dir name
            assert files, f'Multiple *.yaml files found in {dir}, only 1 *.yaml file allowed'
        assert len(files) == 1, f'Multiple *.yaml files found: {files}, only 1 *.yaml file allowed in {dir}'
        return files[0]

    def _unzip(self, path):
        """Unzip data.zip."""
        if not str(path).endswith('.zip'):  # path is data.yaml
            return False, None, path
        unzip_dir = unzip_file(path, path=path.parent)
        assert unzip_dir.is_dir(), f'Error unzipping {path}, {unzip_dir} not found. ' \
                                   f'path/to/abc.zip MUST unzip to path/to/abc/'
        return True, str(unzip_dir), self._find_yaml(unzip_dir)  # zipped, data_dir, yaml_path

    def _hub_ops(self, f):
        """Saves a compressed image for HUB previews."""
        compress_one_image(f, self.im_dir / Path(f).name)  # save to dataset-hub

    def get_json(self, save=False, verbose=False):
        """Return dataset JSON for Ultralytics HUB."""
        from ultralytics.yolo.data import YOLODataset  # ClassificationDataset

        def _round(labels):
            """Update labels to integer class and 4 decimal place floats."""
            if self.task == 'detect':
                coordinates = labels['bboxes']
            elif self.task == 'segment':
                coordinates = [x.flatten() for x in labels['segments']]
            elif self.task == 'pose':
                n = labels['keypoints'].shape[0]
                coordinates = np.concatenate((labels['bboxes'], labels['keypoints'].reshape(n, -1)), 1)
            else:
                raise ValueError('Undefined dataset task.')
            zipped = zip(labels['cls'], coordinates)
            return [[int(c), *(round(float(x), 4) for x in points)] for c, points in zipped]

        for split in 'train', 'val', 'test':
            if self.data.get(split) is None:
                self.stats[split] = None  # i.e. no test set
                continue

            dataset = YOLODataset(img_path=self.data[split],
                                  data=self.data,
                                  use_segments=self.task == 'segment',
                                  use_keypoints=self.task == 'pose')
            x = np.array([
                np.bincount(label['cls'].astype(int).flatten(), minlength=self.data['nc'])
                for label in tqdm(dataset.labels, total=len(dataset), desc='Statistics')])  # shape(128x80)
            self.stats[split] = {
                'instance_stats': {
                    'total': int(x.sum()),
                    'per_class': x.sum(0).tolist()},
                'image_stats': {
                    'total': len(dataset),
                    'unlabelled': int(np.all(x == 0, 1).sum()),
                    'per_class': (x > 0).sum(0).tolist()},
                'labels': [{
                    Path(k).name: _round(v)} for k, v in zip(dataset.im_files, dataset.labels)]}

        # Save, print and return
        if save:
            stats_path = self.hub_dir / 'stats.json'
            LOGGER.info(f'Saving {stats_path.resolve()}...')
            with open(stats_path, 'w') as f:
                json.dump(self.stats, f)  # save stats.json
        if verbose:
            LOGGER.info(json.dumps(self.stats, indent=2, sort_keys=False))
        return self.stats

    def process_images(self):
        """Compress images for Ultralytics HUB."""
        from ultralytics.yolo.data import YOLODataset  # ClassificationDataset

        for split in 'train', 'val', 'test':
            if self.data.get(split) is None:
                continue
            dataset = YOLODataset(img_path=self.data[split], data=self.data)
            with ThreadPool(NUM_THREADS) as pool:
                for _ in tqdm(pool.imap(self._hub_ops, dataset.im_files), total=len(dataset), desc=f'{split} images'):
                    pass
        LOGGER.info(f'Done. All images saved to {self.im_dir}')
        return self.im_dir


def compress_one_image(f, f_new=None, max_dim=1920, quality=50):
    """
    Compresses a single image file to reduced size while preserving its aspect ratio and quality using either the
    Python Imaging Library (PIL) or OpenCV library. If the input image is smaller than the maximum dimension, it will
    not be resized.

    Args:
        f (str): The path to the input image file.
        f_new (str, optional): The path to the output image file. If not specified, the input file will be overwritten.
        max_dim (int, optional): The maximum dimension (width or height) of the output image. Default is 1920 pixels.
        quality (int, optional): The image compression quality as a percentage. Default is 50%.

    Usage:
        from pathlib import Path
        from ultralytics.yolo.data.utils import compress_one_image
        for f in Path('/Users/glennjocher/Downloads/dataset').rglob('*.jpg'):
            compress_one_image(f)
    """
    try:  # use PIL
        im = Image.open(f)
        r = max_dim / max(im.height, im.width)  # ratio
        if r < 1.0:  # image too large
            im = im.resize((int(im.width * r), int(im.height * r)))
        im.save(f_new or f, 'JPEG', quality=quality, optimize=True)  # save
    except Exception as e:  # use OpenCV
        LOGGER.info(f'WARNING ⚠️ HUB ops PIL failure {f}: {e}')
        im = cv2.imread(f)
        im_height, im_width = im.shape[:2]
        r = max_dim / max(im_height, im_width)  # ratio
        if r < 1.0:  # image too large
            im = cv2.resize(im, (int(im_width * r), int(im_height * r)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(f_new or f), im)


def delete_dsstore(path):
    """
    Deletes all ".DS_store" files under a specified directory.

    Args:
        path (str, optional): The directory path where the ".DS_store" files should be deleted.

    Usage:
        from ultralytics.yolo.data.utils import delete_dsstore
        delete_dsstore('/Users/glennjocher/Downloads/dataset')

    Note:
        ".DS_store" files are created by the Apple operating system and contain metadata about folders and files. They
        are hidden system files and can cause issues when transferring files between different operating systems.
    """
    # Delete Apple .DS_store files
    files = list(Path(path).rglob('.DS_store'))
    LOGGER.info(f'Deleting *.DS_store files: {files}')
    for f in files:
        f.unlink()


def zip_directory(dir, use_zipfile_library=True):
    """
    Zips a directory and saves the archive to the specified output path.

    Args:
        dir (str): The path to the directory to be zipped.
        use_zipfile_library (bool): Whether to use zipfile library or shutil for zipping.

    Usage:
        from ultralytics.yolo.data.utils import zip_directory
        zip_directory('/Users/glennjocher/Downloads/playground')

        zip -r coco8-pose.zip coco8-pose
    """
    delete_dsstore(dir)
    if use_zipfile_library:
        dir = Path(dir)
        with zipfile.ZipFile(dir.with_suffix('.zip'), 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for file_path in dir.glob('**/*'):
                if file_path.is_file():
                    zip_file.write(file_path, file_path.relative_to(dir))
    else:
        import shutil
        shutil.make_archive(dir, 'zip', dir)

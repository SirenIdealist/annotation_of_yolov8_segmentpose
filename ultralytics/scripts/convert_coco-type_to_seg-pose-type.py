import os
import json
from pathlib import Path
import numpy as np
import shutil
"""
这个文件是用来将coco格式的标注文件转换为适合于该多任务的seg_pose_type的数据形式，具体来说：
- 路径下有annotations子路径，里面分别存放着训练集和验证集的含目标检测、实例分割和关键点检测的标注信息的json文件，如：
    - peduncle_train_503.json
    - peduncle_val_145.json
- 路径下有images子路径，里面分别存放着训练集和验证集的图片文件，如：
    - images/train/xxx.jpg ...
    - images/val/xxx.jpg ...

运行时只需要传入该路径的绝对路径参数，即可得到：
- labels子路径，里面分别存放着训练集和验证集的标签txt文件，每个图片对应一个同名的txt文件，即：
    - labels/train/xxx.txt ...
    - labels/val/xxx.txt ...
    - 内容格式为：cls_id x_center y_center width height kpt1_x kpt1_y kpt1_vis ... kpt4_x kpt4_y kpt4_vis seg_point1_x seg_point1_y ...
- train.txt，里面存放着训练集所有图片的相对路径
- val.txt，里面存放着验证集所有图片的相对路径
    """

def ensure_dir(path, clean=False):
    if clean and os.path.exists(path):
        shutil.rmtree(path)
    Path(path).mkdir(parents=True, exist_ok=True)

def convert_one_json(json_path, label_dir, image_dir, txt_list_path, rel_img_prefix):
    # 删除已有标签目录和txt文件，实现覆盖
    ensure_dir(label_dir, clean=True)
    if os.path.exists(txt_list_path):
        os.remove(txt_list_path)

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    images = {img['id']: img for img in data['images']}
    imgToAnns = {}
    for ann in data['annotations']:
        imgToAnns.setdefault(ann['image_id'], []).append(ann)
    image_paths = []
    for img_id, img in images.items():
        file_name = img['file_name']
        w, h = img['width'], img['height']
        anns = imgToAnns.get(img_id, [])
        label_lines = []
        for ann in anns:
            if ann.get('iscrowd', False):
                continue
            bbox = np.array(ann['bbox'], dtype=np.float32)
            bbox[:2] += bbox[2:] / 2
            bbox[[0, 2]] /= w
            bbox[[1, 3]] /= h
            # keypoints
            kpts = np.array(ann.get('keypoints', [0]*12), dtype=np.float32).reshape(-1, 3)
            if len(kpts) < 4:
                kpts = np.zeros((4, 3), dtype=np.float32)
            kpts[:, 0] /= w
            kpts[:, 1] /= h
            # mask
            segs = ann.get('segmentation', [])
            mask_points = []
            if isinstance(segs, list) and len(segs) > 0:
                for seg in segs:
                    seg = np.array(seg, dtype=np.float32).reshape(-1, 2)
                    seg[:, 0] /= w
                    seg[:, 1] /= h
                    mask_points.extend(seg.flatten().tolist())
            # 拼接
            line = []
            line.append(0)  # cls_id, int
            line.extend([float(x) for x in bbox.tolist()])
            for kp in kpts:
                line.append(float(kp[0]))
                line.append(float(kp[1]))
                line.append(int(kp[2]))  # kpt_vis as int
            line.extend([float(x) for x in mask_points])
            # 格式化输出
            out = []
            out.append(f"{int(line[0])}")
            out.extend([f"{x:.6f}" for x in line[1:5]])
            for i in range(4):
                out.append(f"{line[5+i*3]:.6f}")
                out.append(f"{line[6+i*3]:.6f}")
                out.append(f"{int(line[7+i*3])}")
            out.extend([f"{x:.6f}" for x in line[17:]])
            label_lines.append(' '.join(out))
        label_path = os.path.join(label_dir, file_name.replace('.jpg', '.txt'))
        with open(label_path, 'w', encoding='utf-8') as f:
            for l in label_lines:
                f.write(l + '\n')
        rel_img_path = os.path.join(rel_img_prefix, file_name).replace("\\", "/")
        image_paths.append(rel_img_path)
    with open(txt_list_path, 'w', encoding='utf-8') as f:
        for p in image_paths:
            f.write(f'{p}\n')

if __name__ == '__main__':
    # 修改为你的数据根目录
    data_root = r"E:\source_code\annotation_of_yolov8_segmentpose\data\tomato_peduncle\seg_pose_type"
    ann_dir = os.path.join(data_root, 'annotations')
    img_dir = os.path.join(data_root, 'images')
    label_dir = os.path.join(data_root, 'labels')
    # 训练集
    convert_one_json(
        json_path=os.path.join(ann_dir, 'peduncle_train_503.json'),
        label_dir=os.path.join(label_dir, 'train'),
        image_dir=os.path.join(img_dir, 'train'),
        txt_list_path=os.path.join(data_root, 'train.txt'),
        rel_img_prefix='images/train'
    )
    # 验证集
    convert_one_json(
        json_path=os.path.join(ann_dir, 'peduncle_val_145.json'),
        label_dir=os.path.join(label_dir, 'val'),
        image_dir=os.path.join(img_dir, 'val'),
        txt_list_path=os.path.join(data_root, 'val.txt'),
        rel_img_prefix='images/val'
    )
    print('Done!')
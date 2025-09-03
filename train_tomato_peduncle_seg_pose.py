from ultralytics import YOLO

import argparse

parser = argparse.ArgumentParser(description='Process SegmentPoseModel.')
parser.add_argument('-w','--weights', default='', type=str, help='path to weights model')
parser.add_argument('--mode', default='train', type=str, help='train | val | export | predict expects')
parser.add_argument('-m','--model', default='m', type=str, help='n | s | m | l | x expects')
parser.add_argument('-i','--images', nargs='*', help='list of paths to images to predict')

DATA_PATH = r"E:\source_code\annotation_of_yolov8_segmentpose\ultralytics\datasets\tomato_peduncle_seg_pose.yaml"
MODEL_PATH = r"E:\source_code\annotation_of_yolov8_segmentpose\ultralytics\models\v8\tomato_peduncle_seg_pose.yaml"

if __name__ == '__main__':
    args = parser.parse_args()
    weights = args.weights if args.weights else f'yolov8{args.model}-seg.pt'
    model = YOLO(model=MODEL_PATH, task='segment_pose')
    print("--"*50)   
    if args.mode == 'train':        
        model.train(data=DATA_PATH, name=f'train_segmentposecoco_{args.model}', lr0=1e-4, epochs=200, batch=32, device='cpu')
        # model = YOLO(model=args.weights, task='segment_pose')
        # model.train(resume=True)
    elif args.mode == 'val':
        model.val(data=DATA_PATH, name=f'val_segmentposecoco_{args.model}')
    elif args.mode == 'export':
        model.export(format='onnx', device=0, simplify=True, dynamic=True) 
    elif args.mode == 'predict':
        if len(args.images):            
            model.predict(source=args.images, name=f'predict_segmentposecoco_{args.model}', imgsz=640, save=True, task='segment_pose')    
    else:
        print('train | val | export expects')

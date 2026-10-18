from pathlib import Path
import argparse
import random
import numpy as np
import cv2
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data_COCO128"
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
MODEL_PATH = OUT / "model.keras"
NAMES = ["person", "rider", "car", "bus", "truck", "bike", "motor", "traffic light", "traffic sign", "train"]
IMG_SIZE = 64

def preprocess(crop):
    return cv2.cvtColor(cv2.resize(crop, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0

def load_crops(split, limit):
    images_dir = DATASET / 'images'
    labels_dir = DATASET / 'labels'
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(f'Missing dataset folders: {images_dir} or {labels_dir}')
    files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png'))
    if limit > 0: files = files[:limit]
    xs, ys = [], []
    for path in files:
        label = labels_dir / (path.stem + '.txt')
        if not label.exists(): continue
        im = cv2.imread(str(path))
        if im is None: continue
        h,w = im.shape[:2]
        for line in label.read_text(encoding='utf-8').splitlines():
            parts=line.split()
            if len(parts)!=5: continue
            try:
                cls=int(parts[0]); xc,yc,bw,bh=map(float,parts[1:])
            except ValueError: continue
            if not 0 <= cls < len(NAMES): continue
            x1=max(0,int((xc-bw/2)*w)); y1=max(0,int((yc-bh/2)*h))
            x2=min(w,int((xc+bw/2)*w)); y2=min(h,int((yc+bh/2)*h))
            if x2<=x1 or y2<=y1: continue
            xs.append(preprocess(im[y1:y2,x1:x2])); ys.append(cls)
    if not xs: raise RuntimeError(f'No labeled crops in {images_dir}. Check image names and YOLO labels.')
    return np.asarray(xs,np.float32),np.asarray(ys,np.int32)

def make_model():
    inputs=tf.keras.Input((IMG_SIZE,IMG_SIZE,3))
    x=tf.keras.layers.Conv2D(32,3,padding='same',activation='relu')(inputs)
    x=tf.keras.layers.MaxPooling2D()(x)
    x=tf.keras.layers.Conv2D(64,3,padding='same',activation='relu')(x)
    x=tf.keras.layers.MaxPooling2D()(x)
    x=tf.keras.layers.Conv2D(128,3,padding='same',activation='relu')(x)
    x=tf.keras.layers.MaxPooling2D(pool_size=4)(x)
    x=tf.keras.layers.Flatten()(x)
    x=tf.keras.layers.Dense(128,activation='relu')(x)
    outputs=tf.keras.layers.Dense(len(NAMES),activation='softmax')(x)
    model=tf.keras.Model(inputs,outputs,name='Fast_RCNN_educational_region_classifier')
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return model

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--image',default=None)
    p.add_argument('--split',choices=['train','val','test'],default='test')
    args=p.parse_args()
    
    images_dir = DATASET / 'images'
    labels_dir = DATASET / 'labels'
    
    if args.image: 
        path = Path(args.image)
    else:
        if not images_dir.is_dir(): 
            raise FileNotFoundError(str(images_dir))
        
        # กรองเอาเฉพาะรูปภาพที่มี label ใน 10 คลาสแรก
        valid_files = []
        for p_file in images_dir.iterdir():
            if p_file.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                lbl = labels_dir / (p_file.stem + '.txt')
                if lbl.exists():
                    for line in lbl.read_text(encoding='utf-8').splitlines():
                        parts = line.split()
                        if parts and 0 <= int(parts[0]) < len(NAMES):
                            valid_files.append(p_file)
                            break
                            
        if not valid_files: 
            raise RuntimeError('No labeled images with target classes found')
        path = random.choice(valid_files)
        
    frame = cv2.imread(str(path))
    if frame is None: 
        raise FileNotFoundError(str(path))
        
    label = labels_dir / (path.stem + '.txt')
    if not label.exists(): 
        raise FileNotFoundError(f'Ground-truth YOLO label required: {label}')
    if not MODEL_PATH.exists(): 
        raise FileNotFoundError(str(MODEL_PATH))
        
    h, w = frame.shape[:2]
    crops = []
    boxes = []
    
    for line in label.read_text(encoding='utf-8').splitlines():
        parts = line.split()
        if len(parts) != 5: 
            continue
        try: 
            cls = int(parts[0])
            xc, yc, bw, bh = map(float, parts[1:])
        except ValueError: 
            continue
        if not 0 <= cls < len(NAMES): 
            continue
            
        x1 = max(0, int((xc - bw / 2) * w))
        y1 = max(0, int((yc - bh / 2) * h))
        x2 = min(w, int((xc + bw / 2) * w))
        y2 = min(h, int((yc + bh / 2) * h))
        
        if x2 <= x1 or y2 <= y1: 
            continue
            
        crops.append(preprocess(frame[y1:y2, x1:x2]))
        boxes.append((x1, y1, x2, y2))
        
    if not crops: 
        raise RuntimeError('No valid labeled objects in image')
        
    model = tf.keras.models.load_model(MODEL_PATH)
    predictions = model.predict(np.asarray(crops, np.float32), verbose=0)
    
    for (x1, y1, x2, y2), pred in zip(boxes, predictions):
        cls = int(np.argmax(pred))
        score = float(np.max(pred))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{NAMES[cls]} {score:.2f}', (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 2)
        
    target = OUT / 'image_result.jpg'
    cv2.imwrite(str(target), frame)
    print(f'Input: {path}; Objects: {len(boxes)}; Saved: {target}')
    print('Ground-truth boxes used; not standalone object detection.')

if __name__ == '__main__': 
    main()
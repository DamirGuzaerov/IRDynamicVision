from ultralytics import YOLO

def main():
    model = YOLO('yolov8l.pt')

    model.train(data="data.yaml", epochs=46, imgsz=640, fliplr = 0.5,copy_paste = 0.1, mixup = 1, batch = 2, device = 0, workers = 3)

if __name__ == 'main':
    main()
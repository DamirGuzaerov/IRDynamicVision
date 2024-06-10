import os
import cv2
import numpy as np
import argparse
from datetime import datetime
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from collections import defaultdict

# Парсинг аргументов командной строки
parser = argparse.ArgumentParser(description='YOLO Object Detection')
parser.add_argument('--source', type=str, default='0', help='Source of video: 0 for webcam or path to video file')
parser.add_argument('--real_time', action='store_true', default=False, help='Show real-time output')
parser.add_argument('--save', type=int, choices=[0, 1], default=1, help='Save output video: 1 for true, 0 for false')
args = parser.parse_args()

track_history = defaultdict(lambda: [])
model_path = os.path.join('.', 'runs', 'detect', 'train19', 'weights', 'best.pt')
model = YOLO(model_path)
names = model.model.names

source = args.source if args.source != '0' else 0
cap = cv2.VideoCapture(source)
assert cap.isOpened(), "Error opening video source"

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Генерация временной метки и создание директории для результатов
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
result_dir = os.path.join('result', f'detect_{timestamp}')
os.makedirs(result_dir, exist_ok=True)

# Обновление пути видео и лог-файлов
result_video_path = os.path.join(result_dir, 'object_tracking.mp4')
log_file_path = os.path.join(result_dir, 'detection_log.txt')

result = None
if args.save:
    result = cv2.VideoWriter(result_video_path,
                             cv2.VideoWriter_fourcc(*'mp4v'),
                             fps,
                             (w, h))

with open(log_file_path, 'w') as log_file:
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            results = model.track(frame, persist=True, verbose=False)
            boxes = results[0].boxes.xyxy.cpu()
            if results[0].boxes.id is not None:
                clss = results[0].boxes.cls.cpu().tolist()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                confs = results[0].boxes.conf.float().cpu().tolist()

                annotator = Annotator(frame, line_width=2)
                for box, cls, track_id, conf in zip(boxes, clss, track_ids, confs):
                    annotator.box_label(box, color=colors(int(cls), True), label=names[int(cls)])

                    # Логирование обнаружений
                    log_file.write(
                        f"Track ID: {track_id}, Class: {names[int(cls)]}, BBox: {box.numpy().tolist()}, Confidence: {conf}\n")

                    track = track_history[track_id]
                    track.append((int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)))
                    if len(track) > 30:
                        track.pop(0)

                    # Полный трек
                    points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.circle(frame, track[-1], 7, colors(int(cls), True), -1)
                    cv2.polylines(frame, [points], isClosed=False, color=colors(int(cls), True), thickness=2)

            if args.real_time:
                cv2.imshow('YOLO Object Detection', frame)

            if args.save:
                result.write(frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    if args.save:
        result.release()
cap.release()
cv2.destroyAllWindows()
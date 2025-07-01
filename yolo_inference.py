from ultralytics import YOLO 

model = YOLO('yolov8n')
# model.to('cuda')

results = model.predict('input_videos/08fd33_4.mp4',save=True,device='0')
print(results[0])
print('=====================================')
for box in results[0].boxes:
    print(box)

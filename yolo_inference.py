from ultralytics import YOLO 

model = YOLO('models/best.pt')
# model.to('cuda')

results = model.predict('input_videos/08fd33_4.mp4',save=True,device='0', stream=True)

# print(results[0])
# print('=====================================')
# for box in results[0].boxes:
#     print(box)

for r in results:
    print(r)  # or r.boxes, r.probs, etc.
    # break  # remove this to process all frames

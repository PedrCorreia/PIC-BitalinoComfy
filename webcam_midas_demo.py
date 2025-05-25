import cv2
import torch
import matplotlib.pyplot as plt

# Load MiDaS model and transform
midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
midas.to('cpu').eval()
midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = midas_transforms.small_transform

cap = cv2.VideoCapture(0)
plt.ion()
fig, ax = plt.subplots()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).unsqueeze(0).to('cpu')

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode='bicubic',
            align_corners=False
        ).squeeze()
        output = prediction.cpu().numpy()

    ax.clear()
    ax.imshow(output, cmap='magma')
    plt.pause(0.001)

    cv2.imshow('Webcam', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()

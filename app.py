import streamlit as st
import torch
import timm
import torchvision.transforms as transforms
from PIL import Image
import io
import cv2
import numpy as np
import requests

# Download the file from GitHub
model_url = "https://github.com/RThaweewat/streamlit-ome/blob/main/pretrained.pt?raw=true"
response = requests.get(model_url)
response.raise_for_status()


# Load the model from the downloaded content
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model("convnext_tiny", pretrained=False, num_classes=4)
model.load_state_dict(torch.load(io.BytesIO(response.content), map_location=device))
model.to(device)
model.eval()


def auto_crop_circle_image(image):
    img = cv2.imread(image, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    thresholded = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 2)

    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = max(contours, key=cv2.contourArea)
    center, radius = cv2.minEnclosingCircle(largest_contour)

    x_center, y_center, radius = int(center[0]), int(center[1]), int(radius)

    x1, y1 = x_center - radius, y_center - radius
    x2, y2 = x_center + radius, y_center + radius
    cropped_img = img[y1:y2, x1:x2]

    mask = np.zeros_like(cropped_img)
    cv2.circle(mask, (radius, radius), radius, (255, 255, 255), -1)

    masked_img = cv2.bitwise_and(cropped_img, mask)
    masked_img = Image.fromarray(masked_img)

    transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(masked_img).unsqueeze(0)


# Define the prediction function
def predict(image, model, class_labels):
    image_tensor = (auto_crop_circle_image(image)).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        scores = torch.nn.functional.softmax(outputs, dim=1)

    class_probabilities = {class_label: score.item() for class_label, score in zip(class_labels, scores[0])}
    return class_probabilities


# Streamlit app
st.title("Image Classification with TIMM PyTorch Model")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

class_labels = ['aom', 'normal', 'ome', 'prev_ome']

if uploaded_file is not None:
    image = Image.open(io.BytesIO(uploaded_file.getvalue()))
    st.image(image, caption="Uploaded image", use_column_width=True)
    st.write("")

    if st.button("Classify"):
        class_probabilities = predict(image, model, class_labels)

        st.subheader("Class probabilities")
        for class_label, score in class_probabilities.items():
            st.write(f"{class_label}: {score:.4f}")

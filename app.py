import streamlit as st
from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import numpy as np
import GroundingDINO.groundingdino.datasets.transforms as T
from PIL import Image
import os
import wget

st.title('Grounding DINO')
# Model Backbone
backbone = st.selectbox('Choose Backbone:', ['Swin-T', 'Swin-B'])
if backbone == 'Swin-T':
    CONFIG_PATH = os.path.join('GroundingDINO', 'groundingdino', 'config', 'GroundingDINO_SwinT_OGC.py')
    WEIGHTS_PATH = "groundingdino_swint_ogc.pth"
    if not os.path.exists(WEIGHTS_PATH):
        wget.download('https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth')
else:
    CONFIG_PATH = os.path.join('GroundingDINO', 'groundingdino', 'config', 'GroundingDINO_SwinB_cfg.py')
    WEIGHTS_PATH = "groundingdino_swinb_cogcoor.pth"
    if not os.path.exists(WEIGHTS_PATH):
        wget.download('https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth')

# Transformation
transform = T.Compose(
    [
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

BOX_TRESHOLD = st.sidebar.slider('Box Threshold:', min_value=0.0, max_value=1.0, value=0.35)
TEXT_TRESHOLD = st.sidebar.slider('Text Threshold:', min_value=0.0, max_value=1.0, value=0.25)

model = load_model(CONFIG_PATH, WEIGHTS_PATH)

upload_img_file = st.file_uploader(
    'Upload Image', type=['jpg', 'jpeg', 'png'])
FRAME_WINDOW = st.image([])
if upload_img_file is not None:
    file_bytes = np.asarray(
        bytearray(upload_img_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    image = np.asarray(img)
    image_transformed, _ = transform(img, None)

    FRAME_WINDOW.image(img, channels='BGR')

    # TEXT_PROMPT = "glass most to the right"
    TEXT_PROMPT = st.text_input('Text Prompt:', value="glass most to the right")
    if st.checkbox('Predict'):
        boxes, logits, phrases = predict(
            model=model, 
            image=image_transformed, 
            caption=TEXT_PROMPT, 
            box_threshold=BOX_TRESHOLD, 
            text_threshold=TEXT_TRESHOLD
        )

        annotated_frame = annotate(image_source=image, boxes=boxes, logits=logits, phrases=phrases)
        FRAME_WINDOW.image(annotated_frame, channels='BGR')

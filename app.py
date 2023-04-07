import streamlit as st
import os
from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate
import supervision as sv
import cv2
import numpy as np
import GroundingDINO.groundingdino.datasets.transforms as T
from typing import Tuple
from PIL import Image


st.title('Grounding DINO')

CONFIG_PATH = 'GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
WEIGHTS_PATH = "groundingdino_swint_ogc.pth"
transform = T.Compose(
    [
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

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

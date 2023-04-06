import os
from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate
import supervision as sv


# CONFIG_PATH = os.path.join('HOME', "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
CONFIG_PATH = 'GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
print(CONFIG_PATH, "; exist:", os.path.isfile(CONFIG_PATH))

# !wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth


WEIGHTS_NAME = "groundingdino_swint_ogc.pth"
WEIGHTS_PATH = os.path.join(WEIGHTS_NAME)
print(WEIGHTS_PATH, "; exist:", os.path.isfile(WEIGHTS_PATH))



model = load_model(CONFIG_PATH, WEIGHTS_PATH)

IMAGE_NAME = "img.jpeg"
IMAGE_PATH = os.path.join(IMAGE_NAME)

# TEXT_PROMPT = "chair with man sitting on it"
TEXT_PROMPT = "glass most to the right"
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

image_source, image = load_image(IMAGE_PATH)

boxes, logits, phrases = predict(
    model=model, 
    image=image, 
    caption=TEXT_PROMPT, 
    box_threshold=BOX_TRESHOLD, 
    text_threshold=TEXT_TRESHOLD
)

annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)

# %matplotlib inline
sv.plot_image(annotated_frame, (16, 16))

FROM ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive
COPY . /home
WORKDIR /home
RUN apt-get update && \
    apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    wget

RUN pip install torch torchvision torchaudio
RUN pip install -r GroundingDINO/requirements.txt
RUN pip install streamlit
RUN pip install -e ./GroundingDINO/
RUN wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

CMD [ "streamlit", "run", "app.py" ]
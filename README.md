# Streamlit-GroundingDINO
[<img src="https://img.shields.io/badge/Docker-Image-blue.svg?logo=docker">](<https://hub.docker.com/repository/docker/naseemap47/streamlit-dino>) <br>
Dashboard for Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection

## Installtion
1. **Clone this Repo**
```
git clone https://github.com/naseemap47/Streamlit-GroundingDINO.git
cd Streamlit-GroundingDINO
```
2. **Install Libraries**
```
pip install torch torchvision torchaudio
pip install -r GroundingDINO/requirements.txt
pip install streamlit wget
pip install -e ./GroundingDINO/
```

## Run Dashboard
```
streamlit run app.py
```
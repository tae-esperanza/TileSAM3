# TileSAM3
Official repo of TileSAM3 for dropwise condensation segmentation.
#

## Datasets
Both the Low Quality Droplet Dataset (LQDD) and High Quality Droplet Dataset (HQDD) are open access:

|[LQDD](https://drive.google.com/drive/folders/1nZv3meBD8oM4h-nt6HmWO_LPMpwvaWXR?usp=sharing)|[HQDD](https://drive.google.com/drive/folders/1KtezOJSyhnrzKGK5uSNPEc89IpskY7-1?usp=sharing)|
#

## Model Weights
Fine-tuned SAM3 weights are available to use for applications:
[Trained on Folds 1-4 of LQDD](https://drive.google.com/file/d/13cu06Ubx0C697S8wOn63u0QpUZ-YmlcW/view?usp=sharing)

## Usage instructions

Install SAM3 repository
```cli
cd TileSAM3
pip install -e .
git clone https://github.com/facebookresearch/sam3.git
```
Create SAM3 processor
```python
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import torch

bpe_path = f"./sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
model = build_sam3_image_model(bpe_path=bpe_path)
ckpt_path = "./checkpoints/checkpoint.pt"
model.load_state_dict(torch.load(ckpt_path, map_location="cuda")['model'], strict=False)

processor = Sam3Processor(model)
```

Call tileSAM3 pipeline
```python
from TileSAM3.tile_SAM3 import tileSAM3
from PIL import Image

image = Image.open("./images/example.jpg")

results = tileSAM3(
    processor=processor,
    image=image,
    prompt="droplet",
    layers=4,
)
```

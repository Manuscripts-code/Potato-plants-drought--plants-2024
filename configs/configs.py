from pathlib import Path

import mlflow
import numpy as np
from torchvision import transforms

import data_loader.transformations as transforms_hyp

### Registry and log ###
BASE_DIR = Path(__file__).parent.parent.absolute()
MODEL_REGISTRY = Path(BASE_DIR, "experiments")
LOGS_DIR = Path(BASE_DIR, "logs")

MODEL_REGISTRY.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

TRACKING_URI = "file:///" + str(MODEL_REGISTRY.absolute())
mlflow.set_tracking_uri(TRACKING_URI)


### Data loading ###
DATA_DIR = "E:\\janez\\data\\sliced_converted_images\\slikanje_2022_rad_4"
CASHED_IMAGES_DIR = "./data"
USE_CASHED_IMAGES = True
SAVE_CASHED_IMAGES = True
NOISY_BANDS = np.concatenate(
	[np.arange(26), np.arange(140, 171), np.arange(430, 448)]
)  # hardcoded bands to remove
IMG_SIZE = 50
GROUPS = {
	# groupes by labels
	"KK-K": "KIS_krka_control",
	"KK-S": "KIS_krka_drought",
	"KS-K": "KIS_savinja_control",
	"KS-S": "KIS_savinja_drought",
}
TRANSFORM_TRAIN = transforms.Compose(
	[
		transforms_hyp.RandomMirror(),
		transforms_hyp.RandomCrop(IMG_SIZE),
		transforms.ToTensor(),
	]
)
TRANSFORM_TEST = transforms.Compose(
	[
		transforms_hyp.Rescale(IMG_SIZE),
		transforms.ToTensor(),
	]
)
TRANSFORM_DURING_LOADING = transforms.Compose([transforms_hyp.NoTransformation()])

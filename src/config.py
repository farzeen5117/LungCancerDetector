from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_ZIP_PATH = DATA_DIR / "data.zip"
IMAGES_PATH = DATA_DIR / "images"
CLASSES = ["lung_n", "lung_aca", "lung_scc"]

IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 50
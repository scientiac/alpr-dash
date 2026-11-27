"""All tunable settings in one place."""

from pathlib import Path

# ---------- Models (downloaded once, then run locally) ----------
DETECTOR_MODEL = "yolo-v9-t-512-license-plate-end2end"   # finds plates in a frame
OCR_MODEL = "cct-s-v2-global-model"                       # reads the plate text
DETECT_CONF = 0.4          # minimum detector confidence for a plate box
THREADS = 4                # CPU threads per model (more is not faster)

# ---------- Vehicle details (run once per logged vehicle) ----------
VEHICLE_MODEL = "rf-detr-nano-384-coco"                   # finds the vehicle around a plate
VEHICLE_CONF = 0.4
VEHICLE_TYPES = ("car", "motorcycle", "bus", "truck")     # COCO classes that count
CLIP_REPO = "Xenova/mobileclip_blt"                       # names make and colour (zero-shot)
CLIP_FILES = ("onnx/vision_model_int8.onnx", "onnx/text_model_int8.onnx", "tokenizer.json")
CLIP_SIZE = 224            # input size of that model
MAKE_MIN_CONF = 0.5        # below this the make is stored as unknown
COLOURS = ("white", "black", "silver", "grey", "red", "blue", "green",
           "yellow", "orange", "brown", "beige")
# Makes the model chooses from, by vehicle type. Add the ones common where you are.
MAKES = ("Alfa Romeo", "Audi", "BMW", "BYD", "Chevrolet", "Chrysler", "Citroen", "Dacia",
         "Daewoo", "Fiat", "Ford", "Honda", "Hyundai", "Jaguar", "Jeep", "Kia", "Lada",
         "Land Rover", "Lexus", "Mahindra", "Maruti Suzuki", "Mazda", "Mercedes-Benz",
         "MG", "Mini", "Mitsubishi", "Nissan", "Opel", "Peugeot", "Porsche", "Renault", "Rover",
         "Saab", "Seat", "Skoda", "Smart", "Subaru", "Suzuki", "Tata", "Tesla", "Toyota",
         "Volkswagen", "Volvo")
BUS_TRUCK_MAKES = ("Ashok Leyland", "DAF", "Eicher", "Iveco", "MAN", "Mercedes-Benz",
                   "Scania", "Solaris", "Tata", "Volvo")
MOTORCYCLE_MAKES = ("Bajaj", "BMW", "Ducati", "Harley-Davidson", "Hero", "Honda",
                    "Kawasaki", "KTM", "Royal Enfield", "Suzuki", "TVS", "Yamaha")

# ---------- Tracking and voting ----------
MATCH_IOU = 0.2            # box overlap needed to continue a track
MAX_JUMP = 1.5             # ...or centre distance, in plate widths
OCR_EVERY = 2              # min frames between OCR runs on one track
MAX_READS = 12             # keep this many best readings per track
LOST_FRAMES = 15           # frames without a detection before a track ends
MIN_HITS = 3               # detections a track needs before it is logged
MIN_READS = 2              # OCR readings needed before logging
MIN_AGREE = 0.6            # readings must agree with the voted text this much
MIN_CONF = 0.5             # OCR confidence (0-1) needed to keep a reading
MIN_LEN, MAX_LEN = 4, 10   # plausible plate text length
COOLDOWN_S = 30            # same plate seen again within this time = same pass
MIN_SIMILAR = 0.7          # plates this alike on a same-looking vehicle = same pass

# ---------- Direction ----------
# A plate that grows (or moves down the picture) is coming towards the camera.
TOWARDS_CAMERA = "in"      # set to "out" if the camera faces the exit
MIN_GROWTH = 1.15          # plate width change that counts as moving
MIN_SHIFT = 0.05           # ...or vertical movement, as a share of frame height

# ---------- Input / output ----------
TESTSET_DIR = Path(__file__).resolve().parent.parent / "testset"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
STREAM_PREFIXES = ("rtsp://", "rtmp://", "http://", "https://", "udp://", "tcp://")
DATA_DIR = Path("data")                 # everything recorded goes here
DB_FILE = DATA_DIR / "plates.db"        # SQLite database
IMAGE_DIR = DATA_DIR / "images"         # plate and vehicle photos (JPEG)
JPEG_SAVE_QUALITY = 90

# ---------- Dashboard ----------
HOST, PORT = "127.0.0.1", 8000
STREAM_WIDTH = 960         # preview frames are scaled to this width
JPEG_QUALITY = 75

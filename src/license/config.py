"""All tunable settings in one place."""

from pathlib import Path

# ---------- Models (downloaded once, then run locally) ----------
DETECTOR_MODEL = "yolo-v9-t-512-license-plate-end2end"   # finds plates in a frame
OCR_MODEL = "cct-s-v2-global-model"                       # reads the plate text
DETECT_CONF = 0.4          # minimum detector confidence for a plate box
THREADS = 4                # CPU threads per model (more is not faster)

# ---------- Vehicles: detection and tracking (ByteTrack) ----------
VEHICLE_MODEL = "rf-detr-nano-384-coco"                   # finds vehicles (COCO classes)
VEHICLE_THREADS = 8        # this model is larger and gains from more threads
VEHICLE_EVERY = 3          # detect vehicles every Nth frame; the tracker predicts between
VEHICLE_EVERY_LIVE = 2     # ...for live streams, which already skip frames to keep up
VEHICLE_TYPES = ("car", "motorcycle", "bus", "truck")     # COCO classes that count
TRACK_HIGH = 0.5           # ByteTrack: confident detections, matched first
TRACK_LOW = 0.15           # ...weak detections, used only to keep existing tracks alive
TRACK_NEW = 0.6            # confidence needed to start a new vehicle track
TRACK_MATCH_IOU = 0.2      # minimum box overlap for a match
TRACK_BUFFER = 30          # frames a lost vehicle track is kept before it ends
MIN_VEHICLE_HITS = 6       # detections a vehicle needs before it is recorded
MIN_VEHICLE_MOVE = 0.08    # ...and how far it must move (share of frame size);
                           # parked cars stay still and are not recorded
MIN_VEHICLE_AREA = 0.10    # ...and how big it must get (share of frame area): only vehicles
                           # close enough for their plate to be readable count
RECORD_VEHICLES_WITHOUT_PLATE = True   # record moving vehicles whose plate is never seen
REJOIN_S = 3.0             # a vehicle track that restarts this soon, where the last one ended
                           # and looking the same, is the same vehicle (it was hidden for a moment)

# ---------- Vehicle details (run once per record) ----------
# Colour, make and van use MobileCLIP, whose weights are licensed for
# non-commercial research only. Set to False to run without it (type still works).
DESCRIBE_VEHICLES = True
CLIP_REPO = "Xenova/mobileclip_blt"                       # names make and colour (zero-shot)
CLIP_FILES = ("onnx/vision_model_int8.onnx", "onnx/text_model_int8.onnx", "tokenizer.json")
CLIP_SIZE = 224            # input size of that model
MAKE_MIN_CONF = 0.5        # below this the make is stored as unknown
VAN_MIN_CONF = 0.7         # COCO has no vans; CLIP renames a car/truck "van" above this
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

# ---------- Recognition status ----------
# recognized: confident reading; review: readable but uncertain, a person
# should check it; unrecognized: plate unreadable or not found.
REVIEW_CONF = 0.9          # voted confidence below this -> review
REVIEW_AGREE = 0.85        # agreement between readings below this -> review
REVIEW_READS = 4           # fewer readings than this -> review
UNREADABLE_MIN_HITS = 8    # a plate seen this often but never read is recorded as unreadable
ENHANCE_BELOW = 0.8        # OCR retried on an enhanced crop when confidence is below this

# ---------- Direction ----------
# A plate that grows (or moves down the picture) is coming towards the camera.
TOWARDS_CAMERA = "in"      # set to "out" if the camera faces the exit
MIN_GROWTH = 1.15          # plate width change that counts as moving
MIN_SHIFT = 0.05           # ...or vertical movement, as a share of frame height

# ---------- Input / output ----------
TESTSET_DIR = Path(__file__).resolve().parent.parent / "testset"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
STREAM_PREFIXES = ("rtsp://", "rtmp://", "http://", "https://", "udp://", "tcp://")
MAX_UPLOAD_MB = 2048                    # largest video that can be uploaded
DATA_DIR = Path("data")                 # everything recorded goes here (private to the user)
DB_FILE = DATA_DIR / "plates.db"        # SQLite database
IMAGE_DIR = DATA_DIR / "images"         # plate and vehicle photos (JPEG)
UPLOAD_DIR = DATA_DIR / "uploads"       # uploaded videos
VIDEO_DIR = DATA_DIR / "videos"         # processed videos with the detections drawn on
JPEG_SAVE_QUALITY = 90
SAVE_ANNOTATED = True                   # write the annotated video for uploads and files

# ---------- Dashboard ----------
HOST, PORT = "127.0.0.1", 8000
STREAM_WIDTH = 960         # preview frames are scaled to this width
JPEG_QUALITY = 75

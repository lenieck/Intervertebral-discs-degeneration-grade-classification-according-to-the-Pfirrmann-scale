from ultralytics import YOLO

model = YOLO("yolo26n-cls.pt")

results = model.train(
    data="/SPIDER_training_margin5_patient_split",
    epochs=50,
    imgsz=320,
    batch=32,
    device=0,

    hsv_h=0.015,
    hsv_s=0.07,
    hsv_v=0.04,
    fliplr=0.5,
    flipud=0.1,
    scale=0.05,
    translate=0.05,
    degrees=5.0,
    shear=2.0,
    perspective=0.002,
    erasing=0.1,

    mosaic=0.3,
    mixup=0.1,
    cutmix=0.05,

    auto_augment="randaugment",
    patience=10,

    lr0=0.005,
)


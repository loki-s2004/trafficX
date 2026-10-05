import cv2
from ultralytics import YOLO

# ================== LOAD MODELS ==================
helmet_model = YOLO("runs/detect/train3/weights/best.pt")
person_model = YOLO("yolo11s.pt")

# ================== LOAD IMAGE ==================
img = cv2.imread("sample/bike2.jpg")
if img is None:
    print("Image not found")
    exit()

# ================== PERSON & BIKE ==================
results = person_model(img, conf=0.25)

persons = []
bikes = []

for r in results:
    for box in r.boxes:
        label = person_model.names[int(box.cls[0])]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if label == "person":
            persons.append((x1, y1, x2, y2))
        elif label in ["motorbike", "motorcycle"]:
            bikes.append((x1, y1, x2, y2))

# ================== TRIPLE RIDING ==================
valid_riders = []  # store riders for helmet filtering

for bx1, by1, bx2, by2 in bikes:

    bike_w = bx2 - bx1
    bike_h = by2 - by1
    bike_cx = (bx1 + bx2) // 2
    bike_cy = (by1 + by2) // 2

    riders = []

    for px1, py1, px2, py2 in persons:

        pcx = (px1 + px2) // 2
        pcy = (py1 + py2) // 2

        # ✅ IMPROVED CONDITIONS
        horizontal_ok = abs(pcx - bike_cx) < bike_w * 0.6

        vertical_ok = (
            py2 > by1 - bike_h * 0.3 and
            py1 < by2
        )

        center_ok = abs(pcy - bike_cy) < bike_h

        if horizontal_ok and vertical_ok and center_ok:
            riders.append((px1, py1, px2, py2))

    # store riders for helmet filtering
    valid_riders.extend(riders)

    if len(riders) >= 3:
        cv2.rectangle(img, (bx1, by1), (bx2, by2), (0, 0, 255), 3)
        cv2.putText(
            img,
            "TRIPLE RIDING / VIOLATION",
            (bx1, by1 - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

# ================== NO HELMET (FILTERED) ==================
helmet_results = helmet_model(img, conf=0.2)

for r in helmet_results:
    for box in r.boxes:
        label = helmet_model.names[int(box.cls[0])]

        if label == "no_helmet":
            hx1, hy1, hx2, hy2 = map(int, box.xyxy[0])

            # ✅ ONLY CHECK RIDERS (ignore pedestrians)
            is_rider = False
            for px1, py1, px2, py2 in valid_riders:
                if hx1 > px1 and hx2 < px2 and hy2 > py1:
                    is_rider = True
                    break

            if not is_rider:
                continue  # 🚫 ignore non-riders

            cv2.rectangle(img, (hx1, hy1), (hx2, hy2), (0, 0, 255), 2)
            cv2.putText(
                img,
                "NO HELMET",
                (hx1, hy1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

# ================== DISPLAY ==================
cv2.imshow("TrafficX Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
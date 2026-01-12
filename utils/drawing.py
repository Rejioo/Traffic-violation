# import cv2

# def draw_results(image, persons, helmet_map, bike_info):
#     img = image.copy()

#     # draw persons + helmet status
#     for i, p in enumerate(persons):
#         x1, y1, x2, y2 = map(int, p)
#         has_helmet = helmet_map.get(i, False)

#         color = (0,255,0) if has_helmet else (0,0,255)
#         label = "HELMET" if has_helmet else "NO_HELMET"

#         cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
#         cv2.putText(img, label, (x1, y1-6),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#     # draw bikes + passenger count
#     for item in bike_info:
#         bx1, by1, bx2, by2 = map(int, item["bike"])
#         count = item["count"]
#         violation = item["passenger_violation"]

#         color = (0,0,255) if violation else (0,255,0)

#         cv2.rectangle(img, (bx1,by1), (bx2,by2), color, 3)
#         cv2.putText(img, f"Riders: {count}",
#                     (bx1, by1-10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

#         if violation:
#             cv2.putText(img, "PASSENGER VIOLATION",
#                         (bx1, by2+30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

#     return img
import cv2

def draw_results(
    image,
    persons,
    helmet_map,
    bike_info,
    red_light_map=None,
    signal_state=None,
    stop_line_y=None
):
    img = image.copy()

    # ---------- DRAW STOP LINE ----------
    if stop_line_y is not None:
        h, w, _ = img.shape
        cv2.line(img, (0, stop_line_y), (w, stop_line_y), (0, 0, 255), 2)

    # ---------- DRAW SIGNAL STATE ----------
    if signal_state is not None:
        color = (0, 0, 255) if signal_state == "RED" else (0, 255, 0)
        cv2.putText(
            img,
            f"Signal: {signal_state}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

    # ---------- DRAW PERSONS + HELMET + RED LIGHT ----------
    for i, p in enumerate(persons):
        x1, y1, x2, y2 = map(int, p)

        has_helmet = helmet_map.get(i, False)
        helmet_color = (0, 255, 0) if has_helmet else (0, 0, 255)
        helmet_label = "HELMET" if has_helmet else "NO_HELMET"

        cv2.rectangle(img, (x1, y1), (x2, y2), helmet_color, 2)
        cv2.putText(
            img,
            helmet_label,
            (x1, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            helmet_color,
            2
        )

        # red light violation overlay
        if red_light_map and red_light_map.get(i, False):
            cv2.putText(
                img,
                "RED LIGHT VIOLATION",
                (x1, y1 - 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

    # ---------- DRAW BIKES + PASSENGER COUNT ----------
    for item in bike_info:
        bx1, by1, bx2, by2 = map(int, item["bike"])
        count = item["count"]
        violation = item["passenger_violation"]

        color = (0, 0, 255) if violation else (0, 255, 0)

        cv2.rectangle(img, (bx1, by1), (bx2, by2), color, 3)
        cv2.putText(
            img,
            f"Riders: {count}",
            (bx1, by1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

        if violation:
            cv2.putText(
                img,
                "PASSENGER VIOLATION",
                (bx1, by2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

    return img

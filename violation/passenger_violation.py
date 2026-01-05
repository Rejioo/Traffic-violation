from utils.geometry import area, iou

def analyze_passengers(persons, motorcycles):
    bike_info = []

    for m in motorcycles:
        riders = []
        m_area = area(m)

        for p in persons:
            p_area = area(p)
            overlap = iou(p, m)

            px = (p[0] + p[2]) / 2
            py = p[3]

            inside = (m[0] < px < m[2]) and (m[1] < py < m[3])
            size_ok = (p_area / m_area) > 0.25

            if size_ok and (overlap > 0.25 or inside):
                riders.append(p)

        bike_info.append({
            "bike": m,
            "riders": riders,
            "count": len(riders),
            "passenger_violation": len(riders) > 2
        })

    return bike_info

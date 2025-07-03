import re

attribute_keywords = {
    "phòng ngủ": "bedrooms", 
    "phòng tắm": "bathrooms", 
    "tầng": "stories",
    "máy lạnh": "airconditioning", 
    "tầng hầm": "basement", 
    "đường chính": "mainroad",
    "nước nóng": "hotwaterheating", 
    "diện tích": "area", 
    "giá": "price",
    "chỗ đậu xe": "parking", 
    "nội thất": "furnishingstatus",
    "phòng khách": "guestroom", 
    "khu ưu tiên": "prefarea"
}

def parse_conditions(text):
    text = text.lower()
    conditions = []
    clauses = re.split(r" và |, ", text)
    for clause in clauses:
        for phrase, column in attribute_keywords.items():
            # Phủ định nội thất
            if ("không có nội thất" in clause or "không nội thất" in clause) and column == "furnishingstatus":
                conditions.append((column, "==", "unfurnished"))
                continue
            # Phủ định máy lạnh, nước nóng, basement...
            if ("không có " + phrase) in clause or ("không " + phrase) in clause:
                conditions.append((column, "==", "no"))
                continue
            # Khoảng từ ... đến ...
            match_range = re.search(r"từ (\d+) đến (\d+).*" + re.escape(phrase), clause)
            match_range2 = re.search(re.escape(phrase) + r".*từ (\d+) đến (\d+)", clause)
            if match_range:
                x, y = int(match_range.group(1)), int(match_range.group(2))
                conditions.append((column, ">=", x))
                conditions.append((column, "<=", y))
                continue
            if match_range2:
                x, y = int(match_range2.group(1)), int(match_range2.group(2))
                conditions.append((column, ">=", x))
                conditions.append((column, "<=", y))
                continue
            # Điều kiện nhỏ hơn/ít hơn/< (cả hai chiều)
            match_lt = re.search(rf"{re.escape(phrase)}.*(nhỏ hơn|ít hơn|dưới|<)\s*(\d+)", clause)
            match_lt2 = re.search(r"(nhỏ hơn|ít hơn|dưới|<)\s*(\d+).*" + re.escape(phrase), clause)
            if match_lt:
                val = int(match_lt.group(2))
                conditions.append((column, "<", val))
                continue
            if match_lt2:
                val = int(match_lt2.group(2))
                conditions.append((column, "<", val))
                continue
            # Điều kiện lớn hơn/nhiều hơn/> (cả hai chiều)
            match_gt = re.search(rf"{re.escape(phrase)}.*(lớn hơn|nhiều hơn|trên|>)\s*(\d+)", clause)
            match_gt2 = re.search(r"(lớn hơn|nhiều hơn|trên|>)\s*(\d+).*" + re.escape(phrase), clause)
            if match_gt:
                val = int(match_gt.group(2))
                conditions.append((column, ">", val))
                continue
            if match_gt2:
                val = int(match_gt2.group(2))
                conditions.append((column, ">", val))
                continue
            # Số lượng (ví dụ: 1 chỗ đậu xe)
            match_eq = re.search(r"(\d+)\s*" + re.escape(phrase), clause)
            if match_eq:
                val = int(match_eq.group(1))
                conditions.append((column, "==", val))
                continue
            # Có thuộc tính
            if phrase in clause:
                if column in ["airconditioning", "basement", "hotwaterheating", "mainroad", "guestroom", "prefarea"]:
                    conditions.append((column, "==", "yes"))
                elif column == "furnishingstatus":
                    conditions.append((column, "!=", "unfurnished"))
                    # Điều kiện bằng/ là/ =
                match_eq1 = re.search(rf"{re.escape(phrase)}.*(bằng|là|=)\s*(\d+)", clause)
                match_eq2 = re.search(r"(bằng|là|=)\s*(\d+).*" + re.escape(phrase), clause)
                if match_eq1:
                    val = int(match_eq1.group(2))
                    conditions.append((column, "==", val))
                    continue
                if match_eq2:
                    val = int(match_eq2.group(2))
                    conditions.append((column, "==", val))
                    continue
    return conditions
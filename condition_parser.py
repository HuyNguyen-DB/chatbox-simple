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
            # 1. Phủ định nội thất
            if ("không có nội thất" in clause or "không nội thất" in clause) and column == "furnishingstatus":
                conditions.append((column, "==", "unfurnished"))
                continue
            # 2. Phủ định các thuộc tính yes/no (mở rộng mẫu)
            neg_patterns = [
                f"không {phrase}",
                f"không ở {phrase}",
                f"không thuộc {phrase}",
                f"không nằm trong {phrase}",
                f"không có {phrase}",
                f"không nằm ở {phrase}"
            ]
            if any(p in clause for p in neg_patterns):
                if column == "furnishingstatus":
                    conditions.append((column, "==", "unfurnished"))
                else:
                    conditions.append((column, "==", "no"))
                continue
            # 3. Khẳng định các thuộc tính yes/no (mở rộng mẫu)
            pos_patterns = [
                f"{phrase}",
                f"có {phrase}",
                f"ở {phrase}",
                f"thuộc {phrase}",
                f"nằm trong {phrase}",
                f"nằm ở {phrase}"
            ]
            for p in pos_patterns:
                # Chỉ match nếu không có "không" phía trước
                if p in clause and f"không {p}" not in clause:
                    if column in ["airconditioning", "basement", "hotwaterheating", "mainroad", "guestroom", "prefarea"]:
                        conditions.append((column, "==", "yes"))
                        break
                    elif column == "furnishingstatus":
                        # Chỉ lấy furnished hoặc semi-furnished, không lấy unfurnished
                        conditions.append((column, "in", ["furnished", "semi-furnished"]))
                        break
            # 4. Khoảng từ ... đến ...
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
            # 5. Điều kiện nhỏ hơn/ít hơn/< (cả hai chiều)
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
            # 6. Điều kiện lớn hơn/nhiều hơn/> (cả hai chiều)
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
            # 7. Điều kiện bằng/ là/ =
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
            # 8. Số lượng (ví dụ: 1 chỗ đậu xe)
            match_num = re.search(r"(\d+)\s*" + re.escape(phrase), clause)
            if match_num:
                val = int(match_num.group(1))
                conditions.append((column, "==", val))
                continue
    return conditions
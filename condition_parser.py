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
    import re
    text = text.lower()
    conditions = []
    clauses = re.split(r" và |, ", text)
    for clause in clauses:
        for phrase, column in attribute_keywords.items():
            # Phủ định nội thất
            if ("không có nội thất" in clause or "không nội thất" in clause) and column == "furnishingstatus":
                conditions.append((column, "==", "unfurnished"))
            # Phủ định máy lạnh, nước nóng, basement...
            elif ("không có " + phrase) in clause or ("không " + phrase) in clause:
                conditions.append((column, "==", "no"))
            # Số lượng (ví dụ: 1 chỗ đậu xe)
            elif match := re.search(r"(\d+)\s*" + re.escape(phrase), clause):
                val = int(match.group(1))
                conditions.append((column, "==", val))
            # Khoảng từ ... đến ...
            elif match := re.search(r"từ (\d+) đến (\d+).*" + re.escape(phrase), clause):
                x, y = int(match.group(1)), int(match.group(2))
                conditions.append((column, ">=", x))
                conditions.append((column, "<=", y))
            # Có thuộc tính
            elif phrase in clause:
                if column in ["airconditioning", "basement", "hotwaterheating", "mainroad", "guestroom", "prefarea"]:
                    conditions.append((column, "==", "yes"))
                elif column == "furnishingstatus":
                    conditions.append((column, "!=", "unfurnished"))
    return conditions

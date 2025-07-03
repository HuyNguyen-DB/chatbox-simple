def detect_intent(text):
    text = text.lower()
    if any(x in text for x in ["bao nhiêu", "số lượng", "có mấy"]):
        return "count"
    elif any(x in text for x in ["trung bình", "giá trung bình", "diện tích trung bình"]):
        return "mean"
    elif any(x in text for x in ["tỷ lệ", "phần trăm", "chiếm bao nhiêu"]):
        return "ratio"
    elif any(x in text for x in ["liệt kê", "danh sách", "nhà", "căn nhà"]):
        return "list"
    else:
        return None
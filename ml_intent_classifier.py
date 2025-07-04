import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Câu hỏi mẫu để train model
# Chia thành 4 loại: count, mean, ratio, list
train_texts = [
    # Count (30)
    "Có bao nhiêu căn nhà", "Số lượng nhà", "Có mấy căn", "Tổng số nhà",
    "Có bao nhiêu căn có máy lạnh", "Có bao nhiêu nhà có 3 phòng ngủ",
    "Có bao nhiêu nhà có nước nóng", "Có bao nhiêu nhà có basement",
    "Có bao nhiêu nhà có 2 tầng", "Có bao nhiêu nhà có diện tích trên 8000",
    "Có bao nhiêu nhà không có máy lạnh", "Có bao nhiêu nhà không có nước nóng",
    "Có bao nhiêu nhà không có nội thất", "Số lượng nhà có 1 chỗ đậu xe",
    "Số lượng nhà có từ 2 đến 4 phòng tắm", "Có bao nhiêu nhà có 4 phòng tắm",
    "Có bao nhiêu nhà có diện tích lớn", "Có bao nhiêu nhà không có basement",
    "Có bao nhiêu nhà có nhiều tầng", "Có bao nhiêu nhà có ít phòng ngủ", "Có tổng cộng bao nhiêu căn nhà?",
    "Có bao nhiêu nhà ở khu ưu tiên", "Số căn nhà hiện có là bao nhiêu?", "Có tất cả mấy căn nhà?", "Có bao nhiêu căn nhà đang bán?",
    "Có bao nhiêu căn nhà phù hợp điều kiện?",
    "Hiện có bao nhiêu căn nhà trên hệ thống?",
    "Có tổng số bao nhiêu căn nhà đang rao bán?",
    "Cho tôi biết số lượng nhà hiện tại?",
    "Bao nhiêu căn nhà đáp ứng điều kiện tìm kiếm?",
    # Mean (30)
    "Giá trung bình", "Diện tích trung bình", "Trung bình giá", "Trung bình diện tích",
    "Giá nhà trung bình", "Diện tích nhà trung bình", "Diện tích trung bình nhà có basement",
    "Giá trung bình nhà có 2 tầng", "Trung bình số phòng ngủ của nhà có máy lạnh",
    "Nhà có nước nóng thì diện tích trung bình là bao nhiêu", "Diện tích trung bình của nhà có nước nóng",
    "Diện tích trung bình nhà có hotwaterheating", "Trung bình diện tích nhà có nước nóng",
    "Trung bình diện tích nhà có 1 phòng ngủ", "Trung bình giá nhà có 3 phòng ngủ",
    "Giá trung bình của nhà không có máy lạnh", "Diện tích trung bình của nhà không có nội thất",
    "Diện tích trung bình nhà có từ 4 đến 5 phòng ngủ", "Giá trung bình của nhà có 3 phòng ngủ là bao nhiêu",
    "Nhà không có máy lạnh thì giá trung bình bao nhiêu", "Trung bình giá nhà có nhiều tầng", "Trung bình giá nhà là bao nhiêu?",
    "Trung bình diện tích các căn nhà là bao nhiêu?", "Giá trung bình của nhà có nội thất là bao nhiêu?", 
    "Diện tích trung bình của nhà có máy lạnh là bao nhiêu?", "Trung bình mỗi căn nhà có bao nhiêu phòng ngủ?",
    "Trung bình mỗi căn nhà có bao nhiêu tầng?",
    "Giá bán trung bình của các căn nhà là bao nhiêu?",
    "Diện tích trung bình của nhà có tầng hầm là bao nhiêu?",
    "Trung bình mỗi căn nhà có mấy phòng tắm?",
    "Trung bình mỗi căn nhà có bao nhiêu chỗ đậu xe?",
    # Ratio (30)
    "Tỷ lệ nhà có máy lạnh", "Phần trăm nhà có nội thất", "Chiếm bao nhiêu phần trăm",
    "Tỷ lệ nhà có nước nóng", "Tỷ lệ nhà có 2 phòng tắm", "Tỷ lệ nhà có basement",
    "Tỷ lệ nhà có 3 tầng", "Tỷ lệ nhà có diện tích trên 9000", "Tỷ lệ nhà không có máy lạnh",
    "Tỷ lệ nhà không có nội thất", "Tỷ lệ nhà có nhiều tầng", "Tỷ lệ nhà có ít phòng ngủ",
    "Tỷ lệ nhà có từ 2 đến 4 phòng tắm", "Tỷ lệ nhà có 1 chỗ đậu xe", "Tỷ lệ nhà có diện tích lớn",
    "Tỷ lệ nhà có giá cao", "Tỷ lệ nhà có nội thất đầy đủ", "Tỷ lệ nhà không có basement",
    "Tỷ lệ nhà có guestroom", "Tỷ lệ nhà ở khu ưu tiên",
    "Tỷ lệ nhà có tầng hầm là bao nhiêu phần trăm?",
    "Bao nhiêu phần trăm nhà có nước nóng?",
    "Phần trăm nhà có nội thất là bao nhiêu?",
    "Có bao nhiêu phần trăm nhà có diện tích trên 5000?",
    "Tỷ lệ nhà có giá trên 10 triệu là bao nhiêu?",
    "Tỷ lệ nhà có chỗ đậu xe là bao nhiêu phần trăm?",
    "Bao nhiêu phần trăm nhà có phòng khách?",
    "Tỷ lệ nhà có nội thất bán được là bao nhiêu?",
    "Phần trăm nhà có máy lạnh là bao nhiêu?",
    "Chiếm bao nhiêu phần trăm nhà có tầng hầm?",
    # List (27)
    "Liệt kê nhà có máy lạnh", "Danh sách nhà có 3 phòng ngủ", "Nhà có 1 phòng ngủ",
    "Nhà có diện tích trên 8000", "Nhà có nội thất", "Nhà có nước nóng", "Nhà có 2 tầng",
    "Nhà không có máy lạnh", "Nhà không có nước nóng", "Nhà có basement", "Nhà có 4 phòng tắm",
    "Nhà có diện tích lớn", "Nhà có giá cao", "Nhà có nhiều tầng", "Nhà có ít phòng ngủ",
    "Liệt kê nhà có từ 2 đến 4 phòng tắm", "Liệt kê nhà không có nội thất",
    "Liệt kê nhà có 1 chỗ đậu xe", "Liệt kê nhà không có basement", "Tìm nhà có diện tích trên 3000", "Tìm nhà có nội thất",
    "Liệt kê các căn nhà có diện tích lớn hơn 5000",
    "Danh sách nhà có 2 tầng và có máy lạnh",
    "Cho tôi xem các căn nhà có giá dưới 8 triệu",
    "Hiển thị các căn nhà có nội thất đầy đủ",
    "Tìm các căn nhà có tầng hầm và nước nóng",
    "Liệt kê các căn nhà có giá dưới 5 triệu",
    "Cho tôi xem danh sách nhà có 3 tầng",
    "Hiển thị các căn nhà có phòng khách và nước nóng"
]
    

train_labels = (
    ["count"] * 30 +
    ["mean"] * 30 +
    ["ratio"] * 30 +
    ["list"] * 30
)

print("Số lượng train_texts:", len(train_texts))
print("Số lượng train_labels:", len(train_labels))

MODEL_PATH = "intent_model.joblib"
VEC_PATH = "intent_vectorizer.joblib"

def train_and_save():
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(train_texts)
    clf = LogisticRegression()
    clf.fit(X, train_labels)
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(vectorizer, VEC_PATH)

def ml_detect_intent(text):
    text_lower = text.lower()
    # Ưu tiên rule cho "tìm"
    if text_lower.strip().startswith("tìm") or "tìm nhà" in text_lower:
        return "list"
    # Nếu chưa có model thì huấn luyện
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VEC_PATH):
        train_and_save()
    clf = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VEC_PATH)
    X = vectorizer.transform([text])
    return clf.predict(X)[0]

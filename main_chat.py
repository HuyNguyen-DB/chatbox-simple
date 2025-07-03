from data_loader import load_data
from intent_classifier import detect_intent
from condition_parser import parse_conditions
from response_generator import generate_response
from ml_intent_classifier import ml_detect_intent  # Thay thế intent_classifier
import pandas as pd

def apply_conditions(df, conditions):
    for col, op, val in conditions:
        if col in df.columns:
            # Nếu là số, ép kiểu số
            if pd.api.types.is_numeric_dtype(df[col]):
                try:
                    val = float(val)
                except:
                    continue  # Bỏ qua nếu không ép được
            # Nếu là chuỗi, chuẩn hóa về chữ thường
            else:
                df[col] = df[col].astype(str).str.lower()
                val = str(val).lower()
        if op == "==":
            df = df[df[col] == val]
        elif op == ">":
            df = df[df[col] > val]
        elif op == "<":
            df = df[df[col] < val]
        elif op == ">=":
            df = df[df[col] >= val]
        elif op == "<=":
            df = df[df[col] <= val]
    return df

def main():
    df = load_data()
    print("🤖 Chatbox AI (Offline) – Nhập 'exit' để thoát.")
    while True:
        query = input("🧑 Bạn: ")
        if query.strip().lower() == "exit":
            print("👋 Tạm biệt!")
            break

        intent = ml_detect_intent(query)
        conditions = parse_conditions(query)
        df_filtered = apply_conditions(df.copy(), conditions)
        reply = generate_response(intent, df_filtered, df, query)
        print("🤖 Bot:", reply)
        intent = ml_detect_intent(query)  # Dùng ML intent
if __name__ == "__main__":
    main()

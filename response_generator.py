def generate_response(intent, df_filtered, df, text):
    if df_filtered.empty:
        return "⚠️ Không tìm thấy dữ liệu phù hợp với yêu cầu của bạn."

    if intent == "count":
        return f"📊 Có {len(df_filtered)} căn nhà phù hợp với điều kiện bạn đưa ra."
    elif intent == "mean":
        text = text.lower()
        if "giá" in text and "price" in df_filtered.columns:
            mean_val = df_filtered["price"].mean()
            return f"💰 Giá trung bình của các căn nhà là khoảng {mean_val:,.0f} INR."
        elif ("diện tích" in text or "area" in text) and "area" in df_filtered.columns:
            mean_val = df_filtered["area"].mean()
            return f"📏 Diện tích trung bình của các căn nhà là {mean_val:.2f} sqft."
        elif "area" in df_filtered.columns:
            mean_val = df_filtered["area"].mean()
            return f"📏 Diện tích trung bình của các căn nhà là {mean_val:.2f} sqft."
        else:
            numeric_cols = df_filtered.select_dtypes(include='number').columns.tolist()
            if numeric_cols:
                col = numeric_cols[0]
                mean_val = df_filtered[col].mean()
                return f"🔎 Giá trị trung bình của '{col}' là {mean_val:.2f}."
            else:
                return "❔ Bạn muốn tính trung bình theo tiêu chí nào?"
    elif intent == "ratio":
        ratio = len(df_filtered) / len(df)
        return f"📈 Khoảng {ratio * 100:.2f}% số căn nhà thỏa mãn điều kiện bạn đưa ra."
    elif intent == "list":
        preview = df_filtered.head(10)
        return f"📝 Danh sách một số căn nhà phù hợp:\n{preview.to_string(index=False)}"
    else:
        return "🤖 Tôi chưa hiểu rõ yêu cầu của bạn."
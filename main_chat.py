import tkinter as tk
from tkinter import scrolledtext, ttk
from data_loader import load_data
from condition_parser import parse_conditions
from response_generator import generate_response
from ml_intent_classifier import ml_detect_intent
import pandas as pd

def apply_conditions(df, conditions):
    for col, op, val in conditions:
        if col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                try:
                    val = float(val) if not isinstance(val, list) else val
                except:
                    continue
            else:
                df[col] = df[col].astype(str).str.lower()
                if isinstance(val, str):
                    val = val.lower()
                elif isinstance(val, list):
                    val = [str(v).lower() for v in val]
        if op == "==":
            df = df[df[col] == val]
        elif op == "!=":
            df = df[df[col] != val]
        elif op == "in":
            df = df[df[col].isin(val)]
        elif op == ">":
            df = df[df[col] > val]
        elif op == "<":
            df = df[df[col] < val]
        elif op == ">=":
            df = df[df[col] >= val]
        elif op == "<=":
            df = df[df[col] <= val]
    return df

class ChatboxGUI:
    def __init__(self, root):
        self.df = load_data()
        root.title("Chatbox AI (Offline)")
        root.geometry("1200x700")

        # Chat area
        self.text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=120, height=12, font=("Consolas", 13), state='disabled')
        self.text_area.pack(padx=10, pady=10, fill=tk.X)

        # Entry
        self.entry = tk.Entry(root, width=120, font=("Consolas", 13))
        self.entry.pack(padx=10, pady=(0,10), fill=tk.X)
        self.entry.bind("<Return>", self.send_message)

        # Table area
        self.table_frame = tk.Frame(root)
        self.table_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.tree = None

        self.text_area.config(state='normal')
        self.text_area.insert(tk.END, "🤖 Chatbox AI (Offline) – Gõ 'exit' để thoát.\n")
        self.text_area.config(state='disabled')

    def show_table(self, df):
        # Xóa bảng cũ nếu có
        if self.tree:
            self.tree.destroy()
        if df.empty:
            return
        self.tree = ttk.Treeview(self.table_frame, show='headings')
        self.tree.pack(fill=tk.BOTH, expand=True)
        style = ttk.Style()
        style.configure("Treeview.Heading", font=("Consolas", 12, "bold"))
        style.configure("Treeview", font=("Consolas", 11))
        # Đặt cột
        self.tree["columns"] = list(df.columns)
        for col in df.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=110, anchor=tk.CENTER)
        # Thêm dữ liệu
        for _, row in df.iterrows():
            self.tree.insert("", tk.END, values=list(row))

    def send_message(self, event=None):
        query = self.entry.get()
        if not query.strip():
            return
        self.text_area.config(state='normal')
        self.text_area.insert(tk.END, f"🧑 Bạn: {query}\n")
        if query.strip().lower() == "exit":
            self.text_area.insert(tk.END, "👋 Tạm biệt!\n")
            self.text_area.config(state='disabled')
            self.entry.config(state='disabled')
            self.text_area.update()
            self.entry.update()
            self.text_area.after(500, lambda: (self.text_area.master.quit(), self.text_area.master.destroy()))
            return
        intent = ml_detect_intent(query)
        conditions = parse_conditions(query)
        df_filtered = apply_conditions(self.df.copy(), conditions)
        reply = generate_response(intent, df_filtered, self.df, query)
        self.text_area.insert(tk.END, f"🤖 Bot: {reply}\n")
        self.text_area.see(tk.END)
        self.text_area.config(state='disabled')
        self.entry.delete(0, tk.END)

        # Nếu là intent "list", show bảng
        if intent == "list" and not df_filtered.empty:
            self.show_table(df_filtered.head(30))  # Hiện tối đa 30 dòng
        else:
            self.show_table(pd.DataFrame())  # Xóa bảng nếu không phải intent list

if __name__ == "__main__":
    root = tk.Tk()
    gui = ChatboxGUI(root)
    root.mainloop()
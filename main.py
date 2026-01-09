import tkinter as tk
from tkinter import messagebox
import sqlite3
import pyperclip
import keyboard
import threading
import torch  # Import torch để tránh lỗi trong logic tìm kiếm
import os
import sys
from sentence_transformers import SentenceTransformer, util

# --- CẤU HÌNH ---
DB_NAME = 'data.db'
# Tên thư mục chứa model nằm cạnh file exe
LOCAL_MODEL_FOLDER = 'ai_model'
HOTKEY_TOGGLE = 'ctrl+shift+space'
SIMILARITY_THRESHOLD = 0.5

class AIOverlayApp:
    def __init__(self, root):
        self.root = root
        self.setup_window()

        # Biến lưu trữ
        self.questions = []
        self.answers = []
        self.embeddings = None

        # UI Components
        self.create_widgets()

        # Load AI & Data
        self.status_label.config(text="Đang tải AI...")
        threading.Thread(target=self.load_ai_and_data, daemon=True).start()

        # Đăng ký phím tắt
        keyboard.add_hotkey(HOTKEY_TOGGLE, self.toggle_visibility)

    def setup_window(self):
        # Cấu hình cửa sổ
        self.root.title("AI Quick Search")

        # --- THAY ĐỔI 1: KÍCH THƯỚC NHỎ HƠN ---
        # Cũ: 600x50 -> Mới: 300x35 (Nhỏ bằng một nửa)
        self.root.geometry("300x35+100+100")

        self.root.overrideredirect(True)
        self.root.attributes('-topmost', True)
        self.root.configure(bg='#fff')

        # --- THAY ĐỔI 2: LÀM MỜ GIAO DIỆN ---
        # 1.0 là rõ nhất, 0.1 là tàng hình.
        # 0.8 là mức vừa phải dễ nhìn
        self.root.attributes('-alpha', 0.6)

        # Biến kéo thả
        self._offsetx = 0
        self._offsety = 0

    def create_widgets(self):
        # 1. Drag handle (Nhỏ lại một chút cho cân đối)
        self.drag_frame = tk.Frame(self.root, bg='#5f6368', width=15, cursor="fleur")
        self.drag_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.drag_frame.bind('<Button-1>', self.click_window)
        self.drag_frame.bind('<B1-Motion>', self.drag_window)

        # 2. Thanh tìm kiếm
        self.search_entry = tk.Entry(
            self.root,
            bg='#fff',
            fg='#5f6368',
            insertbackground='#5f6368',
            font=('Segoe UI', 10),  # Giảm font xuống 10 cho vừa khung nhỏ
            bd=0,
            highlightthickness=0
        )
        self.search_entry.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.search_entry.bind('<Return>', self.perform_search)
        self.search_entry.focus_force()

        # 3. Label trạng thái
        self.status_label = tk.Label(self.root, text="", bg='#fff', fg='#8ab4f8', font=('Arial', 7))
        self.status_label.pack(side=tk.RIGHT, padx=2)

    def load_ai_and_data(self):
        """Load Model từ thư mục 'ai_model' bên cạnh file exe"""
        try:
            # --- ĐOẠN CODE QUAN TRỌNG NHẤT CẦN SỬA ---
            # Xác định đường dẫn file exe đang chạy
            if getattr(sys, 'frozen', False):
                application_path = os.path.dirname(sys.executable)
            else:
                application_path = os.path.dirname(os.path.abspath(__file__))

            # Tạo đường dẫn tuyệt đối tới folder 'ai_model'
            model_path = os.path.join(application_path, LOCAL_MODEL_FOLDER)
            db_path = os.path.join(application_path, DB_NAME)

            # Kiểm tra xem folder model có tồn tại không
            if not os.path.exists(model_path):
                self.update_status("Thiếu folder ai_model!")
                return

            # Load model từ đường dẫn offline
            self.model = SentenceTransformer(model_path)
            # ------------------------------------------

            # Kết nối DB (cũng dùng đường dẫn tuyệt đối cho chắc chắn)
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT question, answer FROM qa")
            rows = cursor.fetchall()
            conn.close()

            if not rows:
                self.update_status("Data trống!")
                return

            self.questions = [row[0] for row in rows]
            self.answers = [row[1] for row in rows]

            self.embeddings = self.model.encode(self.questions, convert_to_tensor=True)
            self.update_status("Sẵn sàng")

        except Exception as e:
            self.update_status("Lỗi Load")
            # In lỗi ra console để debug nếu cần
            print(f"Lỗi chi tiết: {str(e)}")

    def perform_search(self, event=None):
        user_query = self.search_entry.get().strip()
        if not user_query: return

        if self.embeddings is None:
            self.update_status("Chờ AI...")
            return

        self.update_status("...")  # Rút gọn text
        threading.Thread(target=self._search_logic, args=(user_query,), daemon=True).start()

    def _search_logic(self, user_query):
        query_embedding = self.model.encode(user_query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, self.embeddings)[0]
        top_results = torch.topk(cos_scores, k=min(3, len(self.questions)))

        best_idx = -1
        best_score = -1.0

        print(f"\n--- Tìm: {user_query} ---")

        for score, idx in zip(top_results.values, top_results.indices):
            idx = int(idx)
            current_score = float(score)

            # Logic Keyword Boosting cũ của bạn vẫn giữ nguyên
            question_in_db = self.questions[idx].lower()
            user_query_lower = user_query.lower()
            keywords = ["mô hình", "định nghĩa", "ví dụ", "đặc trưng", "quy trình", "dự án"]

            for kw in keywords:
                if kw in user_query_lower and kw in question_in_db:
                    current_score += 0.15

            if current_score > best_score:
                best_score = current_score
                best_idx = idx

        if best_score >= SIMILARITY_THRESHOLD:
            answer = self.answers[best_idx]
            pyperclip.copy(answer)
            self.update_status(f"OK ({int(best_score * 100)}%)")  # Text ngắn gọn
            print(f"==> CHỌN: {self.questions[best_idx]}")
            self.root.after(2000, lambda: self.update_status("Sẵn sàng"))
        else:
            self.update_status("Không thấy")
            print("==> KHÔNG TÌM THẤY")

    def update_status(self, text):
        self.status_label.config(text=text)

    def click_window(self, event):
        self._offsetx = event.x
        self._offsety = event.y

    def drag_window(self, event):
        x = self.root.winfo_pointerx() - self._offsetx
        y = self.root.winfo_pointery() - self._offsety
        self.root.geometry(f'+{x}+{y}')

    def toggle_visibility(self):
        if self.root.winfo_viewable():
            self.root.withdraw()
        else:
            self.root.deiconify()
            self.search_entry.delete(0, tk.END)
            self.search_entry.focus_force()


if __name__ == "__main__":
    root = tk.Tk()
    app = AIOverlayApp(root)
    root.mainloop()
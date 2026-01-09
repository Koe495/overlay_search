from sentence_transformers import SentenceTransformer
import os

# Tên model bạn đang dùng trong code chính
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'

# Tên thư mục muốn lưu
OUTPUT_FOLDER = 'ai_model'


def download_and_save_model():
    print(f"1. Đang tải model '{MODEL_NAME}' từ Internet...")
    print("   (Việc này có thể mất vài phút tùy mạng)...")

    # Tải model về RAM
    model = SentenceTransformer(MODEL_NAME)

    print(f"2. Đang lưu model vào thư mục '{OUTPUT_FOLDER}'...")

    # Lưu xuống ổ cứng thành folder
    model.save(OUTPUT_FOLDER)

    print("-" * 30)
    print("✅ XONG! Đã tạo thành công thư mục 'ai_model'.")
    print(f"Vị trí: {os.path.abspath(OUTPUT_FOLDER)}")
    print("Bây giờ bạn có thể đóng gói file exe được rồi.")


if __name__ == "__main__":
    download_and_save_model()
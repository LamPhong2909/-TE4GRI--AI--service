import json
import redis
import sys

# --- Cấu hình Redis ---
# Trong môi trường sản xuất, các thông tin này nên được đọc từ biến môi trường.
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_DB = 0


# r = redis.Redis(host='localhost', port=6379, db=0)
# r.set('foo', 'bar')
# True
# r.get('foo')
# b'bar'





# --- Các hằng số cho key trên Redis ---
KB_DISEASE_PREFIX = "kb:disease:"
KB_ALIAS_MAP_KEY = "kb:alias_map"

def connect_to_redis():
    """
    Thiết lập và trả về một kết nối đến Redis.
    Thử ping server để đảm bảo kết nối thành công.
    """
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
        r.ping()
        print("✅ Kết nối thành công đến Redis server.")
        return r
    except redis.exceptions.ConnectionError as e:
        print(f"❌ Lỗi kết nối đến Redis: {e}", file=sys.stderr)
        print("Vui lòng đảm bảo Redis server đang chạy trên host và port đã định cấu hình.", file=sys.stderr)
        sys.exit(1)

def validate_knowledge_base(kb_data):
    """
    Xác thực cấu trúc cơ bản của file knowledge base.
    Kiểm tra các trường bắt buộc cho mỗi bệnh.
    """
    print("🔍 Bắt đầu xác thực cấu trúc Knowledge Base...")
    required_keys = ["name", "aliases", "crop", "intents"]
    required_intent_keys = ["request_description", "request_symptoms", "request_treatment", "request_prevention"]
    
    if not isinstance(kb_data, dict):
        print("❌ Lỗi: Knowledge Base phải là một đối tượng JSON.", file=sys.stderr)
        return False

    for disease_id, data in kb_data.items():
        for key in required_keys:
            if key not in data:
                print(f"❌ Lỗi: Bệnh '{disease_id}' thiếu trường bắt buộc '{key}'.", file=sys.stderr)
                return False
        
        intents = data.get("intents", {})
        for intent_key in required_intent_keys:
            if intent_key not in intents:
                print(f"❌ Lỗi: Bệnh '{disease_id}' thiếu intent bắt buộc '{intent_key}'.", file=sys.stderr)
                return False
    
    print("👍 Cấu trúc Knowledge Base hợp lệ.")
    return True

def load_kb_to_redis(r_conn, kb_data):
    """
    Nạp dữ liệu từ Knowledge Base vào Redis.
    1. Xóa dữ liệu cũ để đảm bảo tính nhất quán.
    2. Lưu thông tin chi tiết của mỗi bệnh vào một key riêng.
    3. Xây dựng và lưu bản đồ alias (tên thay thế) để tra cứu nhanh.
    """
    print("\n🔄 Bắt đầu nạp dữ liệu vào Redis...")
    
    # Sử dụng pipeline để thực hiện nhiều lệnh một cách nguyên tử và hiệu quả
    pipe = r_conn.pipeline()
    
    # Bước 1: Xóa dữ liệu cũ
    old_keys = r_conn.keys(f"{KB_DISEASE_PREFIX}*")
    if old_keys:
        pipe.delete(*old_keys)
    pipe.delete(KB_ALIAS_MAP_KEY)
    
    # Bước 2 & 3: Nạp dữ liệu mới và xây dựng alias map
    alias_map = {}
    for disease_id, data in kb_data.items():
        # Lưu thông tin chi tiết của bệnh dưới dạng chuỗi JSON
        redis_key = f"{KB_DISEASE_PREFIX}{disease_id}"
        pipe.set(redis_key, json.dumps(data))
        
        # Thêm tên chính và các tên thay thế vào alias map
        # Chuyển về chữ thường để tìm kiếm không phân biệt hoa-thường
        main_name_lower = data['name'].lower()
        alias_map[main_name_lower] = disease_id
        
        for alias in data.get('aliases', []):
            alias_lower = alias.lower()
            if alias_lower in alias_map and alias_map[alias_lower] != disease_id:
                 print(f"⚠️ Cảnh báo: Alias '{alias}' bị trùng lặp giữa '{disease_id}' và '{alias_map[alias_lower]}'.")
            alias_map[alias_lower] = disease_id

    # Lưu alias map vào Redis Hash
    if alias_map:
        pipe.hset(KB_ALIAS_MAP_KEY, mapping=alias_map)
        
    # Thực thi pipeline
    results = pipe.execute()
    
    print(f"✅ Hoàn tất! Đã xử lý {len(kb_data)} bệnh.")
    print(f"✅ Đã tạo/cập nhật {len(alias_map)} alias trong map.")

def main(kb_path='knowledge_base.json'):
    """
    Hàm chính điều phối toàn bộ quá trình.
    """
    print("--- TE4GRI Knowledge Base Parser and Loader ---")
    
    # Đọc file JSON
    try:
        with open(kb_path, 'r', encoding='utf-8') as f:
            knowledge_base = json.load(f)
        print(f"📄 Đã đọc thành công file '{kb_path}'.")
    except FileNotFoundError:
        print(f"❌ Lỗi: Không tìm thấy file '{kb_path}'.", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Lỗi: File '{kb_path}' không phải là một file JSON hợp lệ. Chi tiết: {e}", file=sys.stderr)
        sys.exit(1)

    # Xác thực cấu trúc file
    if not validate_knowledge_base(knowledge_base):
        sys.exit(1)
        
    # Kết nối Redis
    redis_connection = connect_to_redis()
    
    # Nạp dữ liệu
    load_kb_to_redis(redis_connection, knowledge_base)
    
    print("\n🚀 Quá trình hoàn tất. Cơ sở tri thức đã sẵn sàng trên Redis(Giống với Cloudinary) cho TE4BOT sử dụng.")

if __name__ == "__main__":
    main()

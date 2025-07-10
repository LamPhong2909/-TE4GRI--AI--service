'''
Đây là source mẫu anh Phong tự viết, đúc kết từ các tài liệu kiến trúc, có thể không ăn nhằm và kết nối được tới TE4GRI&TE4BOT NHƯNG đáng để thử nghiệm
'''


import mmap
import json
import os
from fastapi import FastAPI, HTTPException, Depends # DÙng để gọi API
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any
from contextlib import contextmanager
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
import httpx
from redis import Redis
import joblib

# ***************************************
# 1. KIẾN TRÚC DỮ LIỆU NÂNG CAO
# ***************************************

class DiseaseEntity(BaseModel):
    id: str = Field(..., description="Unique disease ID")
    name: str = Field(..., min_length=3, max_length=100)
    aliases: List[str] = Field(default_factory=list, description="Alternative names")
    cv_class_id: int = Field(..., description="Mapped CV class ID")
    symptoms: Dict[str, str] = Field(..., description="Symptoms by growth stage")
    treatment: Dict[str, List[Dict]] = Field(..., description="Treatment options")
    prevention: List[str] = Field(..., description="Prevention methods")
    embedding: Optional[List[float]] = Field(None, description="AI embedding vector")

    @validator('treatment')
    def validate_treatment(cls, v):
        for key, treatments in v.items():
            for t in treatments:
                if 'name' not in t or 'dosage' not in t:
                    raise ValueError("Treatment must contain 'name' and 'dosage'")
        return v

class CVResult(BaseModel):
    class_id: int
    confidence: float = Field(..., ge=0, le=1)
    bounding_boxes: List[Dict] = Field(default_factory=list)
    detected_features: Dict[str, float] = Field(default_factory=dict)

class NLPContext(BaseModel):
    disease_id: str
    confidence: float
    user_message: Optional[str] = None
    previous_context: Dict[str, Any] = Field(default_factory=dict)
    cv_data: Dict[str, Any] = Field(default_factory=dict)

# ***************************************
# 2. MEMORY-MAPPED JSON MANAGER (CHO DỮ LIỆU LỚN)
# ***************************************

class MemoryMappedJSON:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self._file = None
        self._mmap = None
        self.index = self._build_index()
        
    def _build_index(self) -> Dict[str, tuple]:
        """Xây dựng index offset cho truy vấn O(1)"""
        index = {}
        position = 0
        
        with open(self.file_path, 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                if line.strip().startswith('"'):
                    key = line.split('"')[1]
                    start = position
                    end = position + len(line)
                    index[key] = (start, end)
                position += len(line)
        return index

    @contextmanager
    def _get_mmap(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"JSON file not found: {self.file_path}")
            
        with open(self.file_path, 'r+b') as f:
            mmap_obj = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            yield mmap_obj
            mmap_obj.close()

    def get(self, key: str) -> Optional[dict]:
        if key not in self.index:
            return None
            
        start, end = self.index[key]
        with self._get_mmap() as mmap_obj:
            chunk = mmap_obj[start:end].decode('utf-8')
            try:
                # Xử lý dòng JSON không đầy đủ
                if not chunk.strip().endswith('}'):
                    chunk += '}'
                return json.loads(chunk)
            except json.JSONDecodeError:
                # Fallback: tải toàn bộ file nếu cần
                with open(self.file_path) as f:
                    full_data = json.load(f)
                    return full_data.get(key)

    def get_all(self) -> Dict[str, dict]:
        results = {}
        with self._get_mmap() as mmap_obj:
            for key, (start, end) in self.index.items():
                chunk = mmap_obj[start:end].decode('utf-8')
                try:
                    results[key] = json.loads(chunk)
                except json.JSONDecodeError:
                    continue
        return results

# ***************************************
# 3. CV-NLP BRIDGE ENGINE (CORE LOGIC)
# ***************************************

class CVNLPBridge:
    def __init__(self, cv_metadata: MemoryMappedJSON, knowledge_base: MemoryMappedJSON):
        self.cv_metadata = cv_metadata
        self.kb = knowledge_base
        self.vectorizer = TfidfVectorizer()
        self._build_semantic_index()
        self.redis = Redis(host='cache.te4gri.internal', port=6379, db=0)

    def _build_semantic_index(self):
        """Xây dựng chỉ mục ngữ nghĩa để ánh xạ thông minh"""
        all_diseases = self.kb.get_all()
        documents = []
        self.disease_ids = []
        
        for disease_id, data in all_diseases.items():
            text = f"{data['name']} {' '.join(data['aliases'])} " \
                   f"{' '.join(data['symptoms'].values())}"
            documents.append(text)
            self.disease_ids.append(disease_id)
            
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)

    def _get_cache_key(self, cv_result: dict) -> str:
        """Tạo khóa cache duy nhất từ kết quả CV"""
        hash_input = f"{cv_result['class_id']}-{cv_result['confidence']}-" \
                     f"{'-'.join([str(b) for b in cv_result['bounding_boxes']])}"
        return hashlib.sha256(hash_input.encode()).hexdigest()

    def map_cv_to_nlp(self, cv_result: CVResult, user_message: str = "") -> NLPContext:
        # Kiểm tra cache trước
        cache_key = self._get_cache_key(cv_result.dict())
        if cached := self.redis.get(cache_key):
            return NLPContext(**json.loads(cached))

        # Bước 1: Ánh xạ cơ bản từ class ID
        disease_id = self.cv_metadata.get(str(cv_result.class_id))
        
        # Bước 2: Ánh xạ nâng cao dựa trên đặc trưng nếu cần
        if not disease_id or cv_result.confidence < 0.7:
            disease_id = self._semantic_mapping(cv_result)
            
        # Bước 3: Xây dựng ngữ cảnh NLP
        kb_entry = self.kb.get(disease_id)
        if not kb_entry:
            raise ValueError(f"Disease ID {disease_id} not found in knowledge base")
            
        context = NLPContext(
            disease_id=disease_id,
            confidence=cv_result.confidence,
            user_message=user_message,
            cv_data={
                "features": cv_result.detected_features,
                "bounding_boxes": cv_result.bounding_boxes
            }
        )
        
        # Lưu vào cache (TTL 30 phút)
        self.redis.setex(cache_key, 1800, json.dumps(context.dict()))
        return context

    def _semantic_mapping(self, cv_result: CVResult) -> str:
        """Ánh xạ ngữ nghĩa sử dụng embedding và các đặc trưng CV"""
        # Xử lý các đặc trưng CV thành văn bản mô tả
        feature_text = " ".join(
            f"{feat}:{value}" for feat, value in cv_result.detected_features.items()
        )
        
        # Vector hóa truy vấn
        query_vec = self.vectorizer.transform([feature_text])
        
        # Tính toán độ tương đồng cosine
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)
        best_match_idx = np.argmax(similarities)
        
        return self.disease_ids[best_match_idx]

    def update_knowledge(self, new_data: DiseaseEntity):
        """Cập nhật động knowledge base"""
        # 1. Cập nhật bộ nhớ
        self.kb.update(new_data.id, new_data.dict())
        
        # 2. Cập nhật chỉ mục ngữ nghĩa
        self._build_semantic_index()
        
        # 3. Xóa cache liên quan
        self._clear_related_cache(new_data.id)

    def _clear_related_cache(self, disease_id: str):
        """Xóa cache liên quan đến bệnh cụ thể"""
        pattern = f"*{disease_id}*"
        for key in self.redis.scan_iter(match=pattern):
            self.redis.delete(key)

# ***************************************
# 4. API GATEWAY TÍCH HỢP
# ***************************************

app = FastAPI(
    title="TE4GRI AI Gateway",
    version="1.0.0",
    description="Unified API Gateway for Computer Vision and NLP Integration"
)

# Khởi tạo các thành phần
cv_metadata = MemoryMappedJSON("/data/cv_metadata.json")
knowledge_base = MemoryMappedJSON("/data/knowledge_base.json")
cv_nlp_bridge = CVNLPBridge(cv_metadata, knowledge_base)

# ***************************************
# 5. ENDPOINTS API
# ***************************************

class CVRequest(BaseModel):
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    session_id: str

class ChatRequest(BaseModel):
    session_id: str
    message: str
    context: Dict[str, Any] = Field(default_factory=dict)

class UnifiedRequest(BaseModel):
    cv_data: Optional[CVRequest] = None
    chat_data: Optional[ChatRequest] = None

@app.post("/v1/ai/interact", response_model=Dict)
async def unified_ai_endpoint(request: UnifiedRequest):
    """Điểm cuối thống nhất cho tất cả tương tác AI"""
    response = {}
    
    # Xử lý yêu cầu Computer Vision
    if request.cv_data:
        cv_result = await process_cv(request.cv_data)
        nlp_context = cv_nlp_bridge.map_cv_to_nlp(cv_result)
        response['diagnosis'] = {
            "disease": nlp_context.disease_id,
            "confidence": nlp_context.confidence,
            "treatment": get_initial_treatment(nlp_context.disease_id)
        }
    
    # Xử lý yêu cầu Chatbot
    if request.chat_data:
        chat_context = request.chat_data.context
        if 'last_diagnosis' in chat_context:
            chat_context.update(cv_nlp_bridge.kb.get(chat_context['last_diagnosis']))
        
        bot_response = await process_chat(request.chat_data.message, chat_context)
        response['chat'] = bot_response
    
    return response

@app.post("/v1/cv/process", response_model=CVResult)
async def process_cv_endpoint(request: CVRequest):
    """Endpoint riêng cho xử lý Computer Vision"""
    cv_result = await process_cv(request)
    return cv_result

@app.post("/v1/chat/respond", response_model=Dict)
async def chat_endpoint(request: ChatRequest):
    """Endpoint riêng cho tương tác chatbot"""
    return await process_chat(request.message, request.context)

@app.post("/v1/knowledge/update")
async def update_knowledge(entity: DiseaseEntity):
    """Cập nhật động knowledge base"""
    cv_nlp_bridge.update_knowledge(entity)
    return {"status": "success", "updated_id": entity.id}

# ***************************************
# 6. HELPER FUNCTIONS
# ***************************************

async def process_cv(cv_request: CVRequest) -> CVResult:
    """Gọi dịch vụ CV và xử lý kết quả"""
    async with httpx.AsyncClient() as client:
        # Gọi microservice CV
        cv_service_url = "http://cv-service.te4gri/internal/v1/process"
        payload = {
            "session_id": cv_request.session_id,
            "image": cv_request.image_base64 or await fetch_image(cv_request.image_url)
        }
        response = await client.post(cv_service_url, json=payload, timeout=30.0)
        
        if response.status_code != 200:
            raise HTTPException(502, "CV service unavailable")
            
        return CVResult(**response.json())

async def process_chat(message: str, context: dict) -> dict:
    """Gọi dịch vụ NLP và xử lý hội thoại"""
    async with httpx.AsyncClient() as client:
        # Gọi microservice NLP
        nlp_service_url = "http://nlp-service.te4gri/internal/v1/chat"
        payload = {"message": message, "context": context}
        response = await client.post(nlp_service_url, json=payload, timeout=10.0)
        
        if response.status_code != 200:
            raise HTTPException(502, "NLP service unavailable")
            
        return response.json()

def get_initial_treatment(disease_id: str) -> dict:
    """Lấy thông tin điều trị ban đầu từ knowledge base"""
    disease_data = knowledge_base.get(disease_id)
    return {
        "emergency": disease_data['treatment'].get('emergency', []),
        "organic": disease_data['treatment'].get('organic', []),
        "prevention": disease_data['prevention']
    }

async def fetch_image(url: str) -> str:
    """Tải ảnh từ URL và mã hóa base64"""
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        if response.status_code != 200:
            raise HTTPException(400, "Invalid image URL")
        return base64.b64encode(response.content).decode('utf-8')

# ***************************************
# 7. DEPLOYMENT CONFIGURATION
# ***************************************

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        ssl_keyfile="/ssl/te4gri.key",
        ssl_certfile="/ssl/te4gri.crt",
        workers=4,
        timeout_keep_alive=300
    )
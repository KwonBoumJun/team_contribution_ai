from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from analyzer import score_all
import tempfile
import shutil
import numpy as np
# main.py
from analyzer import analyze_text

if __name__ == "__main__":
    sample_text = "여기에 테스트할 텍스트를 넣으세요."
    result = analyze_text(sample_text)
    print(result)

app = FastAPI()

# CORS 설정 (React 연동 시)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 배포 시 도메인 지정 필요
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze/")
async def analyze_files(files: list[UploadFile] = File(...)):
    docs = []
    for file in files:
        contents = await file.read()
        text = contents.decode('utf-8', errors='ignore')
        docs.append({"name": file.filename, "text": text})


    # 예시: 단어 수 기반 평균 길이 계산
    words = text.split()
    word_lengths = [len(w) for w in words]

    if len(word_lengths) == 0:
        avg_length = 0.0
    else:
        avg_length = float(np.mean(word_lengths))

    # NaN, inf 방지 처리
    if np.isnan(avg_length) or np.isinf(avg_length):
        avg_length = 0.0
        
    result = score_all(docs)
    return {"status": "success", "data": result}
#uvicorn main:app --reload
#http://127.0.0.1:8000/docs
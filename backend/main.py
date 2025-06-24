from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from analyzer import score_all, extract_text_from_pdf#, extract_text_from_image
from sqlmodel import SQLModel, Session, select
from models import AnalysisResult
import numpy as np
from datetime import datetime
from db import engine

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# DB 연결
SQLModel.metadata.create_all(engine)  # 서버 시작 시 테이블 생성


def save_result_to_db(result_dict):
    with Session(engine) as session:
        for filename, data in result_dict.items():
            record = AnalysisResult(
                filename=filename,
                word_count=data["word_count"],
                readability=data["readability"],
                similarity=data["similarity"],
                score=data["score"],
                uploaded_at=datetime.utcnow()
            )
            session.add(record)
        session.commit()


@app.post("/analyze/")
async def analyze_files(files: list[UploadFile] = File(...)):
    docs = []
    for file in files:
        contents = await file.read()
        filename = file.filename.lower()
        text = ""
        if filename.endswith(".pdf"):
            text = extract_text_from_pdf(contents)
        # elif filename.endswith((".jpg", ".jpeg", ".png")):
        #     text = extract_text_from_image(contents)
        else:
            try:
                text = contents.decode("utf-8", errors="ignore")
            except Exception:
                text = ""
        docs.append({"name": file.filename, "text": text})

    result = score_all(docs)
    save_result_to_db(result)  # 결과 저장
    return {"status": "success", "data": result}

@app.get("/results/")
def get_all_results():
    with Session(engine) as session:
        results = session.exec(select(AnalysisResult)).all()
        data = [r.dict() for r in results]
        return {"status": "success", "data": data}
#uvicorn main:app --reload
#http://127.0.0.1:8000/docs
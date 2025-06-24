from sqlmodel import Session
from models import AnalysisResult
from database import engine

def save_result_to_db(result_dict):
    with Session(engine) as session:
        for filename, data in result_dict.items():
            record = AnalysisResult(
                filename=filename,
                word_count=data["word_count"],
                readability=data["readability"],
                similarity=data["similarity"],
                score=data["score"],
            )
            session.add(record)
        session.commit()

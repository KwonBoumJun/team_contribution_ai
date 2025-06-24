from models import AnalysisResult
from database import engine

from sqlmodel import Session, select

with Session(engine) as session:
    results = session.exec(select(AnalysisResult)).all()
    for r in results:
        print(r)

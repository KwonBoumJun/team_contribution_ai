import requests  # 추가
# language_tool_python 관련 import, tool 선언은 삭제
import textstat
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def analyze_text(text):
    word_count = len(text.split())
    sentence_count = text.count('.') + text.count('!') + text.count('?')
    readability = textstat.flesch_reading_ease(text)

    # NaN, inf 체크 및 처리
    if readability is None or np.isnan(readability) or np.isinf(readability):
        readability = 0.0
    
    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "readability": readability,
    }

# 2. 유사도 계산 (모든 문서 pairwise cosine similarity)
def compute_similarity(texts):
    tfidf = TfidfVectorizer().fit_transform(texts)
    sim_matrix = cosine_similarity(tfidf)
    avg_similarities = []
    for i in range(len(texts)):
        sim_scores = np.delete(sim_matrix[i], i)
        if len(sim_scores) == 0:
            avg_sim = 0.0
        else:
            avg_sim = np.mean(sim_scores)
            if np.isnan(avg_sim) or np.isinf(avg_sim):
                avg_sim = 0.0
        avg_similarities.append(avg_sim)
    return avg_similarities


def score_all(docs):
    texts = [d['text'] for d in docs]
    names = [d['name'] for d in docs]
    
    similarities = compute_similarity(texts)
    
    results = {}
    for i, doc in enumerate(docs):
        info = analyze_text(doc['text'])
        score = (
            info["word_count"] * 0.2 +
            info["readability"] * 0.3 +
            (1 - similarities[i]) * 100 * 0.2
        )
        # score NaN, inf 처리
        if np.isnan(score) or np.isinf(score):
            score = 0.0
            
        results[names[i]] = {
            "word_count": info["word_count"],
            "readability": info["readability"],
            "similarity": similarities[i],
            "score": round(score, 2)
        }
    return results

if __name__ == "__main__":
    team_docs = [
        {"name": "홍길동", "text": open("hong.txt", encoding="utf-8").read()},
        {"name": "김철수", "text": open("kim.txt", encoding="utf-8").read()},
        {"name": "이영희", "text": open("lee.txt", encoding="utf-8").read()}
    ]
    results = score_all(team_docs)
    for name, data in results.items():
        print(f"{name} → 점수: {data['score']}점 | 유사도: {data['similarity']:.2f}")

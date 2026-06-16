from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    age: int
@app.get("/post")
def test():
    return {"message": "Now browser works"}

@app.post("/post")
def create_item(item: Item):
    return {
        "message": "Data received",
        "data": item
    }





 


from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
from datetime import datetime
import json 
app=FastAPI()
conn = sqlite3.connect("api_logs.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_json TEXT,
    response_json TEXT,
    status_code INTEGER,
    created_at TEXT
)
""")
conn.commit()
model=SentenceTransformer("all-mpnet-base-v2")
audience_profiles = {
    "Jaipur Residents": "jaipur local news traffic weather civic issues city events rajasthan pink city municipal urban",
    "Software Engineers": "software development programming coding cloud computing devops backend frontend api open source engineering",
    "Cricket Fans": "cricket ipl test match odi bcci t20 player stats tournament world cup sports news",
    "Football Fans": "football soccer premier league fifa transfers clubs champions league la liga match analysis bundesliga",
    "Tech Enthusiasts": "gadgets smartphones artificial intelligence laptops consumer electronics wearables reviews product launch tech news",
    "Business Professionals": "startups entrepreneurship finance economy business strategy b2b leadership venture capital funding",
    "College Students": "education internships placements scholarships career guidance campus entrance exams upskilling resume",
    "Healthcare Professionals": "healthcare medicine hospitals treatments public health clinical pharma medical research patient care diagnosis",
    "Travel Lovers": "tourism destinations travel guides hotels adventure backpacking flights itinerary visa solo travel",
    "Environmentalists": "sustainability climate change renewable energy conservation pollution green energy carbon footprint ecology net zero",
    "Movie Lovers": "films cinema actors reviews entertainment bollywood hollywood ott streaming box office",
    "Fitness Enthusiasts": "exercise nutrition gym workouts wellness healthy lifestyle yoga weight loss muscle building diet",
    "Investors": "stock market investments mutual funds financial planning economy nifty sensex portfolio returns trading",
    "Government Job Aspirants": "government exams recruitment upsc ssc railway jobs public sector notifications admit card syllabus results",
    "AI Researchers": "artificial intelligence machine learning deep learning nlp computer vision llm neural networks research papers datasets",
    "Weather Forecast Seekers": "weather forecast temperature rainfall air quality humidity alerts monsoon climate conditions imd real time weather",
    "Real Estate Buyers": "property housing market home buying real estate flat apartment plot rera mortgage home loan",
    "Automobile Enthusiasts": "cars bikes vehicle reviews electric vehicles automotive ev auto expo test drive mileage specifications",
    "Legal Professionals": "law legal news court cases policies judiciary constitution litigation legal consulting high court supreme court",
    "Wedding Planners": "wedding planning outfits venues photography ceremonies catering decoration bridal shaadi event management"
}
audience_names = list(audience_profiles.keys())
audience_texts = list(audience_profiles.values())
audience_embeddings=model.encode(audience_texts)
class InputData(BaseModel):
    headline: str
    content: str
@app.post(
    "/get-score",
    summary="Get Audience Score",
    description="Analyzes headline and content to return top 5 most relevant audience categories with confidence and relevance scores."
)
def get_score(data:InputData):
    text = data.headline + " " + data.content
    text_embedding = model.encode([text])
    similarities = cosine_similarity(text_embedding, audience_embeddings)[0]
    results=[]
    for i, score in enumerate(similarities):
        results.append({
            "category": audience_names[i],
            "confidence_score": round(float(score), 3),
            "relevance_score": round(float(score) * 100, 1)
            })
    top_5 = sorted(results, key=lambda x: x["confidence_score"], reverse=True)[:5]
    request_json = json.dumps(data.model_dump())
    response_json = json.dumps(top_5)
    status_code = 200
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO logs (request_json, response_json, status_code, created_at)
    VALUES (?, ?, ?, ?)""", (request_json, response_json, status_code, created_at))
    conn.commit()
    return top_5
    







  













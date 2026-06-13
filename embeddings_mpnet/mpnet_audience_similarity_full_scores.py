import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
#df = pd.read_csv("Batch_300_Articles.csv",encoding="unicode_escape")
df = pd.read_csv("Batch1_Jaipur_Residents_20_Articles.csv",encoding="unicode_escape")
model = SentenceTransformer('all-mpnet-base-v2')
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
audience_embeddings = model.encode(audience_texts)
articles = df["article"].tolist()
article_embeddings = model.encode(articles)
best_audiences = []
best_scores = []
all_scores_list = []
for article_emb in article_embeddings:
    similarities = cosine_similarity([article_emb],audience_embeddings)[0]
    score_dict = {audience_names[i]: round(similarities[i], 3)
        for i in range(len(audience_names))
    }
    all_scores_list.append(score_dict)
    best_index = similarities.argmax()
    best_audiences.append(audience_names[best_index])
    best_scores.append(round(similarities[best_index], 3))
df["best_audience"] = best_audiences
df["confidence_score"] = best_scores
df["relevance_score"] = [round(s * 100, 2) for s in best_scores]
scores_df = pd.DataFrame(all_scores_list)
df = pd.concat([df, scores_df], axis=1)
df.to_csv("full_score_embedding_output.csv", index=False)


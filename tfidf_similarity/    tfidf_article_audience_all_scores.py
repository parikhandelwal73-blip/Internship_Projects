import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#df = pd.read_csv("Batch_300_Articles.csv", encoding="unicode_escape")
df = pd.read_csv("Batch1_Jaipur_Residents_20_Articles.csv", encoding="unicode_escape")

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
vectorizer = TfidfVectorizer()
all_texts = list(df["article"]) + list(audience_profiles.values())
vectorizer.fit(all_texts)
all_results = []
for _, row in df.iterrows():
    article = row["article"]
    for audience, audience_desc in audience_profiles.items():
        vectors = vectorizer.transform([article, audience_desc])
        similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
        all_results.append({
            "article": article,
            "audience": audience,
            "relevance_score": round(similarity * 100, 2),
            "confidence_score": round(similarity, 3)
        })
result_df = pd.DataFrame(all_results)
result_df.to_csv("all_audience_scores.csv", index=False)


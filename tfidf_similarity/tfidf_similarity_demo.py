from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
article = """
"Healthcare Infrastructure for a Growing Population is becoming an important discussion point for Jaipur residents as the city continues to grow and modernize. Urban development brings new opportunities while also creating challenges that require thoughtful planning and community participation. In recent years, Jaipur has seen increasing investment in infrastructure, digital services, transportation, tourism, and environmental initiatives. These changes are influencing how people commute, access services, conduct business, and experience everyday life.
Residents are increasingly interested in solutions that improve convenience without compromising the city's cultural identity. Whether the focus is transportation, sustainability, education, or economic growth, long-term success depends on balancing modernization with heritage preservation. Local communities, businesses, educational institutions, and government agencies all play a role in shaping outcomes. Collaboration often leads to more practical and sustainable solutions than isolated efforts.
The topic also highlights the importance of public awareness and citizen engagement. When residents understand ongoing developments and participate in discussions, projects are more likely to reflect local needs and priorities. Access to information, transparent decision-making, and community feedback can strengthen trust and improve results. As Jaipur evolves into a larger and more connected urban center, informed citizens will continue to play a key role in guiding its future. By staying engaged and supporting responsible development, residents can help ensure that growth creates benefits for current and future generations."
"""
audience = """
People interested in wedding planning outfits venues photography and ceremonies
"""
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform([article, audience])
similarity = cosine_similarity(vectors[0], vectors[1])
print(similarity[0][0])
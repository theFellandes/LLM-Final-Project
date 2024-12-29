"""
Neo4j + Dual Sentiment (ChatOpenAI & BART) + Star Ratings
Fixed so that the LangChain chain matches the prompt variables.

We use:
 - openai_sentiment_and_stars(review_text) calls openai_chain.invoke({"review_text": review_text})
 - The prompt references {review_text} (no extra placeholders like {sentiment}).
"""

import os
import re
import json
import logging

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 1. Connect to Neo4j
# -----------------------------------------------------------------------------
from neo4j import GraphDatabase

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "your_password")

logger.info(f"Connecting to Neo4j at {NEO4J_URI} with user '{NEO4J_USER}'")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# -----------------------------------------------------------------------------
# 2. OpenAI-based Sentiment + Star Rating (ChatOpenAI)
# -----------------------------------------------------------------------------
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain.chains import LLMChain

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not set; OpenAI-based sentiment will return 'N/A'.")

# A minimal system template that references {review_text}
system_template = """You are a sentiment analysis assistant.
Classify the following review as POSITIVE, NEGATIVE, or NEUTRAL,
and also provide a star rating from 0 to 5 (integer only).

Output your result in JSON ONLY, for example:
{{
  "sentiment": "<POSITIVE|NEGATIVE|NEUTRAL>",
  "stars": <0-5 integer>
}}

Review: {review_text}
"""

system_msg = SystemMessagePromptTemplate.from_template(system_template)
chat_prompt = ChatPromptTemplate.from_messages([system_msg])

chat_openai_llm = None
openai_chain = None

if OPENAI_API_KEY:
    chat_openai_llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name="gpt-3.5-turbo",
        temperature=0.0
    )
    openai_chain = LLMChain(llm=chat_openai_llm, prompt=chat_prompt)

def openai_sentiment_and_stars(review_text: str):
    """
    Calls openai_chain.invoke(...) with {"review_text": ...} to match the template.
    Returns {"sentiment": "<POS|NEG|NEUTRAL>", "stars": int} if successful,
    or a fallback if no API key or invalid JSON is returned.
    """
    if openai_chain is None:
        return {"sentiment": "N/A", "stars": 0}

    result = openai_chain.invoke({"review_text": review_text})
    raw_text = result["text"].strip()

    # Attempt to parse JSON
    try:
        data = json.loads(raw_text)
        sentiment = data.get("sentiment", "N/A").upper()
        stars = int(data.get("stars", 0))
        if sentiment not in ["POSITIVE", "NEGATIVE", "NEUTRAL"]:
            sentiment = "N/A"
        if stars < 0 or stars > 5:
            stars = 0
        return {"sentiment": sentiment, "stars": stars}
    except Exception:
        return {"sentiment": "N/A", "stars": 0}

# -----------------------------------------------------------------------------
# 3. Local BART Zero-Shot Classifier
# -----------------------------------------------------------------------------
from transformers import pipeline

logger.info("Loading local BART model (facebook/bart-large-mnli).")
bart_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# For sentiment
bart_sentiment_labels = ["POSITIVE", "NEGATIVE", "NEUTRAL"]
# For star rating
bart_star_labels = ["0 stars", "1 star", "2 stars", "3 stars", "4 stars", "5 stars"]

def bart_sentiment(review_text: str) -> str:
    """
    Classify as POSITIVE, NEGATIVE, or NEUTRAL using BART zero-shot.
    """
    result = bart_classifier(review_text, bart_sentiment_labels)
    best_label = max(zip(result["labels"], result["scores"]), key=lambda x: x[1])[0]
    return best_label

def bart_star_rating(review_text: str) -> int:
    """
    Classify the review into 0..5 stars using BART zero-shot.
    We'll parse out the digit from the top label (e.g. "3 stars" -> 3).
    """
    result = bart_classifier(review_text, bart_star_labels)
    best_label = max(zip(result["labels"], result["scores"]), key=lambda x: x[1])[0]
    digits = re.findall(r"\d", best_label)
    if digits:
        rating = int(digits[0])
        if 0 <= rating <= 5:
            return rating
    return 0

def bart_sentiment_and_stars(review_text: str):
    return {
        "sentiment": bart_sentiment(review_text),
        "stars": bart_star_rating(review_text),
    }

# -----------------------------------------------------------------------------
# 4. Combined Function: Return BOTH "bots" side-by-side
# -----------------------------------------------------------------------------
def classify_sentiment_with_all(review_text: str):
    """
    Returns dict, e.g.:
    {
      "OpenAI": {"sentiment": "POSITIVE", "stars": 5},
      "BART":   {"sentiment": "NEUTRAL", "stars": 3}
    }
    """
    return {
        "OpenAI": openai_sentiment_and_stars(review_text),
        "BART": bart_sentiment_and_stars(review_text),
    }

# -----------------------------------------------------------------------------
# 5. Neo4j Query Functions
# -----------------------------------------------------------------------------
def get_reviews_for_book(book_name: str):
    query = """
    MATCH (u:User)-[r:REVIEWED_BY]->(b:Book)
    WHERE toLower(b.name) CONTAINS toLower($name)
    RETURN elementId(u) AS userId, r.review AS review
    """
    logger.info(f"Running query:\n{query}\nPARAMS: {{ name: '{book_name}' }}")
    with driver.session() as session:
        results = session.run(query, name=book_name)
        return list(results)

def get_book_sentiment(book_name: str):
    reviews = get_reviews_for_book(book_name)
    logger.info(f"Found {len(reviews)} reviews for '{book_name}'.")

    output = []
    for record in reviews:
        user_id = record["userId"]
        review_text = record["review"]
        sentiments = classify_sentiment_with_all(review_text)
        output.append({
            "userId": user_id,
            "review": review_text,
            "sentiments": sentiments
        })
    return output

# -----------------------------------------------------------------------------
# 6. Simple Chatbot Interface
# -----------------------------------------------------------------------------
def extract_book_name(user_input: str) -> str:
    match = re.search(r"'([^']+)'|\"([^\"]+)\"", user_input)
    if match:
        return match.group(1) or match.group(2)
    return None

def chatbot_interface():
    print("Welcome to the Dual Sentiment + Star Rating Chatbot!")
    print("Ask me: What's the sentiment for \"Gatsby\"?")
    print("Or type 'exit'/'quit' to end.")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        if "sentiment" in user_input.lower() or "score" in user_input.lower():
            book_name = extract_book_name(user_input)
            if not book_name:
                print("Chatbot: I couldn't detect the book name. Please use quotes.")
                continue

            results = get_book_sentiment(book_name)
            if not results:
                print(f"Chatbot: No reviews found for '{book_name}' (partial match).")
                continue

            response = f"Reviews for any book name containing '{book_name}':\n"
            for r in results:
                user_id = r["userId"]
                review_text = r["review"]
                s_openai = r["sentiments"]["OpenAI"]
                s_bart   = r["sentiments"]["BART"]

                response += (
                    f"  - User ID {user_id}:\n"
                    f"    Review: \"{review_text}\"\n"
                    f"    [OpenAI] => {s_openai['sentiment']} ({s_openai['stars']} stars)\n"
                    f"    [BART]   => {s_bart['sentiment']} ({s_bart['stars']} stars)\n"
                )
            print(f"Chatbot:\n{response}")
        else:
            print(
                "Chatbot: I can show you sentiment from both OpenAI & BART plus a star rating (0-5). "
                'Try: "What\'s the sentiment for \"Romeo\"?"'
            )

# -----------------------------------------------------------------------------
# 7. MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    chatbot_interface()

import os
import re
import logging
from neo4j import GraphDatabase
from transformers import pipeline

# Set up logging for your own messages.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Neo4j connection
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "your_password")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Local BART pipeline
bart_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
bart_candidate_labels = ["POSITIVE", "NEGATIVE", "NEUTRAL"]

def bart_sentiment(review_text: str) -> str:
    result = bart_classifier(review_text, bart_candidate_labels)
    best_label = max(zip(result["labels"], result["scores"]), key=lambda x: x[1])[0]
    return best_label

def get_reviews_for_book(book_name: str):
    """
    Retrieve (userName, review) from the graph for a partial match on b.name.
    We'll log out the query and param so you can see what's being sent.
    """
    query = """
    MATCH (u:User)-[r:REVIEWED_BY]->(b:Book)
    WHERE toLower(b.name) CONTAINS toLower($name)
    RETURN u.name AS userName, r.review AS review
    """

    # LOG the query + parameters:
    logger.info(f"Running query:\n{query}\nPARAMS: {{ name: '{book_name}' }}")

    with driver.session() as session:
        results = session.run(query, name=book_name)
        return list(results)

def get_book_sentiment(book_name: str):
    reviews = get_reviews_for_book(book_name)
    if not reviews:
        return []

    output = []
    for record in reviews:
        user_name = record["userName"]
        review_text = record["review"]
        sentiment = bart_sentiment(review_text)
        output.append({
            "userName": user_name,
            "review": review_text,
            "sentiment": sentiment
        })
    return output

def extract_book_name(user_input: str) -> str:
    match = re.search(r"'([^']+)'|\"([^\"]+)\"", user_input)
    if match:
        return match.group(1) or match.group(2)
    return None

def chatbot_interface():
    print("Welcome to the partial-match chatbot!")
    print("Ask me: What's the sentiment for \"Gatsby\"? (or 'exit'/'quit' to end)")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        if "sentiment" in user_input.lower() or "score" in user_input.lower():
            book_name = extract_book_name(user_input)
            if not book_name:
                print("Chatbot: I couldn't detect the book name. Please enclose it in quotes.")
                continue

            results = get_book_sentiment(book_name)
            if not results:
                print(f"Chatbot: No reviews found for '{book_name}'.")
                continue

            response = f"Reviews for any book name containing '{book_name}':\n"
            for r in results:
                response += (
                    f"  - User: {r['userName']}\n"
                    f"    Review: \"{r['review']}\"\n"
                    f"    BART Sentiment => {r['sentiment']}\n"
                )
            print(f"Chatbot:\n{response}")
        else:
            print("Chatbot: Try something like: What's the sentiment for \"Romeo\"?")

if __name__ == "__main__":
    chatbot_interface()

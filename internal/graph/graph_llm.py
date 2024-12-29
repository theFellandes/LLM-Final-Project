import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage


class GraphLLM:
    """
    Handles interaction with OpenAI's API to generate graph structure from book data.
    """

    def __init__(self, model="gpt-4", temperature=0.7, max_tokens=1000):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate_graph_structure(self, book, user_rating=None):
        """
        Generates nodes, edges, and attributes using OpenAI's GPT-4 chat model.
        """
        # Initialize the ChatOpenAI model
        chat = ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # Define the system message and prompt
        system_message = SystemMessage(
            content="You are a highly intelligent assistant that generates graph database structures."
        )
        prompt = ChatPromptTemplate.from_messages(
            [system_message, HumanMessagePromptTemplate.from_template(
                "Given the following book and user rating data:\n\n"
                "Book:\n{book}\n\n"
                "User Rating:\n{user_rating}\n\n"
                "1. Create nodes for the book, author(s), genre(s), and user.\n"
                "2. Define relationships between these nodes, including WRITTEN_BY, BELONGS_TO, REVIEWED_BY, and SIMILAR_TO.\n"
                "3. Suggest additional attributes for nodes and relationships.\n\n"
                "Provide the response as JSON in the following format:\n"
                "{{\n"
                "  'nodes': [{{'label': 'Book', 'properties': {{'id': '...', 'title': '...', ...}}}}, ...],\n"
                "  'edges': [{{'from': 'Book', 'to': 'Author', 'type': 'WRITTEN_BY', 'properties': {{}}}}, ...]\n"
                "}}"
            )]
        )

        # Format the input
        inputs = {"book": book, "user_rating": user_rating or {}}

        # Generate the response
        try:
            response = chat.invoke(prompt.format_prompt(**inputs).to_messages())
            response_content = response.content.strip()

            # Extract and parse JSON from the response
            json_start = response_content.find("{")
            json_end = response_content.rfind("}") + 1
            if json_start == -1 or json_end == -1:
                raise ValueError("No valid JSON found in the response.")

            json_data = json.loads(response_content[json_start:json_end])
            return json_data

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from response: {e}")
            print(f"Response content: {response_content}")
            raise

        except Exception as e:
            print(f"An error occurred: {e}")
            raise


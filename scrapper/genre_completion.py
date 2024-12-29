from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage


class GenreCompletion:
    def __init__(self, model="gpt-4", temperature=0.7, max_tokens=500):
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def complete_genre(self, book_title, book_description):
        """
        Uses OpenAI to infer the genre based on the book's title and description.
        """
        system_message = SystemMessage(
            content="You are a literary expert that identifies book genres based on their title and description."
        )
        prompt = ChatPromptTemplate.from_messages(
            [system_message, HumanMessagePromptTemplate.from_template(
                "Given the following book details:\n\n"
                "Title: {title}\n"
                "Description: {description}\n\n"
                "What genre does this book most likely belong to?"
            )]
        )
        inputs = {"title": book_title, "description": book_description or "No description available"}
        response = self.llm.invoke(prompt.format_prompt(**inputs).to_messages())
        return response.content.strip()

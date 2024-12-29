from pydantic import BaseModel, Field
from langchain.chat_models import ChatOpenAI
from langchain.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage


class GenreFetcher(BaseModel):
    """
    A Pydantic class for fetching or inferring genres for books using Wikipedia and LLM.
    """

    model: str = Field(default="gpt-4", description="Model to use for genre inference")
    temperature: float = Field(default=0.7, description="Sampling temperature for the model")
    max_tokens: int = Field(default=500, description="Maximum token limit for the model")
    llm: ChatOpenAI = Field(default_factory=lambda: None, description="LLM instance for genre inference")
    wiki_tool: WikipediaQueryRun = Field(default_factory=lambda: None, description="Wikipedia tool for fetching content")

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        # Initialize ChatOpenAI and Wikipedia tool
        if self.llm is None:
            object.__setattr__(self, 'llm', ChatOpenAI(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            ))
        if self.wiki_tool is None:
            api_wrapper = WikipediaAPIWrapper()  # Initialize the API wrapper
            object.__setattr__(self, 'wiki_tool', WikipediaQueryRun(api_wrapper=api_wrapper))

    def fetch_genre_from_wikipedia(self, book_title: str) -> str | None:
        """
        Fetches the genre of the book using WikipediaQueryRun.
        """
        try:
            result = self.wiki_tool.run(f"{book_title} genre")
            if "genre" in result.lower():
                return self._extract_genre_from_text(result)
        except Exception as e:
            print(f"Error fetching genre from Wikipedia for '{book_title}': {e}")
        return None

    @staticmethod
    def _extract_genre_from_text(text: str) -> str | None:
        """
        Extracts the genre from text if it exists.
        """
        for sentence in text.split('.'):
            if "genre" in sentence.lower():
                return sentence.split("genre")[-1].strip(":").split(',')[0].strip()
        return None

    def infer_genre_with_llm(self, book_title: str, book_description: str) -> str:
        """
        Uses LLM to infer the genre based on the book's title and description.
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
        inputs = {"title": book_title, "description": book_description}
        response = self.llm(prompt.format_prompt(**inputs).to_messages())
        return response.content.strip()

    def fetch_or_infer_genre(self, book_title: str, book_description: str) -> str:
        """
        Attempts to fetch the genre from Wikipedia first, then uses LLM as a fallback.
        """
        genre = self.fetch_genre_from_wikipedia(book_title)
        if genre:
            return genre
        return self.infer_genre_with_llm(book_title, book_description)

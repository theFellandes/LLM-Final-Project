from pydantic import BaseModel, Field
from typing import List, Union
from langchain.docstore.document import Document
from langchain.text_splitter import TokenTextSplitter
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.document_loaders import WikipediaLoader


class WikipediaDocumentLoader(BaseModel):
    """
    A Pydantic class for loading and splitting content from Wikipedia pages.
    """

    page_title: str = Field(..., description="Title of the Wikipedia page to load")
    chunk_size: int = Field(default=512, description="Size of text chunks after splitting")
    chunk_overlap: int = Field(default=128, description="Overlap size between text chunks")

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        self.wiki_parser = WikipediaAPIWrapper()
        self.text_splitter = TokenTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

    def load_page(self) -> List[Document]:
        """
        Loads the Wikipedia page using WikipediaLoader.
        """
        return WikipediaLoader(query=self.page_title).load()

    def load_full_content(self) -> List[Document]:
        """
        Loads the full content of the Wikipedia page using WikipediaAPIWrapper.
        """
        content = self.wiki_parser.run(self.page_title)
        return [Document(page_content=content, metadata={"source": self.page_title})]

    def split_document(self, raw_documents: Union[List[Document], Document]) -> List[Document]:
        """
        Splits the loaded Wikipedia page content into smaller chunks.
        """
        if isinstance(raw_documents, Document):
            raw_documents = [raw_documents]

        chunks = []
        for doc in raw_documents:
            print(f"Processing document: {doc.metadata.get('source', 'Unknown')}")  # Debug info
            doc_chunks = self.text_splitter.split_documents([doc])
            print(f"Number of chunks created: {len(doc_chunks)}")  # Debug info
            chunks.extend(doc_chunks)

        return chunks

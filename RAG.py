from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

class RAG:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")

        self.vector_store = Chroma(
            embedding_function=self.embeddings,
            collection_name="my_knowledge_base",
            persist_directory="./chroma_db"
        )
        self.llm = ChatOllama(model="llama3.2")

        self._setup_knowledge_base()
        self.chain = self._setup_rag_chain()

    def _setup_knowledge_base(self):
        if self.vector_store._collection.count() == 0:
            raw_text = "The Hochschule der Bildenden Kunste Saar (HBKsaar) was formally established as an independent institution in 1989 through legislation enacted by the state of Saarland. Its foundation marked the continuation and restructuring of earlier art and design education in the region, particularly from the postwar Schule für Kunst und Handwerk, which had operated within the former Fachhochschule des Saarlandes since the 1970s.)"
            docs = [Document(page_content=x) for x in raw_text.strip().split("\n")]

            self.vector_store.add_documents(
                documents=docs,
                ids=["id"+ str(i) for i in range(1,len(docs)+1) ]
            )

    def _setup_rag_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a capable assistant, answering user questions based on the following background information, answer should be brief within 10 words. You MUST always answer in German only. Background: {context}"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "Answer the following questions: {question}")
        ])

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        return (
                {"context": (lambda x: x["question"]) | self.vector_store.as_retriever() | format_docs,
                 "question": lambda x: x["question"],
                 "history": lambda x: x["chat_history"]
                }
                | prompt
                | self.llm
                | StrOutputParser()

        )

    def ask_stream(self, question, history_messages):
        langchain_messages = []
        for msg in history_messages:
            if msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_messages.append(AIMessage(content=msg["content"]))

        for chunk in self.chain.stream({
            "question": str(question),
            "chat_history": langchain_messages
        }):
            yield chunk


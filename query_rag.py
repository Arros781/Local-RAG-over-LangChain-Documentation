import os
os.environ["NO_PROXY"] = "127.0.0.1,localhost"

from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama


# 1. Те же эмбеддинги, что при индексации
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://127.0.0.1:11434",
)

# 2. Открываем ту же коллекцию
vector_store = Chroma(
    collection_name="promt_engineering",       
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

# 3. Промпт
prompt = ChatPromptTemplate.from_messages([
    ("system", """Ты — ассистент, который отвечает на вопросы, используя ТОЛЬКО предоставленный контекст.

ПРАВИЛА:
1. Отвечай строго на основе контекста. Не добавляй информацию от себя.
2. Если в контексте нет ответа — скажи: "В предоставленном контексте нет информации для ответа на этот вопрос."
3. В конце ответа всегда указывай источник: URL из метаданных.
4. Отвечай на русском языке, если вопрос задан на русском.

КОНТЕКСТ:
{context}
"""),
    ("human", "{question}"),
])

# 4. LLM
llm = ChatOllama(
    model="llama3.2",
    base_url="http://127.0.0.1:11434",
    temperature=0,
)

# 5. Chain
chain = prompt | llm


def ask(question: str):
    print(f"\n>>> {question}")
    
    # Поиск
    retrieved_docs = vector_store.similarity_search(question, k=3)
    
    if not retrieved_docs:
        print("Ничего не найдено в базе.")
        return
    
    # Формируем контекст с источниками
    context = "\n\n---\n\n".join([
        f"[Источник: {doc.metadata.get('source', 'N/A')}]\n{doc.page_content}"
        for doc in retrieved_docs
    ])
    
    # Генерация
    answer = chain.invoke({"question": question, "context": context})
    print(answer.content)


if __name__ == "__main__":
    questions = [
        "Что такое RecursiveCharacterTextSplitter?",
        "Какой размер чанка по умолчанию?",
        "Как установить библиотеку?",
    ]
    for q in questions:
        ask(q)
        print("-" * 80)
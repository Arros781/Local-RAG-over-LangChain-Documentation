# Local RAG over LangChain Documentation

RAG-система для ответов на вопросы по технической документации. 
Работает полностью локально: без API-ключей, без облачных сервисов, без интернета 
(после индексации).

Стек: LangChain, Ollama, ChromaDB, WebBaseLoader.

---

## Установка зависимостей



pip install langchain-community langchain-text-splitters langchain-chroma \
            langchain-ollama chromadb beautifulsoup4 lxml requests \
            tiktoken python-dotenv


## Модели Ollama
ollama pull llama3.2
ollama pull nomic-embed-text

## Структура проекта

```bash
local-rag/
├── README.md
├── requirements.txt
├── rag_data.py       # индексация: загрузка → чанкинг → эмбеддинги → ChromaDB
├── query_rag.py      # запросы: поиск → промпт → LLM → ответ
├── test_rag.py       # 14 тестовых вопросов
└── chroma_db/        # persistent vector store

## Запуск

```bash
1. Индексация - запускается 1 раз (python rag_data.py) в следствии загружает страницу документации, разбивает на чанки, считает эмбеддинги
и сохраняет в ./chroma_db.
2.Запросы - (python query_rag.py)
<img width="901" height="288" alt="Снимок экрана 2026-10-04 164421" src="https://github.com/user-attachments/assets/93bda2d7-f0ae-4e0f-9e2c-8b939b8c22c7" />

3.Проверка качества - задаем 14 вопросов. Все вопросы и ответы можете найти в коммите. Тест специально структурирован на типы: прямые, факты, синтез, все контекста, провокации. Что способствует более точной проверки.

#Вся архитектура

```bash
WebPage → WebBaseLoader → Documents
                              ↓
                 RecursiveCharacterTextSplitter
                              ↓
                            Chunks
                              ↓
                     OllamaEmbeddings
                              ↓
                          ChromaDB
                              ↓
        User Query → similarity_search → Top-K Chunks
                              ↓
                    ChatPromptTemplate
                              ↓
                         ChatOllama
                              ↓
                        Answer + Source


##Технические решения
OllamaEmbeddings вместо OpenAIEmbeddings
Ollama предоставляет OpenAI-совместимый API (/v1/), но batch-запросы
на эмбеддинги через него возвращают 503. OllamaEmbeddings использует
нативный эндпоинт /api/embed и работает стабильно.

127.0.0.1 вместо localhost
На Windows localhost может резолвиться в IPv6 (::1), тогда как Ollama
слушает IPv4. Явный 127.0.0.1 устраняет проблему.

NO_PROXY
При включённом VPN или системном прокси запросы к локальной Ollama
уходят через прокси и падают с 503. NO_PROXY=127.0.0.1,localhost
направляет их напрямую.

USER_AGENT
WebBaseLoader без User-Agent определяется сайтами как бот и блокируется.
Реалистичный User-Agent Chrome решает проблему.

Anti-hallucination промпт
Промпт явно ограничивает модель контекстом: если ответа нет — вернуть
"В предоставленном контексте нет информации". Это снижает число
галлюцинаций до нуля на тестовом наборе.

##Ограничения
Работает только со статическими сайтами. Для SPA нужен Playwright.
Один источник. Для нескольких URL — расширить web_paths.
Нет reranking. Cross-encoder улучшил бы retrieval.
Нет streaming. Легко добавить через stream=True.
Нет автоматической оценки. RAGAS — следующий шаг.


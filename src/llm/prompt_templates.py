"""
Prompt templates used across the RAG pipeline and MCP tools.
Centralising prompts here makes it easy to iterate and version them.
"""
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

# ── RAG answer prompt ─────────────────────────────────────────────────────────

RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful assistant that answers questions using the provided context.

Rules:
- Answer using the information in the context below.
- If the context is marked [Web Search Results], your answer is based on live web search — mention this briefly.
- Cite the source number [1], [2], etc. when using a specific passage.
- Be concise but complete.
- Do NOT fabricate information not found in the context.
- If the context is empty or truly irrelevant, say so and offer general guidance if possible.

Context:
{context}""",
    ),
    ("human", "{question}"),
])

# ── Condense question prompt (for conversational RAG) ─────────────────────────

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(
    """Given a chat history and the latest user question, rewrite the question
as a standalone question that is fully self-contained (without needing the history).

Chat History:
{chat_history}

Latest Question: {question}

Standalone Question:"""
)

# ── Document summarisation prompt ─────────────────────────────────────────────

SUMMARISE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert summariser. Produce a concise summary that preserves "
        "the key facts, arguments, and conclusions of the provided document. "
        "Use bullet points when appropriate.",
    ),
    ("human", "Summarise the following document:\n\n{document}"),
])

# ── Keyword extraction prompt ─────────────────────────────────────────────────

KEYWORD_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Extract the 5-10 most important keywords or key phrases from the text below. "
               "Return them as a comma-separated list, no explanations."),
    ("human", "{text}"),
])

# ── Question generation prompt (for evaluation / data synthesis) ──────────────

QUESTION_GEN_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a question generation assistant. Given a passage, generate {n} diverse "
        "questions that can be answered from the passage. Return one question per line.",
    ),
    ("human", "Passage:\n{passage}"),
])

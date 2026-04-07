"""
MCP Concept — Prompts
======================
MCP Prompts are reusable, parameterised prompt templates that clients
can request and fill in. Unlike tools, prompts do not execute code —
they return structured message lists that the client sends to an LLM.

Prompts registered here:
  1. rag_answer         — standard RAG answer prompt
  2. document_analysis  — deep analysis template for a document
  3. qa_evaluation      — evaluate a RAG answer against reference
  4. system_prompt      — configurable system persona
"""
from __future__ import annotations

import logging

from mcp.types import GetPromptResult, PromptMessage, TextContent

logger = logging.getLogger(__name__)


def register_prompts(mcp):
    """Register all prompts on the given FastMCP instance."""

    # ── 1. RAG answer prompt ──────────────────────────────────────────────────

    @mcp.prompt(
        name="rag_answer",
        description=(
            "Standard RAG answer prompt. Provide a question and retrieved context; "
            "the LLM will answer using only the supplied context."
        ),
    )
    def rag_answer_prompt(
        question: str,
        context: str,
        language: str = "English",
    ) -> GetPromptResult:
        """Build a RAG answer prompt from question + context."""
        return GetPromptResult(
            description="RAG answer generation prompt",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=(
                            f"Answer the following question in {language} using ONLY "
                            f"the provided context. Cite sources as [1], [2], etc.\n\n"
                            f"Context:\n{context}\n\n"
                            f"Question: {question}"
                        ),
                    ),
                )
            ],
        )

    # ── 2. Document analysis prompt ───────────────────────────────────────────

    @mcp.prompt(
        name="document_analysis",
        description=(
            "Deep document analysis template. Returns a structured analysis "
            "including: main topics, key facts, entities, sentiment, and gaps."
        ),
    )
    def document_analysis_prompt(
        document: str,
        analysis_depth: str = "standard",  # standard | deep | brief
    ) -> GetPromptResult:
        """Request a structured analysis of a document."""
        depth_instructions = {
            "brief": "Provide a 3-5 bullet summary.",
            "standard": "Include: main topics, key entities, key facts, and your assessment.",
            "deep": (
                "Include: main topics, key entities, key facts, arguments, "
                "counter-arguments, data points, sentiment, potential biases, "
                "and information gaps."
            ),
        }
        instruction = depth_instructions.get(analysis_depth, depth_instructions["standard"])

        return GetPromptResult(
            description="Document analysis prompt",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=(
                            f"Analyse the following document. {instruction}\n\n"
                            f"Document:\n{document}"
                        ),
                    ),
                )
            ],
        )

    # ── 3. RAG evaluation prompt ──────────────────────────────────────────────

    @mcp.prompt(
        name="qa_evaluation",
        description=(
            "Evaluate a RAG-generated answer against a reference answer. "
            "Returns scores for faithfulness, relevance, and completeness."
        ),
    )
    def qa_evaluation_prompt(
        question: str,
        generated_answer: str,
        reference_answer: str,
        context: str = "",
    ) -> GetPromptResult:
        """Build a prompt to evaluate RAG answer quality."""
        context_section = f"\nContext Used:\n{context}\n" if context else ""

        return GetPromptResult(
            description="RAG QA evaluation prompt",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=(
                            "Evaluate the generated answer against the reference answer.\n"
                            "Score each dimension 1-5 and explain your reasoning.\n\n"
                            f"Question: {question}\n"
                            f"{context_section}"
                            f"Generated Answer:\n{generated_answer}\n\n"
                            f"Reference Answer:\n{reference_answer}\n\n"
                            "Dimensions to score:\n"
                            "1. Faithfulness (does it stick to the context?)\n"
                            "2. Relevance (does it answer the question?)\n"
                            "3. Completeness (does it cover all important points?)\n"
                            "4. Conciseness (is it appropriately brief?)\n\n"
                            "Respond in JSON: "
                            '{"faithfulness": <1-5>, "relevance": <1-5>, '
                            '"completeness": <1-5>, "conciseness": <1-5>, '
                            '"overall": <1-5>, "explanation": "..."}'
                        ),
                    ),
                )
            ],
        )

    # ── 4. Configurable system persona ────────────────────────────────────────

    @mcp.prompt(
        name="system_prompt",
        description="Set the system persona for the assistant. Useful for domain specialisation.",
    )
    def system_prompt_template(
        domain: str = "general",
        tone: str = "professional",
    ) -> GetPromptResult:
        """Build a system prompt for a given domain and tone."""
        domain_context = {
            "general": "You are a helpful, knowledgeable assistant.",
            "legal": "You are a legal research assistant with expertise in case law and statutes.",
            "medical": "You are a medical information assistant. Always recommend consulting a doctor.",
            "finance": "You are a financial analysis assistant. Cite data sources carefully.",
            "technical": "You are a technical documentation assistant with deep software knowledge.",
        }
        tone_modifier = {
            "professional": "Respond professionally and formally.",
            "friendly": "Respond in a warm, conversational manner.",
            "concise": "Be extremely concise — one to two sentences maximum per point.",
        }
        persona = domain_context.get(domain, domain_context["general"])
        tone_mod = tone_modifier.get(tone, tone_modifier["professional"])

        return GetPromptResult(
            description="System persona prompt",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(type="text", text=f"{persona} {tone_mod}"),
                )
            ],
        )

    logger.info("Registered %d MCP prompts", 4)

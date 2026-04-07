# How GotoAssistant Works — Plain English Guide

This document explains how GotoAssistant works under the hood. No technical background needed — everything is explained with simple words and everyday comparisons.

---

## The Big Picture

Imagine you hire a very fast research assistant. You hand them a stack of documents. They read everything, highlight the key parts, and file them away. When you ask a question, they quickly find the most relevant highlighted sections, hand them to a smart advisor (GPT-4), who then reads those sections and writes you a clear answer.

GotoAssistant does exactly this — automatically, in seconds.

The process has three phases:
1. **Reading** — the app reads your documents or websites
2. **Storing** — it saves the content in a searchable way
3. **Answering** — when you ask a question, it finds the right content and asks GPT-4 to explain it

---

## Phase 1: Reading Documents

### How it reads websites

When you paste a website link, the app visits every page on that website — just like a human would — and reads the text on each page.

Some websites are normal (the text is already in the page's code). Others are "modern" apps built with frameworks like React — the page starts blank and only fills in after some code runs. To handle these, the app uses a real web browser (Google Chrome, running invisibly in the background) to load the page properly before reading it.

The app is smart enough to detect which type of website it is dealing with and switch to the right approach automatically.

### How it reads files

- **PDF** — extracts the text from each page
- **Word documents (.docx)** — reads the formatted text
- **Text files (.txt, .md)** — reads as-is

### Splitting into small pieces

A 100-page PDF cannot be stuffed into one question to GPT-4 — there is a size limit. So the app cuts documents into small pieces (called "chunks"), roughly 350 words each. Each chunk slightly overlaps with the next, so nothing important gets cut off at a boundary.

Think of it like cutting a long newspaper article into index cards, with each card sharing the last sentence of the previous one.

---

## Phase 2: Storing (the "Memory")

### Turning text into numbers

After splitting, the app converts each chunk of text into a list of numbers (called an "embedding"). This is done by a special AI model that runs locally on the computer — no internet needed, no cost.

Two chunks about similar topics will produce similar sets of numbers. Two chunks about completely different topics will produce very different sets of numbers.

This is how the app can do **meaning-based search** rather than keyword matching. If you ask "how do I store files programmatically?", it will find chunks that talk about "uploading content" or "ingesting documents" even if they never use the word "store".

### The filing cabinet (ChromaDB)

All these numbered chunks are saved in a local database called ChromaDB — think of it as a very smart filing cabinet. It can instantly find the chunks whose numbers are closest to your question's numbers, even if you have stored thousands of pages.

The data is saved to your computer's disk, so it survives if you restart the app.

---

## Phase 3: Answering Questions

When you type a question, here is what happens step by step:

### Step 1: Rephrase the question

The app quietly generates 3 different ways to ask the same question. For example:

- You ask: *"How do I upload a file?"*
- It also searches for: *"Document ingestion steps"*, *"Adding files to the system"*, *"How to store content"*

This increases the chance of finding the right information, since the documents might use different words than you do.

### Step 2: Search the filing cabinet

Each version of the question is converted to numbers, and the filing cabinet finds the chunks with the closest matching numbers. It retrieves a list of candidate chunks.

It also avoids picking the same chunk twice or picking five chunks that all say the same thing — it deliberately picks a diverse set of relevant chunks.

### Step 3: Rank the results

A second small AI model (running locally) re-reads the question and each candidate chunk together and scores how well each one actually answers the question. The best-scoring chunks are kept.

Think of this as a second opinion — the first search casts a wide net, the ranking step picks the best catch.

### Step 4: Web search fallback

If the filing cabinet is empty (no documents have been added) or nothing relevant was found, the app automatically searches the internet using DuckDuckGo and uses those results instead.

### Step 5: Ask GPT-4

The final selected chunks are shown to GPT-4 along with the question and an instruction: "Answer using only this information, and say where each fact came from."

GPT-4 reads the chunks and writes a clear, natural-language answer. The chunks are shown as sources at the bottom of the response so you can verify the information.

---

## The Web Interface

The chat interface you see in your browser is a single HTML page — no complicated framework. It sends your question to the app, streams the answer back word by word (so it appears as it is being written, like ChatGPT), and displays the sources.

The colour theme automatically matches your computer's light/dark mode setting.

---

## Web API (for developers)

GotoAssistant also works as a web API — meaning other programs can talk to it. Every feature in the UI is available as an API call. The interactive documentation is at `http://localhost:9000/docs`.

---

## MCP — Connecting to Claude Desktop

GotoAssistant also speaks a special language called **MCP (Model Context Protocol)**. This is a standard invented by Anthropic (the company behind Claude) that lets AI assistants talk to external tools.

In simple terms: once you connect GotoAssistant to Claude Desktop, Claude can automatically search your documents and add content to your knowledge base during a conversation — without you manually switching apps.

Think of it like giving Claude Desktop a custom plugin that knows your documents.

---

## Why each tool was chosen

### Why OpenAI GPT-4?
It is the best widely-available AI for understanding documents and writing clear answers. The `gpt-4o-mini` version is used by default because it is fast and 95% cheaper than the full `gpt-4o` while giving nearly the same quality for question-answering.

### Why does the embedding model run locally (for free)?
Converting text to numbers (embeddings) happens thousands of times — once for every chunk of every document. Using an online API for this would be slow and expensive. The local model (`BAAI/bge-small-en-v1.5`) is only 33 million parameters — tiny, fast, and completely free to run.

### Why ChromaDB for storage?
It is free, needs no separate server, and saves data to a regular folder on your disk. Other options like Pinecone require a paid account and an internet connection. ChromaDB works out of the box with zero setup.

### Why use a re-ranker?
The first search is like using a map to find a neighbourhood. The re-ranker is like walking around the neighbourhood to find the exact house. It is slower but much more precise — and it runs locally in about 10ms per search, so the user barely notices.

### Why use Docker?
Docker packages the entire app (Python, all libraries, Chrome browser) into one neat box that runs the same way on any computer. Without Docker, installing the app requires matching dozens of library versions on your specific machine. With Docker: one command and it works.

---

## Summary

| What it does | How it does it |
|---|---|
| Reads websites | Uses a real Chrome browser (invisible) to load and read pages |
| Reads files | Extracts text from PDF, Word, and text files |
| Stores knowledge | Cuts text into small pieces and saves them as numbers in a local database |
| Finds relevant content | Converts your question to numbers and finds matching chunks |
| Answers questions | Gives the relevant chunks to GPT-4 and asks it to explain |
| Falls back to web | Searches DuckDuckGo if no local documents match |
| Shows sources | Lists which document or webpage each fact came from |

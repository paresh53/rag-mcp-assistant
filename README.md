# GotoAssistant

**GotoAssistant** is a smart chat app that can read any website or document and then answer your questions about it — powered by OpenAI's GPT-4.

Think of it like this: you give it a website or a PDF, it "reads" and "remembers" that content. Then you ask questions in plain English and it gives you clear, accurate answers — with links to where it found the information.

---

## What can it do?

- **Learn from a website** — paste any website link and it reads every page automatically
- **Learn from a file** — upload a PDF, Word document, or text file
- **Answer your questions** — ask anything in plain English and get a smart answer
- **Search the web** — if it doesn't find the answer in your documents, it searches the internet automatically
- **Remember the conversation** — you can ask follow-up questions naturally

---

## How to open the app

If you are running it locally:

```
http://localhost:9000/
```

If you are using GitHub Codespaces:

```
https://vigilant-space-giggle-q99g57vv9j296w6-9000.app.github.dev
```

---

## How to use it

### Step 1 — Teach it something (optional)

In the left sidebar, find the **"Add Knowledge"** section:

- **Website** — paste a website address (e.g. `https://docs.python.org`) and click **Crawl**. It will read all the pages automatically.
- **Single page** — paste one page's URL and click **Add**.
- **File** — upload a PDF, Word doc, or text file.

You can skip this step entirely. If you ask a question without any documents, it will search the internet for you.

### Step 2 — Ask your question

Type your question in the chat box at the bottom and press **Enter**.

GotoAssistant will find the relevant information and give you a clear answer.

### Step 3 — Start fresh

Click **"Clear All Knowledge"** in the sidebar to remove all documents and start over.

---

## How to run it yourself

### What you need first

- Python 3.11 or newer (download from python.org)
- An OpenAI API key — get one free at [platform.openai.com/api-keys](https://platform.openai.com/api-keys)

### Install

```bash
git clone https://github.com/paresh53/rag-mcp-assistant.git
cd rag-mcp-assistant
pip install --upgrade setuptools pip
pip install -e ".[dev]"
pip install playwright && python -m playwright install chromium
```

### Set your API key

```bash
cp .env.example .env
# Open the .env file and add your key: OPENAI_API_KEY=sk-...
```

### Start the app

```bash
python main.py
```

Then open your browser and go to **http://localhost:9000/**

---

## Run with Docker (even easier)

If you have Docker installed:

```bash
cp .env.example .env   # then add your OPENAI_API_KEY inside .env
docker-compose up --build
```

Open **http://localhost:9000/** in your browser.

---

## Settings you can change

Open the `.env` file and edit these:

| Setting | Default | What it does |
|---|---|---|
| `OPENAI_API_KEY` | *(required)* | Your OpenAI API key — the app won't work without this |
| `LLM_MODEL` | `gpt-4o-mini` | Which AI model to use for answering questions |
| `API_PORT` | `9000` | Which port the app runs on |

Everything else has sensible defaults — you don't need to change anything else.

---

## For developers — API

You can also send requests directly to GotoAssistant from your own code. A full interactive guide is at:

```
http://localhost:9000/docs
```

Quick examples:

```bash
# Add a webpage to its knowledge
curl -X POST http://localhost:9000/documents/ingest/url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://docs.python.org/3/library/functions.html"}'

# Ask a question
curl -X POST http://localhost:9000/rag/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What does the len() function do?", "collection": ""}'
```

---

## License

MIT

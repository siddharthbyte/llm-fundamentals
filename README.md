# LLM Fundamentals: Tokens, Cost and Latency

Hands-on notebook for Module 1 of my AI engineering learning path. It measures three things that decide how an LLM app behaves in practice: **how many tokens text uses, what that costs, and how fast a model responds.**

## What this notebook does

1. **Token counting** with `tiktoken` on four texts: English, Hindi (same meaning), code and numbers.
2. **Cost estimation**: turns token counts into the cost per request and for 10,000 requests, using published per-million-token prices.
3. **Latency measurement**: calls two free-tier LLM APIs (Groq and Google Gemini) with streaming and records time to first token, total time and the token usage reported by the API.

## Key findings

> Fill in your own numbers from your notebook before pushing.

| Experiment | Result |
|---|---|
| Hindi vs English token count | Hindi used ___ tokens vs ___ for English (about ___x more) |
| Cost for 10,000 requests (English vs Hindi) | $___ vs $___ |
| Groq time to first token (3 runs) | about ___ s |
| Groq short answer vs long essay (total time) | ___ s vs ___ s |
| Gemini time to first token (free tier) | ___ s |
| Same prompt, input tokens counted | Groq ___ vs Gemini ___ |

## What I learned

- Text in Hindi splits into many more tokens than English, so it costs more and takes more of the context window for the same meaning.
- Total response time grows with the **length of the answer**, while time to first token stays roughly the same.
- Output length changes between runs of the same prompt, so output cost can only be estimated in advance.
- Different providers report different input token counts for the same prompt, so `tiktoken` gives an estimate and the API's usage numbers are the source of truth for billing.
- Reasoning models can produce hidden "thinking" tokens that are billed but not shown.
- Free tiers can be slow or rate-limited, so real apps need timeouts and retries.

## Tech used

Python, Jupyter, `tiktoken`, `pandas`, `openai` client (pointed at Groq's and Gemini's OpenAI-compatible endpoints), `python-dotenv`.

## How to run it

1. Clone the repo and create a virtual environment:

   ```
   python -m venv llm
   llm\Scripts\activate        # Windows
   pip install -r requirements.txt
   ```

2. Get free API keys (no credit card needed at the time of writing): one from Google AI Studio and one from Groq.

3. Create a `.env` file in the project folder (never commit it):

   ```
   GROQ_API_KEY=your_groq_key
   GEMINI_API_KEY=your_gemini_key
   ```

4. Open `day1.ipynb` and run the cells. Check each provider's current model names, since they change often.

## Notes

- Free-tier limits and model names change frequently, so check each provider's dashboard.
- Don't send private or personal data through free-tier APIs.
- Latency numbers depend on the time of day, the provider's load and my network, so they are indicative, not benchmarks.

## Part of

A 6-module path: LLM fundamentals, building LLM apps, prompt engineering, RAG, agents, shipping.

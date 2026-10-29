# Day 2: Text vs Image Input in Multimodal LLMs

## Overview

The objective of this experiment is to compare text-only input and image input in Large Language Models (LLMs).

Using Python, I tested two models through their APIs to observe how the input modality affects token usage.

The experiment focuses on:

- Input token consumption
- Output token consumption
- Differences between text and image inputs
- Comparison of token usage across model providers

## Models Used

1. **Groq:** Qwen model (`qwen/qwen3.8-27b`)
2. **Google Gemini:** Gemini model configured in the existing experiment

## Technologies Used

- Python
- OpenAI Python SDK
- Groq API
- Google Gemini API
- python-dotenv
- Pandas
- Base64 image encoding

## Experimental Methodology

The experiment was conducted in the following steps:

1. Configured API credentials using a `.env` file.
2. Initialized API clients for Groq and Gemini using the OpenAI-compatible API interface.
3. Prepared a text prompt and an image input.
4. Sent text-only requests to both models.
5. Sent image-based requests to both models.
6. Recorded the input and output token counts reported by each API.
7. Used Pandas to organize the results into a comparative DataFrame.

No additional API requests were required to construct the final comparison table.

## Experimental Results

| Model | Input Type | Input Tokens | Output Tokens |
|---|---|---:|---:|
| Groq - Qwen | Text | 88 | 292 |
| Groq - Qwen | Image | 1817 | 13 |
| Gemini | Text | 20 | 76 |
| Gemini | Image | 1092 | 17 |

## Analysis and Observations

### 1. Input Token Usage

**Groq - Qwen**
- Text input: 88 tokens
- Image input: 1817 tokens
- Image input used approximately 20.65 times as many reported input tokens.

**Gemini**
- Text input: 20 tokens
- Image input: 1092 tokens
- Image input used approximately 54.6 times as many reported input tokens.

Both models reported substantially higher input token usage for image-based requests.

### 2. Output Token Usage

**Groq - Qwen**
- Text output: 292 tokens
- Image output: 13 tokens

**Gemini**
- Text output: 76 tokens
- Image output: 17 tokens

The image-based requests generated shorter responses than the text-based requests in this experiment.

### 3. Key Findings

- Image-based requests consumed more reported input tokens than text-only requests for both tested models.
- Token usage differed between Groq and Gemini.
- Output token consumption depended on the generated response, not only on the input modality.
- Token counts alone cannot determine which model is cheaper or better.

## Limitations

- Only one image and a limited number of requests were tested.
- The text and image inputs did not provide information in exactly equivalent formats.
- API-reported token counts may be calculated differently across providers and models.
- Response latency was not recorded for the image requests.
- Actual API cost was not calculated.
- The experiment did not quantitatively evaluate answer accuracy.

Therefore, the findings apply to this specific experiment and should not be generalized to all LLMs or image inputs.

## Conclusion

This experiment demonstrated that image-based requests can consume substantially more reported input tokens than text-only requests.

However, token consumption is only one aspect of model efficiency. A complete comparison should also consider API pricing, response latency, answer quality, image resolution, and the specific task being performed.

## Skills Practiced

- Working with OpenAI-compatible APIs
- Configuring API keys with environment variables
- Sending text and image inputs to LLMs
- Encoding image files using Base64
- Extracting API token usage
- Creating comparative DataFrames with Pandas
- Interpreting experimental results

---
---

# Experiment 2: LLM Generation Parameters and Reasoning Models

## 1. Temperature

### What is Temperature?

Temperature is a generation parameter that controls the randomness of token selection in a language model.

An LLM calculates scores for possible next tokens and converts them into probabilities. Temperature modifies this probability distribution before token selection.

The mathematical relationship is:

\[
P_i(T)=\frac{e^{z_i/T}}{\sum_j e^{z_j/T}}
\]

Where:

- \(P_i(T)\) is the probability of selecting token \(i\).
- \(z_i\) is the logit, or raw score, of that token.
- \(T\) is the temperature parameter.

### Effect of Temperature

| Temperature | Expected behavior |
|---|---|
| 0.0 | Usually favors the highest-probability token, where supported |
| 0.5 | Relatively focused and predictable generation |
| 1.0 | A common baseline for sampling |
| Higher values | Generally allow more variation in token selection |

These are general guidelines. The actual behavior depends on the model and provider, and some models do not support arbitrary temperature values.

### Experimental Setup

The same prompt was used with different temperature settings:

```python
TEMPERATURE_PROMPT = """
Explain what a Python list is in simple words.
Give one example.
"""

TEMPERATURES = [0.0, 0.5, 1.0]
```

A maximum output-token limit was also introduced:

```python
temperature=temp,
max_tokens=100
```

Here, temperature is the experimental variable, while the output-token limit remains fixed.

### Observations

- Lower temperatures generally make generation more predictable.
- Higher temperatures can produce more varied answers.
- Temperature does not directly increase the model's knowledge or guarantee creativity.
- A maximum output-token limit controls the generation budget and may truncate an answer if it is too small.
- Temperature support and parameter behavior vary between models.

No universal conclusion should be drawn without comparing the actual responses.

---

## 2. What Are Reasoning Models?

Reasoning models are language models designed or trained to perform better on complex, multi-step problems by using additional computation during inference.

They are particularly useful for tasks that involve logical deduction, mathematical problem-solving, complex programming, planning, and analyzing multiple constraints.

Many reasoning models are still based on transformer architectures. Their differences can arise from specialized training, reinforcement learning, inference-time computation, and model-specific reasoning mechanisms.

### Standard LLMs vs Reasoning Models

| Feature | Standard LLM | Reasoning-oriented LLM |
|---|---|---|
| Main focus | General language tasks | Complex multi-step problems |
| Simple questions | Usually efficient | May be unnecessary for simple tasks |
| Complex reasoning | Can perform well | Often optimized for difficult reasoning |
| Latency | Often lower for simple tasks | May be higher on difficult tasks |
| Token usage | Depends on the response | May include additional reasoning tokens |
| Accuracy | Depends on model and task | Can improve on selected reasoning tasks, but is not guaranteed |

Reasoning models are not automatically better at every task. Their usefulness depends on the complexity of the problem and the cost of additional computation.

---

## 3. Question 1: What Is a Reasoning Token, and Does It Appear in Usage Numbers?

A reasoning token is a token generated as part of a model's internal reasoning process, rather than as part of the final user-facing answer.

For example, when solving a mathematical problem, a reasoning-oriented model may perform intermediate computations before returning its final answer.

These intermediate tokens may not appear in the final response shown to the user.

### Do reasoning tokens count toward usage?

**Yes, they can count toward usage and billing.** The exact accounting depends on the model and API.

An API may report:

- Input tokens
- Output tokens
- Reasoning tokens as a separate subset or detail of output usage
- Other model-specific usage categories

Reasoning tokens may be included in the reported output-token total, even if they are not visible in the final answer. They should not automatically be added to the output-token total a second time.

For example, a hypothetical usage report might contain:

| Metric | Tokens |
|---|---:|
| Input tokens | 500 |
| Total output tokens | 300 |
| Reasoning tokens included in output | 180 |
| Visible final-answer tokens | 120 |

In this illustration, reasoning tokens are part of the 300 output tokens. The total is not 480 output tokens.

Actual usage fields and billing rules must be verified in the documentation for the selected model.

---

## 4. Question 2: How Do You Control How Much the Model Thinks?

Some APIs offer controls for reasoning effort or a thinking-token budget.

### A. Reasoning Effort

A reasoning-effort setting controls the amount of reasoning effort requested from a compatible model.

Conceptual examples include:

- Low effort: prioritize speed for relatively simple tasks.
- Medium effort: balance speed and problem-solving capability.
- High effort: allow more computation for complex problems.

The supported settings depend on the model and provider. A reasoning-effort parameter is not universally available, and higher effort does not guarantee a correct answer.

### B. Thinking Budget

Some APIs allow the developer to configure a budget for internal reasoning.

A thinking budget is distinct from the number of tokens displayed in the final answer. Depending on the API, reasoning tokens may also consume part of a broader output-token limit.

For example, a model may use additional internal computation to solve a problem but return a short final answer.

### C. Maximum Output Tokens

Maximum output tokens limit the permitted generation budget. They are not necessarily a direct control over reasoning effort.

A model can use a large reasoning budget and return a short answer, or use little reasoning and return a long explanation.

Therefore, reasoning effort, thinking budget, and maximum output tokens should not be treated as interchangeable settings.

### Important Implementation Note

Reasoning controls differ across providers. Some models reject temperature settings or stop sequences, while others provide dedicated reasoning configuration options.

Always check the API documentation for the specific model before passing these parameters.

---

## 5. Question 3: Which Tasks Benefit From Reasoning Models, and Which Do Not?

### Tasks That May Benefit

- Complex mathematical problems
- Algorithm design and analysis
- Difficult programming and debugging
- Multi-step logical deduction
- Planning with several dependencies
- Evaluating conflicting evidence
- Problems requiring multiple constraints to be satisfied

These tasks can benefit from additional computation because intermediate steps may be necessary to reach a reliable answer.

### Tasks That Often Do Not Need Extensive Reasoning

- Text formatting
- Simple translations
- Basic summarization
- Extracting a clearly stated field
- Converting text to uppercase
- Straightforward classification
- Simple factual questions when reliable information is already available

For such tasks, a faster general-purpose model may provide sufficient quality at lower latency and cost.

### Practical Comparison

| Task | Suggested starting approach |
|---|---|
| Convert a sentence to uppercase | Standard LLM |
| Summarize a short paragraph | Standard LLM |
| Extract an email address from text | Standard LLM |
| Debug a complex algorithm | Reasoning-oriented model |
| Solve a multi-step mathematical problem | Reasoning-oriented model |
| Plan a project with many constraints | Reasoning-oriented model |

These are recommendations, not strict rules. A standard model may perform well on complex tasks, and a reasoning model may still make mistakes.

---

## 6. Key Takeaways

1. Temperature influences how the model selects tokens and how varied the generated response may be.
2. Maximum output tokens limit the generation budget but do not guarantee a particular answer length.
3. Reasoning models may use additional computation to solve complex problems.
4. Reasoning tokens can contribute to API usage and billing even when they are not visible in the final answer.
5. Reasoning effort, thinking budgets, and output-token limits control different aspects of generation.
6. Complex reasoning tasks may benefit from reasoning-oriented models, while simpler tasks often do not require extensive reasoning.
7. Model capabilities and API parameters vary across providers, so experimental results must be interpreted in context.

## Conclusion

This experiment explored important LLM generation parameters and the concept of reasoning-oriented models.

Understanding these controls helps developers balance response quality, consistency, latency, token usage, and cost when building LLM applications.

The next step is to test these settings experimentally, changing one parameter at a time and recording the resulting outputs and usage metrics.


**Day 2 Complete:** 

Part of my ongoing journey learning LLMs, multimodal AI, and practical API experimentation.
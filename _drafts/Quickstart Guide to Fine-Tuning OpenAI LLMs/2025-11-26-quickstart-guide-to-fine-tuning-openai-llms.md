---
layout: post
title: "Fine-Tuning OpenAI LLMs: A Quickstart Guide"
date: 2025-12-13 14:01 -0700
description: <enter description>
tag:
  - LLM
  - GenAI
  - fine-tuning
  - machine-learning 
  - OpenAI 
  - python
  
image: /assets/img/llm-fine-tuning-go-emotions/fine-tuning-blog-title.jpg
---
## What Is LLM Fine-Tuning?

Fine-tuning is the process of taking a pre-trained LLM and adapting it to perform especially well on a narrow problem. With today’s ecosystem of models, from open-source options on Hugging Face to hosted models from OpenAI, Anthropic, and Google, most models are already capable out of the box. Fine-tuning pushes them further by training on examples that closely match the task and responses you care about.

In this post, we fine-tune [GPT-3.5 Turbo](https://platform.openai.com/docs/models/gpt-3.5-turbo) to improve its performance on labeling text with emotion categories. You will see how a well fine-tuned and relatively inexpensive model can outperform more powerful alternatives when the task is clearly defined.
 
---

## When Does It Make Sense to Fine-Tune?

**TL;DR:** Fine-tuning usually has higher upfront cost and setup time than prompt engineering or RAG, but it excels when you want more consistent behavior. This includes tighter control over style, output format, and reliability on a well-defined task.

Even though this post focuses on fine-tuning, in practice you will often get the best results by combining all three techniques:

1. **Prompt engineering:** Iteratively refine the prompt with clear instructions (and a few examples when helpful) to steer the model’s behavior.
2. **Retrieval-augmented generation (RAG):** Pull in the right external or internal context at runtime by retrieving relevant information from your data sources.
3. **Fine-tuning:** Train on task-specific examples to “bake in” the behavior you want, especially for consistent formatting, tone, and accuracy.

If you want a deeper comparison, this IBM article is a good reference: [RAG vs. fine-tuning vs. prompt engineering](https://www.ibm.com/think/topics/rag-vs-fine-tuning-vs-prompt-engineering#:~:text=Prompt%20engineering%20optimizes%20input%20prompts,relevant%20data%20for%20greater%20accuracy.)

---

## What You’ll Have by the End

  - The process for preparing training and validation datasets in JSONL format for fine-tuning
  - A fine-tuned OpenAI model you can use in the OpenAI Playground or via an API endpoint for emotion labeling
  - A simple way to compare the base model versus the fine-tuned model on the same emotion labeling examples
  - Links to the GitHub repo with the resources needed to replicate this end-to-end

---

## About the Dataset I Used

This is an open-source dataset released by Google Research. In their own words, it is “a human-annotated dataset of 58k Reddit comments extracted from popular English-language subreddits and labeled with 27 emotion categories.” 

Each example in the dataset consists of a Reddit comment paired with one or more labels representing predefined emotions. The figure below shows a few sample annotations, taken directly from the Google Research blog post. 

<figure>
  <img src="/assets/img/llm-fine-tuning-go-emotions/go-emotions-examples.png" alt="GoEmotions examples">
  <figcaption>
    Fig 1. Some examples of comments and emotion labels.
    Reference:
    <a href="https://research.google/blog/goemotions-a-dataset-for-fine-grained-emotion-classification/" target="_blank">
      Google Research
    </a>
  </figcaption>
</figure>

You can read more about the dataset here: [GoEmotions: A Dataset for Fine-Grained Emotion Classification](https://research.google/blog/goemotions-a-dataset-for-fine-grained-emotion-classification/)

---

## Prerequisites

Before you kick-off with this exercise, do the following: 

- Set up an account for yourself on the [OpenAI Platform](https://platform.openai.com/docs/overview) and create an API key for this project.
- Clone and download the project's Github repo - [llm-finetuning-go-emotions](https://github.com/adikamath/llm-finetuning-go-emotions)
- Set up a Python virtual environment locally and install required packages listed in [requirements.txt](https://github.com/adikamath/llm-finetuning-go-emotions/blob/main/requirements.txt)

> **Reference:** I also followed parts of this excellent YouTube tutorial during my experiment:  
> 🎥 [Fine-tuning tutorial by Adam Lucek](https://www.youtube.com/watch?v=GZ4W1nRw_Ac)

---

## Step 1 – Preparing the Training and Validation Data

<!-- Explain at a high level how you:
     - Loaded the dataset into a notebook
     - Did basic cleaning (dropping empty rows, normalizing labels)
     - Split into train vs validation sets. -->

**Notebook snapshot (data exploration)**

![Notebook preview – exploring and cleaning the dataset](/assets/images/fine-tune-notebook-preview.png)

<!-- Screenshot idea:
     - A cropped Jupyter cell showing df.head() and maybe some basic cleaning steps. -->

---

## Step 2 – Validating the Dataset & Estimating Cost

<!-- Describe:
     - How you validated the JSONL structure (e.g., simple checks in Python, or OpenAI’s file validation)
     - How you estimated token counts and rough training cost
     - Any quick helper functions or sanity checks you ran. -->

**Validation and token stats snapshot**

![Notebook preview – dataset validation and token estimate](/assets/images/fine-tune-validation-preview.png)

<!-- Screenshot idea:
     - Notebook cell printing token counts, number of examples, etc.
     - Or an editor view with a valid JSONL snippet. -->

---

## Step 3 – Fine-Tuning the Model in the OpenAI UI

<!-- Walk through:
     - Going to the Fine-tuning section in the OpenAI dashboard
     - Uploading train.jsonl and validation.jsonl
     - Selecting a base model
     - Starting the fine-tuning job. Keep it high-level and friendly. -->

**OpenAI fine-tuning setup**

![OpenAI fine-tuning UI – file upload and model configuration](/assets/images/fine-tune-openai-ui-setup.png)

<!-- Screenshot idea:
     - The fine-tuning creation screen showing file selection + base model. -->

**Fine-tuning job status**

![OpenAI fine-tuning UI – job status view](/assets/images/fine-tune-openai-ui-status.png)

<!-- Screenshot idea:
     - The dashboard showing the job queued/running/completed. -->

---

## Step 4 – Testing and Comparing the Fine-Tuned Model

<!-- Explain how you:
     - Grabbed the new fine-tuned model name from the UI
     - Sent a few test prompts using a small Python script or notebook
     - Compared base model vs fine-tuned responses (qualitatively or with simple metrics). -->

**Comparison snapshot (optional)**

![Notebook preview – comparing base vs fine-tuned responses](/assets/images/fine-tune-comparison-preview.png)

<!-- Screenshot idea:
     - A notebook cell where you print base vs fine-tuned outputs for a few examples.
     - Even a simple text table is fine. -->

---

## Lessons Learned from This Experiment

<!-- 4–6 bullets of reflection:
     - Data formatting took more time than the fine-tuning itself
     - Starting with a smaller dataset made iteration easier
     - Validation + token estimates helped avoid surprises
     - The UI is great for a first run
     - Anything you’d do differently next time. -->

---

## How You Can Try This Yourself

<!-- Short, actionable checklist:
     - Pick a simple task (e.g., classification, tagging, formatting)
     - Assemble a small labeled dataset
     - Use a notebook to clean data and write JSONL
     - Upload files in the OpenAI UI and run a fine-tune
     - Compare the new model to the base model.
     Add a link to your GitHub repo here. -->

> 🔗 **GitHub repo:** [llm-finetuning-go-emotions](https://github.com/adikamath/llm-finetuning-go-emotions)

---

## Closing Thoughts

<!-- 2–3 sentences:
     - Fine-tuning felt intimidating but the basic loop is simple
     - Encourage the reader to run a small, imperfect experiment of their own. -->
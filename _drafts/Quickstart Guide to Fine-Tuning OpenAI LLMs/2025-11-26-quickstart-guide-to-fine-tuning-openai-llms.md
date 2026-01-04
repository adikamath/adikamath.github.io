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

In this post, we fine-tune [GPT-3.5 Turbo](https://platform.openai.com/docs/models/gpt-3.5-turbo){:target="_blank" rel="noopener noreferrer"} to improve its performance on labeling text with emotion categories. You will see how a well fine-tuned and relatively inexpensive model can outperform more powerful alternatives when the task is clearly defined.
 
---

## When Does It Make Sense to Fine-Tune?

**TL;DR:** Fine-tuning usually has higher upfront cost and setup time than prompt engineering or RAG, but it excels when you want more consistent behavior. This includes tighter control over style, output format, and reliability on a well-defined task.

Even though this post focuses on fine-tuning, in practice you will often get the best results by combining all three techniques:

1. **Prompt engineering:** Iteratively refine the prompt with clear instructions (and a few examples when helpful) to steer the model’s behavior.
2. **Retrieval-augmented generation (RAG):** Pull in the right external or internal context at runtime by retrieving relevant information from your data sources.
3. **Fine-tuning:** Train on task-specific examples to “bake in” the behavior you want, especially for consistent formatting, tone, and accuracy.

If you want a deeper comparison, this IBM article is a good reference: [RAG vs. fine-tuning vs. prompt engineering](https://www.ibm.com/think/topics/rag-vs-fine-tuning-vs-prompt-engineering#:~:text=Prompt%20engineering%20optimizes%20input%20prompts,relevant%20data%20for%20greater%20accuracy.){:target="_blank" rel="noopener noreferrer"}

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

You can read more about the dataset here: [GoEmotions: A Dataset for Fine-Grained Emotion Classification](https://research.google/blog/goemotions-a-dataset-for-fine-grained-emotion-classification/){:target="_blank" rel="noopener noreferrer"}

---

## Prerequisites

Before you kick-off with this exercise, do the following: 

- Set up an account for yourself on the [OpenAI Platform](https://platform.openai.com/docs/overview){:target="_blank" rel="noopener noreferrer"} and create an API key for this project.
- Clone and download the project's Github repo - [llm-finetuning-go-emotions](https://github.com/adikamath/llm-finetuning-go-emotions){:target="_blank" rel="noopener noreferrer"}
- Set up a Python virtual environment locally and install required packages listed in [requirements.txt](https://github.com/adikamath/llm-finetuning-go-emotions/blob/main/requirements.txt){:target="_blank" rel="noopener noreferrer"}

> **Reference:** I also followed parts of this excellent YouTube tutorial during my experiment:  
> 🎥 [Fine-tuning tutorial by Adam Lucek](https://www.youtube.com/watch?v=GZ4W1nRw_Ac){:target="_blank" rel="noopener noreferrer"}

---

## Step 1 – Preparing the Training and Validation Data

<!-- Explain at a high level how you:
     - Loaded the dataset into a notebook
     - Did basic cleaning (dropping empty rows, normalizing labels)
     - Split into train vs validation sets. --> 

- Using the HuggingFace `datasets` library, load the GoEmotions dataset into the notebook.
- Select the first 1000 rows of the dataset. We'll later split this to create the training and validation datasets.
- Manually create a label index and join it to the text comments. Once you have done that, this is how the dataset will look:
  <figure>
    <img src="/assets/img/llm-fine-tuning-go-emotions/joined_dataset.png" alt="Sample of text comments and their emotion labels">
    <figcaption>Fig 2. A sampling of text comments and their respective emotion labels.</figcaption>
  </figure>
- Analyze the dataset to ensure that all emotion labels are present at least once (i.e., there are no missing labels).
- Here’s the system prompt that I used in my project. Feel free to modify it and see how the performance changes in your case:
  ```python
  prompt_template = """
  You are a highly intelligent emotion classification assistant.
  You carefully read a comment, and label it with one or more pre-selected emotion labels to it.
  The emotion labels are listed here:
  {emotions}
  Your output should be just the emotion label that you are applying to the comment, and if you
  think there are multiple labels then output the labels separated by commas.
  The comment to analyze is here: {comment}
  """
  ```
  
- Next create training and validation datasets in JSONL format. Each line in the JSONL files is one training/validation example. In the case of training, each example will contain the system prompt, the user prompt and the LLM response. See below for the template:

```python
training_example = {
        "messages": [
            {
                "role": "system",
                "content": f"You are a highly intelligent emotion classification assistant. You carefully read a comment, and label it with one or more pre-selected emotion labels to it. The emotion labels are listed here: {emotion_labels_str}.Your output should be just the emotion label that you are applying to the comment, and if you think there are multple labels then output the labels separated by commas."
            },
            {
                "role": "user",
                "content": f"Label the emotion of this comment: {row['comment']}"
            },
            {
                "role": "assistant",
                "content": row['emotion labels']
            }
        ]
    }
```

You can find the code in the **Creating training and validation datasets for fine-tuning OpenAI LLMs** section in the notebook.

--- 

## Step 3 – LLM Setup & Performance Comparison 

Now that you have your training and validation datasets prepared, go ahead and setup access to OpenAI LLMs using the API key that you created and then use [GPT-4o mini](https://platform.openai.com/docs/models/gpt-4o-mini){:target="_blank" rel="noopener noreferrer"} to generate labels for the first 100 examples. You can find all of the code in the **# LLM setup for performance comparison** section of the notebook. Here is an overview of all the steps in this section: 
- Create a **.env** file in your project folder and paste your OpenAI API key here (be sure to secure this). 
- Load your OpenAI API key into your environment variable and import the required OpenAI and LangChain libraries.
- Initiliaze the GPT-4o mini model and use LangChain to create an inference chain (you'll use the prompt that you set up earlier as part of the inference chain).
- Run inference on the first 100 examples of the dataset and then join the results back to the dataset. You should end up with something that looks like the dataset snapshot below. We'll come back to this later in the project.
  <figure>
    <img src="/assets/img/llm-fine-tuning-go-emotions/gpt4omini-preview.png" alt="Sample of text comments and their emotion labels">
    <figcaption>Fig 3. A sample of GPT-4o mini generated emotion labels.</figcaption>
  </figure>

You'll notice at a glance that GPT-4o mini has a lot more emotion labels than the default/base-case labels that were part of the dataset.

--- 

## Step 4 – Validating the Dataset & Estimating Cost

Before kicking off fine-tuning, you should run a quick validation pass to make sure the JSONL dataset is well-formed and doesn’t have any obvious issues, such as missing messages or examples that exceed token limits. This step also helps you estimate training cost by counting tokens and factoring in the number of epochs.

This notebook ([finetune_dataset_validation.ipynb](https://github.com/adikamath/llm-finetuning-go-emotions/blob/main/finetune_dataset_validation.ipynb){:target="_blank" rel="noopener noreferrer"}) is meant to be a lightweight pre-flight check. Refer to this notebook and the linked [OpenAI cookbook](https://cookbook.openai.com/examples/chat_finetuning_data_prep){:target="_blank" rel="noopener noreferrer"} if you want to dig into the validation logic or cost calculations in more detail.

---

## Step 3 – Fine-Tuning the Model in the OpenAI UI

<!-- Walk through:
     - Going to the Fine-tuning section in the OpenAI dashboard
     - Uploading train.jsonl and validation.jsonl
     - Selecting a base model
     - Starting the fine-tuning job. Keep it high-level and friendly. --> 

This is where we get to the fun part of configuring and kicking off the fine-tuning job. 

- Sign into your OpenAI Platform account and then head to the **Fine-tuning** section. 
- Click on the **+ Create** button to open up the fine-tuning job configuration form. 
- Choose any latest available version of GPT-3.5 Turbo as the **Base Model**. 
- Enter a value in the **Suffix** field like **llm-finetuning-go-emotions** that will be added into the name of the fine-tuned model to help you identify and differentiate the fine-tuned model.
- Upload the training and validation datasets in JSONL format- **training_data.jsonl** and **validation_data.jsonl** respectively. 
- Let all other fields have the default values and then click **Create** to queue and kick-off fine-tuning. 

You can find my version if the file below. 

<figure>
    <img src="/assets/img/llm-fine-tuning-go-emotions/openai-ft-config1.png" alt="Sample of text comments and their emotion labels">
  </figure>
<figure>
    <img src="/assets/img/llm-fine-tuning-go-emotions/openai-ft-config2.png" alt="Sample of text comments and their emotion labels">
    <figcaption>Fig 4. Fine-tuning job configuration in the OpenAI Platform</figcaption>
  </figure>


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

> 🔗 **GitHub repo:** [llm-finetuning-go-emotions](https://github.com/adikamath/llm-finetuning-go-emotions){:target="_blank" rel="noopener noreferrer"}

---

## Closing Thoughts

<!-- 2–3 sentences:
     - Fine-tuning felt intimidating but the basic loop is simple
     - Encourage the reader to run a small, imperfect experiment of their own. -->
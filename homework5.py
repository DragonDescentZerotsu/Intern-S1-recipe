# %% [markdown]
# # Homework 5: Large Language Models & Prompting
# 
# ## Total Points: 100 points
# - **Overview**: In this assignment, we will examine some of the latest language models you may be familiar with like Google's [Gemma](https://ai.google.dev/gemma) model. We'll cover:
# 
#   - Zero-shot prompting
#   - Prompt engineering
#   - Few-shot prompting
#   - Prompting instruction-tuned models
#   - Chain-of-Thought Reasoning prompting
# 
# - **HuggingFace Account Setup**: You will need a HuggingFace account and authorization token, you can [sign up here](https://huggingface.co/login) and learn [how to get an authorization token here](https://huggingface.co/docs/hub/en/security-tokens). The models accessed with the HuggingFace API are free, but sometimes require acceptance of their terms on the corresponding model page. [Here](https://huggingface.co/google/gemma-2-2b) are the terms acceptance page for Gemma, a recent and powerful LLM that that you will use in this assignment. Make sure to be logged in first, in other to see the terms page.
# 
# - **Deliverables:** This assignment has several deliverables:
#   - Code (this notebook) *(Automatic Graded)*
#     - Section 1: model loading and prompting functionality
#     - Section 2: answers to questions
#     - Section 3: answers to questions
#     - Section 4: answers to question
#     - Section 5: answers to question
#     - Section 6: answers to question
#   - Write Up (Report.pdf) *(Manually Graded)*
#     - Section 2: answers to questions
#     - Section 3: answers to question
#     - Section 4: answers to question
#     - Section 5: answers to question
#     - Section 6: answers to question
# 
# - **Grading**: We will use the auto-grading system called `PennGrader`. To complete the homework assignment, you should implement anything marked with `#TODO` and run the cell with `#PennGrader` note.
# 
# 
# ## Recommended Readings
# - [Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165.pdf). Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, ...others. ArXiV 2020.
# - [Gemma: Open Models Based on Gemini Research and Technology
# ](https://arxiv.org/abs/2403.08295).  Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, ..others. ArXiV 2024.
# - [Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing](https://arxiv.org/pdf/2107.13586.pdf). Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, Graham Neubig. ACM Computing Surveys 2021.
# - [Gemma: Prompt Engineering Guide](https://www.promptingguide.ai/models/gemma). Elvis Saravia. Prompt Engineering Courses 2024
# - [Best practices for prompt engineering with OpenAI API](https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-openai-api). Jessica Shieh. OpenAI 2023.
# - [Training language models to follow instructions with human feedback](https://arxiv.org/pdf/2203.02155.pdf). Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, ...others. ArXiV 2020.
# - [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/pdf/2201.11903.pdf). Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, brian ichter, Fei Xia, Ed H. Chi, Quoc V Le, Denny Zhou. NeurIPS 2022.
# 
# ## To get started, **make a copy** of this colab notebook into your google drive!

# %% [markdown]
# ## Setup 1: PennGrader Setup

# %%
## DO NOT CHANGE ANYTHING, JUST RUN
%%capture
!pip install penngrader-client gdown

# %%
%%writefile notebook-config.yaml
grader_api_url: 'https://23whrwph9h.execute-api.us-east-1.amazonaws.com/default/Grader23'
grader_api_key: 'flfkE736fA6Z8GxMDJe2q8Kfk8UDqjsG3GVqOFOa'

# %%
!cat notebook-config.yaml

# %%
from penngrader.grader import *

## TODO - Start
STUDENT_ID = 25266323 # YOUR PENN-ID GOES HERE AS AN INTEGER#
## TODO - End

SECRET = STUDENT_ID
grader = PennGrader('notebook-config.yaml', 'cis5300_25f_HW5', STUDENT_ID, SECRET)

# %%
# check if the PennGrader is set up correctly
# do not chance this cell, see if you get 4/4!
name_str = 'Mark Yatskar'
grader.grade(test_case_id = 'name_test', answer = name_str)

# %% [markdown]
# ## Setup 2: Dataset / Packages
# - **Run the following cells.**
# - **Agree on the Gemma usage [here](https://huggingface.co/google/gemma-2-2b)** Make sure to be logged in to your HuggingFace Account.
# - **Update your Huggingface Token (see [here](https://huggingface.co/docs/hub/en/security-tokens) on how to get it!)**

# %%
%%capture
!pip install -U transformers datasets bitsandbytes accelerate

# %% [markdown]
#  ⚠️⚠️⚠️ **RESTART RUNTIME FOR PACKAGES TO BE INSTALLED**

# %%
from time import sleep
from datasets import load_dataset
import os
from tqdm import tqdm

IMDB_DATASET = load_dataset("imdb", split='train').shuffle(42)[0:200]
IMDB_DATASET_X = IMDB_DATASET['text']
IMDB_DATASET_Y = IMDB_DATASET['label']
del IMDB_DATASET
os.environ['HF_CACHE'] = '/content/drive/My Drive/transformers_cache'
os.environ['HF_TOKEN'] = '' # TODO

# %% [markdown]
# # Section 1: Setting up Huggingface Models (5 points)
# 
#  In this section you will write the functionality to load and call HuggingFace models.

# %%
from transformers import pipeline
from transformers import BitsAndBytesConfig
from transformers import set_seed
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
)

set_seed(42) # do not change for autograder

pretrained_model_name = "google/gemma-2-2b"
instruction_tuned_model_name = "google/gemma-2-2b-it"

# %%
def load_model_and_tokenizer(
               model_name,
               model_kwargs={
                'quantization_config':quantization_config if torch.cuda.is_available() else None,
                'max_length': 1024,
                'dtype':torch.float16,
                'device_map': 'auto'}
               ):
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  tokenizer.padding_side = "left"
  tokenizer.pad_token = tokenizer.eos_token

  model = AutoModelForCausalLM.from_pretrained(
      model_name,
      **{k: v for k, v in model_kwargs.items() if v is not None}
  )

  ## TODO: load model (with model kwargs) and tokenizer.
  ## Make sure you set the tokenizer padding side to left.
  ## See this discussion to understand why: https://github.com/huggingface/transformers/issues/26569

  return model, tokenizer

# %% [markdown]
# You need to run the cells below for PennGrader to work.

# %%
model_pretrained, tokenizer_pretrained = load_model_and_tokenizer(pretrained_model_name)

# %%
model_instruction_tuned, tokenizer_instruction_tuned = load_model_and_tokenizer(instruction_tuned_model_name)

# %%
grader.grade(test_case_id = 'grade_model_tokenizer_pretrained', answer = (model_pretrained.name_or_path, (tokenizer_pretrained.name_or_path, tokenizer_pretrained.padding_side)))

# %%
grader.grade(test_case_id = 'grade_model_tokenizer_instruction_tuned', answer = (model_instruction_tuned.name_or_path, (tokenizer_instruction_tuned.name_or_path, tokenizer_instruction_tuned.padding_side)))

# %% [markdown]
# Now you will write functions to run the models on given prompts. Notice that the model will output an answer and then continue generation, especially the pre-trained model  - make sure you postprocess the response.

# %%
@torch.no_grad()
def run_model(prompt,
              model,
              tokenizer,
              apply_chat_template=False,
              generation_kwargs={
                  'min_new_tokens': 1,
                  'max_new_tokens': 25,
                  'temperature': 0.1,
                  'top_p': 0.95,
                  'do_sample': True,
              }
              ):
  set_seed(42)

  if apply_chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=False
        )

  inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
  input_len = inputs.attention_mask.shape[1]
  output_ids = model.generate(**inputs, **generation_kwargs)

  new_token_ids = output_ids[0][input_len:]
  output = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
  # output = output.split("\n")[0].strip()

  return output


@torch.no_grad()
def run_model_batch(prompts,
              model,
              tokenizer,
              apply_chat_template=False,
              generation_kwargs={
                  'min_new_tokens': 1,
                  'max_new_tokens': 25,
                  'temperature': 0.1,
                  'top_p': 0.95,
                  'do_sample': True
              },
              tokenizer_kwargs={
                  'padding':'longest',
              }
            ):
  set_seed(42) # Do not change seed. Needed for grading

  ## TODO: generate text for batch (list of strings) input.
  ## If apply_chat_template is true, then format the prompt accordingly and apply the template.
  ## Return only new tokens.
  ## See here for more information: https://huggingface.co/google/gemma-2b-it
  if apply_chat_template:
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=False
            ) for p in prompts
        ]

  tokenized = tokenizer(prompts, return_tensors="pt", **tokenizer_kwargs).to(model.device)
  input_lens = [len(mask) for mask in tokenized.attention_mask]

  output_ids = model.generate(**tokenized, **generation_kwargs)

  batch_output = []
  for i, inp_len in enumerate(input_lens):
      new_ids = output_ids[i][inp_len:]
      out_text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
      # out_text = out_text.split("\n")[0].strip()

      batch_output.append(out_text)

  return batch_output

# %% [markdown]
# Run the following cells for the PennGrader to work.

# %%
TEST_DECODE_A = run_model('IMLZDHEFOQ', model_pretrained, tokenizer_pretrained, apply_chat_template=False, generation_kwargs={'min_new_tokens': 3, 'max_new_tokens': 20, 'do_sample': False})
TEST_DECODE_B = run_model('IMLZDHEFOQ', model_instruction_tuned, tokenizer_instruction_tuned, apply_chat_template=False, generation_kwargs={'min_new_tokens': 3,  'max_new_tokens': 20,'do_sample': False})
TEST_DECODE_C = run_model('IMLZDHEFOQ', model_instruction_tuned, tokenizer_instruction_tuned, apply_chat_template=True, generation_kwargs={'min_new_tokens': 3, 'max_new_tokens': 20, 'do_sample': False})
TEST_BATCH_DECODE_A = run_model_batch(['B FERE WV NGJ', 'L ZSO FHKNJM', 'JB YQFVPR DK'], model_pretrained, tokenizer_pretrained, apply_chat_template=False, generation_kwargs={'min_new_tokens': 3,  'max_new_tokens': 20,'do_sample': False})
TEST_BATCH_DECODE_B = run_model_batch(['BFE REWV NGJ', 'LZS OFHK NJM', 'JBY QFVPR DK'], model_instruction_tuned, tokenizer_instruction_tuned, apply_chat_template=False, generation_kwargs={'min_new_tokens': 3, 'max_new_tokens': 20, 'do_sample': False})
TEST_BATCH_DECODE_C = run_model_batch(['BFER EWVN GJ', 'LZSO FHKNJ M', 'JBYQ FVP R DK'], model_instruction_tuned, tokenizer_instruction_tuned, apply_chat_template=True, generation_kwargs={'min_new_tokens': 3,  'max_new_tokens': 20,'do_sample': False}, tokenizer_kwargs={'padding':'longest'})

grader.grade(test_case_id = 'grade_run_model', answer = (TEST_DECODE_A, TEST_DECODE_B, TEST_DECODE_C, TEST_BATCH_DECODE_A,TEST_BATCH_DECODE_B, TEST_BATCH_DECODE_C))

# %% [markdown]
# # Section 2: Exploring Prompting (10 points)
# **Background:** Prompting is a way to guide a language model, which is ultimately just a model that predicts the most likely next sequence of words, to complete some arbitrary task you want it to complete. We'll walk through a few examples and then you'll try creating your own prompts.
# 
# A language model will "complete" (just like autocomplete) your prompt with what words are most likely to come next. We demonstrate this is the case by showing how Gemma completes movie quotes, when giving it the beginning of the **quote**:

# %%
input_texts = ["Life is like a box of chocolates,", "With great power,", "The name's Bond.", "Houston, we", "I've a feeling we're not in"]
output_texts = run_model_batch(input_texts, model_pretrained, tokenizer_pretrained)
for input_text, output_text in zip(input_texts, output_texts):
  print(f'Input: {input_text}\nCompletion: {output_text}\n\n')
  print(100*'-')

# %% [markdown]
# Now imagine we give a prompt like this:

# %%
question = "Question: Who was the first president of the United States? Answer:"

print(f"Input: {question}\nCompletion: {run_model(question, model_pretrained, tokenizer_pretrained)}")

# %% [markdown]
# By posing a question and writing "Answer:" at the end, we make it such that the most likely next sequence of words is the answer to the question! This is the key to large language models being able to perform arbitrary tasks, even though they are only trained to predict the next word.
# 
# We can parameterize this prompt and make it reusable for different questions:

# %%
QA_PROMPT = "Question: {input} Answer:"
input_questions = ["What company did Steve Jobs found?", "What's the movie with Tom Cruise about fighter jets?", "What color are bananas?"]
answers = run_model_batch([QA_PROMPT.replace('{input}', q) for q in input_questions], model_pretrained, tokenizer_pretrained)
for input_question, answer in zip(input_questions, answers):
  print(f'Input:{QA_PROMPT.replace("{input}", input_question)}\nCompletion: {answer}\n'+"-"*100)

# %% [markdown]
# Now that you've seen a few examples it's time for you to come up with a few of your own prompts! Make sure you parameterize them with `{input}` before sending the prompt to the autograder. All your prompts should be reuseable when the autograder does `.replace("{input}", ...)` on them.
# 
# Note: These models are not easy to control. Therefore, it's okay if your prompt does not always get the answer right or also spews extra text along with the answer (as long as the answer comes first). Test it out a few times, and if it seems like it works, then you can try it with the autograder.

# %% [markdown]
# - **Problem 2.1:** Write a prompt that returns the capital of country.

# %%
!gdown https://drive.google.com/uc?id=1wfck3vfSDgi7oItmIxEnqMTLKa3B3W-o

# %%
# TODO
CAPITAL_OF_COUNTRY_PROMPT = "The capital of country {input} is"
# PennGrader - DO NOT CHANGE
from functools import partial
countries = [l.strip() for l in open('capitals.txt').readlines()]
model_outputs = []
for batch in range(0, len(countries), 5):
  model_outputs.extend(
      run_model_batch([CAPITAL_OF_COUNTRY_PROMPT.replace('{input}', country) for country in countries[batch:batch+5]],
                      model_pretrained,
                      tokenizer_pretrained,
                      generation_kwargs={'max_new_tokens': 5, 'do_sample': False}
                      )
  )
grader.grade(test_case_id = 'test_capital_of_country', answer = model_outputs)

# %% [markdown]
#  - **Problem 2.2:** Write a prompt that given a famous movie returns the director.

# %%
!gdown https://drive.google.com/uc?id=1Jh8ZJIRW4C3O6t0KJQDLTwsyzcE7HjzO

# %%
# TODO
DIRECTOR_OF_MOVIE_PROMPT = "Question: Who is the director of movie {input}? Answer:"

# PennGrader - DO NOT CHANGE
movies = [l.strip() for l in open('directors.txt').readlines()]
model_outputs=[]
for batch in range(0, len(movies), 5):
  model_outputs.extend(run_model_batch([DIRECTOR_OF_MOVIE_PROMPT.replace('{input}', movie) for movie in movies[batch:batch+5]],
                                       model_pretrained,
                                       tokenizer_pretrained,
                                       generation_kwargs={'max_new_tokens': 5, 'do_sample': False}))
grader.grade(test_case_id = 'test_director_of_movie', answer = model_outputs)

# %% [markdown]
#  - **Problem 2.3:** Write a prompt that given a word, returns a list of synonyms.
# 

# %%
!gdown https://drive.google.com/uc?id=1CpnKz6_FldorSKk15lErgtLU_rL6SvfX

# %%
# TODO
SYNONYMS_OF_WORD_PROMPT = "The synonyms of the word {input} are"

# PennGrader - DO NOT CHANGE
words = [l.strip() for l in open('synonyms.txt').readlines()]
model_outputs = []
for batch in range(0, len(words), 5):
  model_outputs.extend(run_model_batch([SYNONYMS_OF_WORD_PROMPT.replace('{input}', word) for word in words[batch:batch+5]],
                                       model_pretrained,
                                       tokenizer_pretrained,
                                       generation_kwargs={'max_new_tokens': 5, 'do_sample': False}
                                      ))
grader.grade(test_case_id = 'test_synonyms_of_word', answer = model_outputs)

# %% [markdown]
#  - **Problem 2.4:** Write a prompt that given a food item ("cookies"), returns a list of ingredients used to make that food item.

# %%
!gdown https://drive.google.com/uc?id=1q-UPQuFX-NamdUup8A_Tiw6H-nhITLVN

# %%
# TODO
INGREDIENTS_OF_FOOD_PROMPT = "The ingredients of {input} are"

# PennGrader - DO NOT CHANGE
foods = [l.strip() for l in open('ingredients.txt').readlines()]
model_outputs = []
for batch in range(0, len(foods), 5):
  model_outputs.extend(run_model_batch([INGREDIENTS_OF_FOOD_PROMPT.replace('{input}', food) for food in foods[batch:batch+5]],
                                       model_pretrained,
                                       tokenizer_pretrained,
                                       generation_kwargs={'max_new_tokens': 10, 'do_sample': False}
                                      ))

grader.grade(test_case_id = 'test_ingredients_of_food', answer = model_outputs)

# %% [markdown]
# **Problem 2.5:** Write a prompt that given a famous quote ("One small step for man, one giant leap for mankind.", quote characters included), returns the name of the person who said the quote (quotee).
# 
# *Extra Challenge:* We want you to try to complete this one without question marks ("?") or question words ("Who", "What", etc.). You will only get full points if your prompt does not contain those. Hint: Reading, Section 2, may help you with this if you can't figure it out.

# %%
!gdown https://drive.google.com/uc?id=11bCKArwJqsjSYZ-oh_hb8RoW1cFEykPM

# %%
# TODO
QUOTEE_OF_QUOTE_PROMPT = "Quote: {input}  Speaker:"

# PennGrader - DO NOT CHANGE
quotes = [l.strip() for l in open('quotees.txt').readlines()]
model_outputs = []
for batch in range(0, len(quotes), 5):
  model_outputs.extend(run_model_batch([QUOTEE_OF_QUOTE_PROMPT.replace('{input}', quote) for quote in quotes[batch:batch+5]],
                                       model_pretrained,
                                       tokenizer_pretrained,
                                       generation_kwargs={'max_new_tokens': 10, 'do_sample': False}
                                      ))

grader.grade(test_case_id = 'test_quotee_of_quote', answer = (model_outputs, QUOTEE_OF_QUOTE_PROMPT))

# %% [markdown]
# # Section 3: Prompt Engineering (20 points)
# 
# ---
# 
# 
# 
# The prompts you have used up to this point have been fairly basic and straightforward to create. But what if you have a more difficult task and it seems like your prompt isn't working? *Prompt engineering* is the procecss of iterating on a prompt in clever ways to induce the model to produce what you want. The best way of prompt engineering systematically vs. randomly is by understanding how the underlying model was trained and what data it was trained on to best prompt the model.
# 
# Imagine we want the model to generate a quote in Donald Trump's style of talking about a certain topic:

# %%
DONALD_TRUMP_PROMPT = "Question: What would Donald Trump say about {input}? Answer:"
DONALD_TRUMP_PROMPT_ENGINEERED_1 = 'On the topic of {input}, Donald Trump said"'
DONALD_TRUMP_PROMPT_ENGINEERED_2 = 'On the topic of {input}, Donald Trump expressed optimism saying"'
DONALD_TRUMP_PROMPT_ENGINEERED_3 = 'On the topic of {input}, Donald Trump expressed doubt saying"'

# Doesn't work
print(f'Input: {DONALD_TRUMP_PROMPT.replace("{input}", "the stock market")}\nCompletion: {run_model(DONALD_TRUMP_PROMPT.replace("{input}", "the stock market"), model_pretrained, tokenizer_pretrained)}')
print('--'*20)
# Works!
print(f'Input: {DONALD_TRUMP_PROMPT_ENGINEERED_1.replace("{input}", "the stock market")}\nCompletion: {run_model(DONALD_TRUMP_PROMPT_ENGINEERED_1.replace("{input}", "the stock market"), model_pretrained, tokenizer_pretrained)}')
print('--'*20)
# Works!
print(f'Input: {DONALD_TRUMP_PROMPT_ENGINEERED_2.replace("{input}", "the stock market")}\nCompletion: {run_model(DONALD_TRUMP_PROMPT_ENGINEERED_2.replace("{input}", "the stock market"), model_pretrained, tokenizer_pretrained)}')
print('--'*20)
# Works!
print(f'Input: {DONALD_TRUMP_PROMPT_ENGINEERED_3.replace("{input}", "the stock market")}\nCompletion: {run_model(DONALD_TRUMP_PROMPT_ENGINEERED_3.replace("{input}", "the stock market"), model_pretrained, tokenizer_pretrained)}')
print('--'*20)

# %% [markdown]
# The first naive prompt doesn't really work. After prompt engineering, not only do we get a much more realistic generation of his style, but we can also control whether he is talking about the topic positively or negatively.
# 
# **Please respond to the following questions in your `report.pdf`**
# 
# * **Problem 3.1:** Why did the `DONALD_TRUMP_PROMPT_ENGINEERED_1` prompt work much better than the `DONALD_TRUMP_PROMPT` prompt?

# %% [markdown]
# A prompt that is well-engineered can effectively solve difficult NLP tasks that previously were solved by fine-tuning models. In lecture, we showed some examples of these.
# 
# **Problem 3.2:** Write a prompt that will solve the [sentiment classification task](https://en.wikipedia.org/wiki/Sentiment_analysis), and classify [movie reviews](https://ai.stanford.edu/~amaas/data/sentiment/) as *positive* or *negative*. `IMDB_DATASET_X` and `IMDB_DATASET_Y` contain 200 reviews and sentiment labels (1 = positive, 0 = negative). Get as high of an accuracy as you can on these. Place your `MOVIE_SENTIMENT` prompt and `POSITIVE_VEBALIZERS` and `NEGATIVE_VERBALIZERS` in `report.pdf` for manual grading. Along with your `correct` (out of 200) score.

# %%
# TODO
MOVIE_SENTIMENT_PROMPT = "Task: Determine if this movie review is positive or negative. Movie Review: {input}, Answer (Positive/Negative):"

POSITIVE_VERBALIZERS = [
    "good",
    "positive"
]
NEGATIVE_VERBALIZERS = [
    "bad",
    "negative"
]

def map_to_sentiment_label(model_output):
    for v in POSITIVE_VERBALIZERS:
        if v.lower() in model_output[:20].lower():
            # print(model_output[:20].lower())
            return 1
    for v in NEGATIVE_VERBALIZERS:
        if v.lower() in model_output[:20].lower():
            return 0

    print(model_output[:20].lower())
    return None

correct = 0
for review, label in zip(IMDB_DATASET_X, IMDB_DATASET_Y):
    model_output = run_model(MOVIE_SENTIMENT_PROMPT.replace("{input}", review), model_pretrained, tokenizer_pretrained)
    prediction = map_to_sentiment_label(model_output)
    if prediction == label:
        correct += 1
    # print(f"Prediction: {prediction}, Label: {label}")
print(f"Correct: {correct}/200 ", f"Accuracy: {(correct/200)*100}%")

# %% [markdown]
# # Section 4: Few-Shot Prompting (20 points)
# 
# The prompts you have seen up until this point are zero-shot prompts, in that we are asking the model to complete a task without any examples. By providing some examples in the prompt, the model becomes significantly more capable. We'll show an example.
# 
# Consider the task of figuring out a more complex version of a word:

# %%
ZERO_SHOT_COMPLEX_PROMPT = "Question: What is a more complex word for {input}? Answer:"
FEW_SHOT_COMPLEX_PROMPT = "angry: indignant\nsad: sorrowful\n{input}:"

# Doesn't work
print(f'Input: {ZERO_SHOT_COMPLEX_PROMPT.replace("{input}", "confused")}\nCompletion: {run_model(ZERO_SHOT_COMPLEX_PROMPT.replace("{input}", "confused"), model_pretrained, tokenizer_pretrained)}')
print(20*'-')
# Works!
print(f'Input: {FEW_SHOT_COMPLEX_PROMPT.replace("{input}", "confused")}\nCompletion: {run_model(FEW_SHOT_COMPLEX_PROMPT.replace("{input}", "confused"), model_pretrained, tokenizer_pretrained)}')

# %% [markdown]
# The first zero-shot prompt where we have no example doesn't work at all, where as when we give 2 examples in the few-shot prompt (2-shot prompt), it works.
# 
# Now that you've seen an example of few-shot prompting, it's your turn to try it.
# 
# **Problem 4.1:** Write a few-shot prompt that converts an input into a [Jeopardy! style answer](https://en.wikipedia.org/wiki/Jeopardy!#:~:text=Rather%20than%20being%20given%20questions,the%20form%20of%20a%20question.) (The Great Lakes -> "What are the Great Lakes?" or Taylor Swift -> "Who is Taylor Swift?")
# 

# %%
!gdown https://drive.google.com/uc?id=1NAWQdvImD7L8WJMlHhcuGpApTBcpklmb

# %%
# TODO
TO_JEOPARDY_ANSWER_PROMPT = """Convert the following trivia answers into Jeopardy-style questions.

Answer: Huron, Ontario, Michigan, Erie, and Superior
Question: What are the Great Lakes?

Answer: Taylor Swift
Question: Who is Taylor Swift?

Answer: {input}
Question:"""

# PennGrader - DO NOT CHANGE
words = [l.strip() for l in open('jeopardy.txt').readlines()]
model_outputs =  []
for batch in range(0, len(words), 5):
  model_outputs.extend(
      run_model_batch([TO_JEOPARDY_ANSWER_PROMPT.replace('{input}', word) for word in words[batch:batch+5]],
                      model_pretrained,
                      tokenizer_pretrained,
                      generation_kwargs={'max_new_tokens': 10, 'do_sample': False}
                      )
  )


grader.grade(test_case_id = 'test_to_jeopardy_answer', answer = model_outputs)

# %% [markdown]
# **Problem 4.2:** Write a few-shot prompt that translates a Korean word to an English word.

# %%
!gdown https://drive.google.com/uc?id=1QtvLigJfQdj2mqo2KT7kyOzURnrVb7L8

# %%
# TODO
KOREAN_TO_ENGLISH_PROMPT = """Translate the following Korean words to English.

책상: desk
고양이: cat
학교: school
사과: apple
{input}:"""

# PennGrader - DO NOT CHANGE
korean_words = [l.strip() for l in open('korean_to_english.txt').readlines()]
model_outputs = []
for batch in range(0, len(korean_words), 5):
  model_outputs.extend(
      run_model_batch([KOREAN_TO_ENGLISH_PROMPT.replace('{input}', word) for word in korean_words[batch:batch+5]],
                      model_pretrained,
                      tokenizer_pretrained,
                      generation_kwargs={'max_new_tokens': 10, 'do_sample': False}
                      )
  )
grader.grade(test_case_id = 'test_korean_to_english', answer = model_outputs)

# %% [markdown]
# **Please respond to the following question in your `report.pdf`**
# 
# **Problem 4.3:** Come up with 3 more arbitrary tasks, where a zero-shot prompt might not suffice, and a few-shot prompt would be required. Provide a short write up describing what your tasks are. Provide examples of a zero-prompt not working for it. Then, show us your few-shot prompt and some results. Be creative and try to pick 3 tasks that are somewhat distinct from each other!

# %% [markdown]
# # Section 5: Prompting Instruction-Tuned Models (15 points)
# 
# Large language models can be *instruction-tuned*, fine-tuned with examples of instructions and responses to those instructions, to make them easier to prompt and friendlier to humans. Instruction-tuned models can more easily be given natural langauge instructions describing a task you want them to complete. This makes it so that they are more performant without requiring as much prompt engineering and makes them more likely to succeed with just zero-shot prompting. The version of Gemma we were working with in previous exercises was not instruction-tuned, we now will use instruction-tuned models from here on out:

# %%
KOREAN_TO_ENGLISH_INSTRUCTION_PROMPT = "Translate the Korean word \"{input}\" to English."

# Doesn't work on non-instruction tuned model
print(f'Input: {KOREAN_TO_ENGLISH_INSTRUCTION_PROMPT.replace("{input}", "책상")}\nCompletion: {run_model(KOREAN_TO_ENGLISH_INSTRUCTION_PROMPT.replace("{input}", "책상"),  model_pretrained, tokenizer_pretrained)}')
print(20*'-')
# Works and is simpler!
print(f'Input: {KOREAN_TO_ENGLISH_INSTRUCTION_PROMPT.replace("{input}", "책상")}\nCompletion: {run_model(KOREAN_TO_ENGLISH_INSTRUCTION_PROMPT.replace("{input}", "책상"),model_instruction_tuned, tokenizer_instruction_tuned,  apply_chat_template=True)}')

# %% [markdown]
# As you can see, these instruction-tuned models make it much simpler to complete complex tasks since you can "talk" to them naturally. We'll now ask you to try.
# 
# **Problem 5.1:** Write a prompt that returns the Spanish word given an English word (painting -> pintura).
# 
# *Extra Challenge:* We want you to complete this one such that the model only returns a single Spanish word and nothing else. You will only get points if your model only returns a single Spanish word and nothing else.

# %%
!gdown https://drive.google.com/uc?id=1-sHAmzLnEcYgpewaz3O_iLPXJb1sA7Yp

# %%
# TODO
ENGLISH_TO_SPANISH_PROMPT = """You are a translator from English to Spanish. Translate the provided English word/sentence into a single Spanish word. Answer with only a single Spanish word and nothing else. For example, cat -> gato
house -> casa
painting -> pintura

hello -> hola
thank -> gracias
love -> amor
sorry -> perdón
congratulate -> felicitar
well done -> bravo
it's okay -> bien
nice to meet you -> encantado
weather -> clima
cheer up -> ánimo
delicious -> delicioso
where -> dónde
be careful -> cuidado
sleep -> dormir
time -> tiempo
help -> ayuda
price -> precio
thing -> cosa
who -> quién
name -> nombre
american -> estadounidense
student -> estudiante
hunger -> hambre
thirst -> sed
tiredness -> cansancio
joy -> alegría
sadness -> tristeza
annoyance -> molestia
worry -> preocupación
fear -> miedo
good -> bueno
bad -> malo
fast -> rápido
slow -> lento
meeting -> reunión
tomorrow -> mañana
next -> próximo
don't know -> desconocer
know -> saber
miss -> extrañar
waiting -> esperar
depart -> salir
return -> volver
morning -> mañana
evening -> tarde
goodbye -> adiós
again -> nuevamente
here -> aquí
there -> allí
now -> ahora
a little -> poco
arrival -> llegada
action -> acción
thought -> pensamiento
coffee -> café
cake -> pastel
home -> hogar

Therefore, the single word translation of "{input}" -> """


# PennGrader - DO NOT CHANGE
words = [line.strip() for line in open('english_to_spanish.txt').readlines()]
# print(words)
model_outputs = []
for batch in range(0, len(words), 5):
  model_outputs.extend(
      run_model_batch([ENGLISH_TO_SPANISH_PROMPT.replace('{input}', word) for word in words[batch:batch+5]],
                      model_instruction_tuned,
                      tokenizer_instruction_tuned,
                      generation_kwargs={'max_new_tokens': 10, 'do_sample': False}
                      )
  )
# print(model_outputs)
grader.grade(test_case_id = 'test_english_to_spanish', answer = model_outputs)

# %% [markdown]
# **Please respond to the following question in your `report.pdf`**
# 
# **Problem 5.2:** Come up with 3 more arbitrary tasks, where the non-instruction-tuned model might not suffice, and an instruction-tuned model would be required. Provide a short write up describing what your tasks are. Provide examples of a prompt not working on a non-instruction-tuned model. Then, show us your instruction prompt on an instruction-tuned model and some results. Be creative and try to pick 3 tasks that are somewhat distinct from each other!

# %% [markdown]
# # Section 6: Chain-of-Thought Reasoning (30 points)
# 
# One recent method to prompt large language models is Chain-of-Thought Prompting. This is similar to few-shot prompting, except you not only provide a few examples, but you also provide an explanation with a reasoning chain to the model. Providing this reasoning chain as been shown to improve performance on a wide variety of tasks.
# 
# We demonstrate on a task that consists of 2 arithmetic operations over 3 single digit numbers:

# %%
FEW_SHOT_ARITHMETIC_PROMPT = "2 * 4 + 2? 10\n6 + 7 - 2? 11\n{input}?"
COT_ARITHMETIC_PROMPT = "2 * 4 + 2? 2 * 4 = 8. 8 + 2 = 10\n6 + 7 - 2? 6 + 7 = 13. 13 - 2 = 11\n{input}?"

 # Doesn't work without CoT prompting
output = run_model(FEW_SHOT_ARITHMETIC_PROMPT.replace("{input}", "20 + 10 - 5"),
                   model_instruction_tuned,
                   tokenizer_instruction_tuned)
print(20*'-')
print(f'Input: {FEW_SHOT_ARITHMETIC_PROMPT.replace("{input}", "20 + 10 - 5")}\nCompletion: {output}')
print(20*'-')
# Works with CoT prompting
output = run_model(COT_ARITHMETIC_PROMPT.replace("{input}", "20 + 10 - 5"),
                   model_instruction_tuned,
                   tokenizer_instruction_tuned,
                   apply_chat_template=True,
                   generation_kwargs={
                       "max_new_tokens": 25,
                       "temperature": 0.1,
                       "top_p": 0.95,
                       "do_sample": True
                       }
                   )
print(f'Input: {COT_ARITHMETIC_PROMPT.replace("{input}", "20 + 10 - 5")}\n'+'Completion: '+ output)
print(20*'-')


# %% [markdown]
# Next, we create a dataset with 50 examples:

# %%
import random
import re

def compute(x, operand, y):
    if operand == '+':
        return x + y
    elif operand == '-':
        return x - y
    elif operand == '*':
        return x * y

def create_arithmetic_dataset(n_examples, seed = 42):
    random.seed(seed)
    X = []
    y = []
    for i in range(n_examples):
        num_1 = random.randint(0,9)
        operator_1 = random.choice(['+', '-', '*'])
        num_2 = random.randint(0,9)
        operator_2 = random.choice(['+', '-', '*'])
        num_3 = random.randint(0,9)
        if operator_2 == '*' and operator_1 != '*':
            # Order of operations:
            # Do the right-hand side first
            intermediate = compute(num_2, operator_2, num_3)
            final = compute(num_1, operator_1, intermediate)
        else:
            intermediate = compute(num_1, operator_1, num_2)
            final = compute(intermediate, operator_2, num_3)
        X.append(f'{num_1} {operator_1} {num_2} {operator_2} {num_3}')
        y.append(final)
    return X, y

def parse_answer(model_output):
    '''Parses the output of the model to get the final answer.'''
    try:
        # Gets the last number of the string in the first line using regex and returns that
        return int(re.search(r'(\d+)(?!.*\d)', model_output.strip().split('\n')[0])[0])
    except TypeError:
        return None

arithmetic_X, arithmetic_y = create_arithmetic_dataset(50)

# %% [markdown]
# **Please respond to the following questions in your `report.pdf`**
# 
# **Problem 6.1:** Your job is to investigate how few-shot Chain-of-Thought prompting performs vs. regular few-shot prompting over the entire arithmetic dataset and grade how many out of 50 are correct. Perform this experiment 6 times each with a different number of regular few-shot examples (1 example, 2 examples, 4 examples, 8 examples, 16 examples, 32 examples) and 6 times again each with a different number of Chain-of-Thought few-shot examples (1 CoT example, 2 CoT examples, 4 CoT examples, 8 CoT examples, 16 CoT examples, 32 CoT examples).
# 
# Create a table or plot of (N examples) vs. (% questions correct by the model with a few-shot prompt with N examples) vs. (% questions correct by the model with a CoT prompt with N examples). Report this table or plot in `report.pdf` with a short write-up about your observations. Keep the code used to build your table or plot in your notebook for inspection during grading.
# 
# *Note:* Make sure you use the **instruction tuned model**.
# 
# *Hint:* You might find the `parse_answer` function helpful when grading how many of the model's outputs are correct or not.

# %%
# TODO - Solve Problem 5.1 here
import math

def build_arithmetic_prompt(shot_X, shot_y, query_expr, use_cot=False):
    """
    构造 few-shot 或 few-shot+CoT 的算术 prompt。
    shot_X / shot_y: few-shot 示例
    query_expr: 当前要预测的表达式 (形如 '3 + 4 * 2')
    """
    header = (
        "You are a calculator. "
        "Given some examples of arithmetic problems and their answers, "
        "answer the last question. \n\n"
    )

    lines = []
    for expr, ans in zip(shot_X, shot_y):
        n1, op1, n2, op2, n3 = expr.split()
        n1, n2, n3 = int(n1), int(n2), int(n3)
        if use_cot:
            # 按 create_arithmetic_dataset 的逻辑生成中间步骤
            if op2 == '*' and op1 != '*':
                inter = compute(n2, op2, n3)
                steps = f"{n2} {op2} {n3} = {inter}. {n1} {op1} {inter} = {ans}"
            else:
                inter = compute(n1, op1, n2)
                steps = f"{n1} {op1} {n2} = {inter}. {inter} {op2} {n3} = {ans}"
            lines.append(f"{expr}? {steps}")
        else:
            lines.append(f"{expr}? {ans}")
    # 查询样本：不给出步骤，留给模型去类比推理
    lines.append(f"{query_expr}?")
    return header + "\n".join(lines)


def evaluate_arithmetic(n_shots, use_cot=False):
    """
    用 n_shots 个 few-shot 示例，在 50 个算术样本上评估准确率。
    """
    shot_X = arithmetic_X[:n_shots]
    shot_y = arithmetic_y[:n_shots]

    prompts = [
        build_arithmetic_prompt(shot_X, shot_y, expr, use_cot=use_cot)
        for expr in arithmetic_X
    ]
    # print('use CoT' if use_cot else 'don\'t use plain')
    # print(prompts)

    outputs = run_model_batch(
        prompts,
        model_instruction_tuned,
        tokenizer_instruction_tuned,
        apply_chat_template=True,
        generation_kwargs={
            "min_new_tokens": 1,
            "max_new_tokens": 25,
            "temperature": 0.1,
            "top_p": 0.95,
            "do_sample": False,   # 评估时关闭采样，结果更稳定
        },
    )

    preds = [parse_answer(o) for o in outputs]
    correct = sum(int(p == gt) for p, gt in zip(preds, arithmetic_y))
    return correct / len(arithmetic_y)


shot_sizes = [1, 2, 4, 8, 16, 32]
plain_accs = []
cot_accs = []

for n in shot_sizes:
    acc_plain = evaluate_arithmetic(n_shots=n, use_cot=False)
    acc_cot = evaluate_arithmetic(n_shots=n, use_cot=True)
    plain_accs.append(acc_plain)
    cot_accs.append(acc_cot)
    print(f"N = {n:2d} | plain = {acc_plain:.2f}, CoT = {acc_cot:.2f}")



# %% [markdown]
# ### Figure

# %%
import matplotlib.pyplot as plt

plt.figure()
plt.plot(shot_sizes, plain_accs, marker="o", label="Few-shot")
plt.plot(shot_sizes, cot_accs, marker="o", label="Few-shot + CoT")
plt.xlabel("Number of examples in prompt (N)")
plt.ylabel("Accuracy on 50 problems")
plt.xscale("log", base=2)
plt.ylim(0, 1.05)
plt.legend()
plt.title("Few-shot vs. Chain-of-Thought on 2-op arithmetic")
plt.show()


# %% [markdown]
# # Submissions

# %% [markdown]
# ## Free-response Checklist (check if you missed anything!)
# We will look for the following free-responses in this notebook:
# - Section 2: Question responses
# - Section 3: Question response
# - Section 4: Question response
# - Section 6: Table/Plot and short write-up

# %% [markdown]
# 

# %% [markdown]
# ## GradeScope File Submission
# Here are the deliverables you need to submit to GradeScope:
# - Write-up (`report.pdf`):
#     - Section 2: Question responses
#     - Section 3: Question response
#     - Section 4: Question response
#     - Section 5: Question response
#     - Section 6: Table/Plot and short write-up
# - Code:
#     - This notebook and py file: rename to `homework5.ipynb` and `homework5.py`. You can download the notebook and py file by going to the top-left corner of this webpage, `File -> Download -> Download .ipynb/.py`

# %%




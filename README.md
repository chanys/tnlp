
# TNLP: Transformer for Natural Language Processing

This is the code repo for the TNLP project, by [Yee Seng Chan](https://chanys.github.io/about/). 

This repo contains code to perform the following NLP tasks:
* Sequence classification: assigns a label to a text.
* Token classification: assigns a label to individual tokens in a sentence.
* Span pair classification: assigns a label to a pair of text spans.
* Sequence to sequence: generation of output text given input text.
* Contrastive modeling: differentiate between positive and negative pairs of examples.
* Instruction fine-tuning: fine-tuning of a pretrained generative LLM on instruction data.
* Chat fine-tuning: following Zephyr-7B, the code here fine-tunes on a turn-base dialog dataset between "user" and "assistant".
* Direct preference optimization (DPO): following Zephyr-7B, the code here fine-tunes on a preference dataset containing "chosen" and "rejected" messages.
* End-to-end retrieval augmented generation (RAG): joint training of a retriever model and a generator model.
By feeding in documents from different domains, this allows for domain adaptation of the generative LLM.

## Table of Contents

- [Tasks Summary](#tasks-summary)
- [Repo Structure](#repo-structure)
- [Relevant Blog Articles](#relevant-blog-articles)
- [Tasks](#tasks)
  - [Sequence Classification](#sequence-classification)
  - [Sequence Classification (customized)](#sequence-classification-customized)
  - [Token Classification](#token-classification)
  - [SpanPair Classification](#spanpair-classification)
  - [Sequence to Sequence](#sequence-to-sequence)
  - [Contrastive Modeling (using Sentence Transformers)](#contrastive-modeling-using-sentence-transformers)
  - [Contrastive Modeling (using custom classes)](#contrastive-modeling-using-custom-classes)
  - [Instruction Fine-tuning](#instruction-fine-tuning)
  - [Chat Fine-tuning](#chat-fine-tuning)
  - [Direct Preference Optimization (DPO)](#direct-preference-optimization-dpo)
  - [Retrieval Augmented Generation (RAG)](#retrieval-augmented-generation-rag)


## Tasks Summary

The following Table summarizes the classes and example use cases for each task.

| Task                           | APIs and Classes                                                                                                                                                                                                                                                  | Example Use Case in Codebase                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
|:-------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Sequence classification        | - [AutoModelForSequenceClassification](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForSequenceClassification) for modeling<br/>- [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer#trainer) for training  | Emotion classification.<br/>- Using [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased) as encoder to classify Twitter messages for emotion.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| Sequence classification        | - Custom class `CustomSequenceModel` for modeling<br/>- Custom training code<br/>- Custom class `SpanDataset` to manage examples                                                                                                                                  | Restaurant category classification.<br/>- Using [bert-base-uncased](https://huggingface.co/bert-base-uncased) as encoder to classify TripAdvisor restaurant reviews as "business", "family", or "romantic".                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| Token classification           | - [AutoModelForTokenClassification](https://huggingface.co/transformers/v3.0.2/model_doc/auto.html#automodelfortokenclassification) for modeling<br/>- [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer#trainer) for training              | Named entity extraction.<br/>- Using [microsoft/deberta-v3-base](https://huggingface.co/microsoft/deberta-v3-base) as encoder to identify named entities in CoNLL-2003 data.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| Span pair classification       | - Custom class `CustomSpanPairModel` for modeling<br/>- Custom training code                                                                                                                                                                                      | Relation extraction.<br/>- Using [microsoft/deberta-v3-base](https://huggingface.co/microsoft/deberta-v3-base) as encoder for the custom span-pair model, to perform relation extraction.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| Sequence to sequence           | - [AutoModelForSeq2SeqLM](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForSeq2SeqLM) for modeling<br/>- [Seq2SeqTrainer](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainer) for training | Summarization.<br/>- Using [google/flan-t5-base](https://huggingface.co/google/flan-t5-base) as the LM to generate summary for dialogue conversations.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| Contrastive modeling           | - [SentenceTransformer](https://www.sbert.net/index.html) for modeling                                                                                                                                                                                            | Query to answer-snippet-candidate similarity using sentence-transformers.<br/>- Train [intfloat/e5-base-v2](https://huggingface.co/intfloat/e5-base-v2) with [TripletLoss](https://www.sbert.net/docs/package_reference/losses.html#tripletloss) on English biomedical queries, associated (chosen) answers, and (rejected) candidate snippets from PubMed.                                                                                                                                                                                                                                                                                                                                              |
| Contrastive modeling           | - Custom class `CustomTripletContrastiveModel` for modeling which includes an encoder, linear layers and a custom loss based on cosine distance<br/>-Custom class `TripletDataset` to manage examples                                                             | Query to answer-snippet-candidate similarity using custom NN.<br/>- Using [microsoft/deberta-v3-base](https://huggingface.co/microsoft/deberta-v3-base) as encoder. Experiment uses English biomedical queries, associated (chosen) answers, and (rejected) candidate snippets from PubMed.                                                                                                                                                                                                                                                                                                                                                                                                              |
| Instruction fine-tuning        | - [AutoModelForCausalLM](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForCausalLM) for modeling<br/>- [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer#trainer) for training                              | Instruction fine-tuning of pretrained LM.<br/>- Performs instruction fine-tuning of [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) with LoRA on the [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) dataset.                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| Chat fine-tuning               | - [SFTTrainer](https://huggingface.co/docs/trl/sft_trainer) for training                                                                                                                                                                                          | Chat fine-tuning of pretrained LM.<br/>- Perform chat fine-tuning of [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) with 4-bit quantization on the [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) dataset                                                                                                                                                                                                                                                                                                                                                                                                                   |
| Direct preference optimization | - [DPOTrainer](https://huggingface.co/docs/trl/main/en/dpo_trainer) for training                                                                                                                                                                                  | Direct preference optimization (DPO) for alignment to human preferences.<br/>- Takes the above chat-fine-tuned model and performs DPO training on the [HuggingFaceH4/ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) dataset                                                                                                                                                                                                                                                                                                                                                                                                                             |
| End-to-end RAG       | - Joint training of retriever and generator<br/>- Custom training code<br/>- Custom class `RagDataset` to manage examples                                                                                                                                         | Joint training of retriever and generator for end-to-end RAG.<br/>- [AutoModel](https://huggingface.co/docs/transformers/v4.36.1/en/model_doc/auto#transformers.AutoModel) using [intfloat/e5-large-v2](https://huggingface.co/intfloat/e5-large-v2) as retriever.<br/>- [AutoModelForCausalLM](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForCausalLM) using [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) as generator.<br/>- Using SQUAD-v2 as training data:<br/>&nbsp;&nbsp;&nbsp;- (query, candidate passage) pairs for training retriever.<br/>&nbsp;&nbsp;&nbsp;- generator's task is to predict answer given (query, passage). |


## Repo Structure

The main packages and directories in this repository are outlined below:
* `src.model`: This directory encompasses the modeling code for various model types, each residing in its own sub-directory.
* `src.task`: To demonstrate the codebase's applicability to different tasks, each task is accompanied by one or more use cases. 
The `src.task` package contains `.yaml` files for each task and its corresponding example use case. 
For instance, the "emotion" dataset is employed to exemplify the "sequence classification" task, 
resulting in the `emotion.(train|test|inference).yaml` files within the `src.task.sequence_model` directory.
* `src.data`: Custom classes, which are subclasses of `torch.utils.data.Dataset`, are implemented for certain tasks within this package.
* `src.scripts`: The main entry point for training, testing, and running inference for different tasks is the `run_model.py` script.
Additionally, the `run_convert_dataset.py` script is employed to transform Hugging Face datasets or raw data files
into the formats used in the `data` directory of this repository.
* `data`: This repository extensively utilizes datasets from Hugging Face, as well as a few external datasets. 
Some datasets may include fields unnecessary for building ML models, or they might be overly large. 
For the sake of storage efficiency and illustration, the `run_convert_dataset.py` script is utilized to extract 
the essential information into `.jsonl` files within the data directory.

## Relevant Blog Articles

This repo leverages multiple models (e.g. BERT, Deberta, T5, LLaMA), sentence transformers, 
efficiency techniques (PEFT, LoRA, qLoRA), and modeling approaches (e.g. RAG, DPO), etc. 
For more information, refer to the following blog articles:

* Transformer: https://chanys.github.io/transformer-architecture
* Sentence transformer: https://chanys.github.io/sbert
* Models:
  * BERT: https://chanys.github.io/bert
    * DistilBERT: https://chanys.github.io/knowledge-distillation
  * Deberta:
    * https://chanys.github.io/deberta
    * https://chanys.github.io/deberta-v3
  * T5: https://chanys.github.io/t5
    * FLAN: https://chanys.github.io/flan-palm
  * LLaMA-2 https://chanys.github.io/llama2
  * Mistral https://chanys.github.io/mistral
  * Zephyr https://chanys.github.io/zephyr
* Parameter Efficient Fine Tuning:
  * PEFT: https://chanys.github.io/peft
  * LoRA: https://chanys.github.io/lora
* Quantization and qLoRA: https://chanys.github.io/qlora
* Retrieval-Augmented Generation (RAG): https://chanys.github.io/rag/
  * End-to-end RAG https://chanys.github.io/rag-domain-qa/
* Direct Preference Optimization (DPO): https://chanys.github.io/dpo/


## Tasks

This section provides more details for each task.

### Sequence Classification

Example use case: 
* Given an English Twitter message, classify it as: `anger`, `fear`, `joy`, `love`, `sadness`, or `surprise`.
* The data is the [emotion](https://huggingface.co/datasets/dair-ai/emotion) dataset from Hugging Face. 

Modeling:
* Classification model: [AutoModelForSequenceClassification](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForSequenceClassification) that uses [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased) as the encoder.
* Trainer: [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer#trainer)

Commands for train, test, inference:
```
python src/scripts/run_model.py --params src/task/sequence_model/emotion.train.yaml --mode train
python src/scripts/run_model.py --params src/task/sequence_model/emotion.test.yaml --mode test
python src/scripts/run_model.py --params src/task/sequence_model/emotion.inference.yaml --mode inference
```


### Sequence Classification (customized)

Sometimes we might want to use our own custom class that subclasses `Dataset`,
or our own training code instead of using [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer#trainer). The following python scripts illustrate this:
* `src.model.sequence_model.custom_sequence_classification.py`
* `src.model.sequence_model.custom_sequence_model.py`
* `src.data.span_dataset.py`

Example use case:
* Classify a restaurant review text into one of the following categories: `business`, `family`, or `romantic`. 
To obtain the necessary data, I initially identified the top 10 restaurants on TripAdvisor within each category of "business," "family," and "romantic."
Subsequently, I collected 15 reviews for each restaurant.

Modeling:
* Classification model: a custom `CustomSequenceModel` class which internally uses [AutoModelForSequenceClassification](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForSequenceClassification) as encoder.
* Trainer: custom training code in `custom_sequence_model.py`.
* Data class: a custom `SpanDataset` class defined in `span_dataset.py` that subclasses [Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset).

Commands for train, test: 
```
python src/scripts/run_model.py --params src/task/sequence_model/restaurant.train.yaml --mode train
python src/scripts/run_model.py --params src/task/sequence_model/restaurant.test.yaml --mode test
```


### Token Classification

Example use case:
* Named entity recognition (NER). 
Given a sequence of word tokens, identify four types of named entities: persons, locations, organizations and names of miscellaneous entities that do not belong to the previous three groups.
* The data is the [conll2003](https://huggingface.co/datasets/conll2003) dataset.

Modeling:
* Classification model: [AutoModelForTokenClassification](https://huggingface.co/transformers/v3.0.2/model_doc/auto.html#automodelfortokenclassification) that uses [microsoft/deberta-v3-base](https://huggingface.co/microsoft/deberta-v3-base) as the encoder.
* Trainer: [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer#trainer)

Commands for train, test, inference:
```
python src/scripts/run_model.py --params src/task/token_model/conll2003.train.yaml --mode train
python src/scripts/run_model.py --params src/task/token_model/conll2003.test.yaml --mode test
python src/scripts/run_model.py --params src/task/token_model/conll2003.inference.yaml --mode inference
```


### SpanPair Classification

Given a pair of text spans, I want to average their embeddings, then put through linear classification layers.
The following custom code achieves this:
* `src.model.spanpair_model.custom_spanpair_classification.py`
* `src.model.spanpair_model.custom_spanpair_model.py`

Example use case:
* Relation extraction.
For instance, given the sentence "Aragaki Yui is an Japanese actress" and the two text spans "Aragaki Yui" and "Japanese", 
identify a `nationality` relation label.
* The data used is the [NYT-H dataset](https://github.com/Spico197/NYT-H) from the paper 
"Towards Accurate and Consistent Evaluation: A Dataset for Distantly-Supervised Relation Extraction" by Zhu et. al. (2020).

Modeling:
* Classification model: a custom `CustomSpanPairModel` defined in `custom_spanpair_model.py`.
Internally, we used [AutoModel](https://huggingface.co/transformers/v3.0.2/model_doc/auto.html#automodel) to encode a piece of text.
When given two predetermined spans within the given text, we first extract their associated span embeddings. 
If a span consists of multiple subword tokens, we take the mean of the subword embeddings.
Then, the two span embeddings are aggregated together and put through linear layers for classification.
* Trainer: custom training code in `custom_spanpair_classification.py`.

Commands for train, test:
```
python src/scripts/run_model.py --params src/task/spanpair_model/nyth.train.yaml --mode train
python src/scripts/run_model.py --params src/task/spanpair_model/nyth.test.yaml --mode test
```


### Sequence to Sequence

Example use case:
* Given a dialogue conversation, generate the summary.
* The data is the [samsum](https://huggingface.co/datasets/samsum) dataset, 
described in the paper "SAMSum Corpus: A Human-annotated Dialogue Dataset for Abstractive Summarization" by Gliwa et. al. (2019).

Modeling:
* Seq2seq model: [AutoModelForSeq2SeqLM](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForSeq2SeqLM),
where we used [google/flan-t5-base](https://huggingface.co/google/flan-t5-base)
* Trainer: [Seq2SeqTrainer](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainer)
* Efficiency: the code loads T5 in 8-bits and also leveraged LoRA from [PEFT](https://github.com/huggingface/peft).

Commands for train, test, inference:
```
python src/scripts/run_model.py --params src/task/seq2seq_model/summarization.train.yaml --mode train
python src/scripts/run_model.py --params src/task/seq2seq_model/summarization.test.yaml --mode test
python src/scripts/run_model.py --params src/task/seq2seq_model/summarization.inference.yaml --mode inference
```


### Contrastive Modeling (using Sentence Transformers)

Contrastive modeling involves training a model to distinguish between positive pairs and negative pairs in the input data.
For instance, given a query, we might want to identify the correct response text among a pool of candidates.
Another example use case is entity linking, i.e. correctly link a text span to its knowledge base entry.

Example use case:
* Data from [BioASQ11](http://participants-area.bioasq.org/Tasks/11b/trainingDataset/), (registration is required to download the data).
* The data consists of English biomedical questions, an associated set of candidate snippets from PubMed articles, 
and a single "ideal answer".
We use the question, "ideal answer", and candidate snippets as `query`, `chosen`, and `rejected` respectively.

Modeling:
* Model: using [SentenceTransformer](https://www.sbert.net/index.html) with [TripletLoss](https://www.sbert.net/docs/package_reference/losses.html#tripletloss).
Given a triplet of (anchor, positive, negative), the loss is defined as
`loss = max(||anchor - positive|| - ||anchor - negative|| + margin, 0)`,  
which aims to minimize the distance between anchor and positive while maximizing the distance between anchor and negative.
Using the BioASQ data, we define `query`, `chosen`, and `rejected` as the `anchor`, `positive`, and `negative` respectively.

Commands for train and test:
```
python src/scripts/run_model.py --params src/task/contrastive_model/bioasq.sentence.train.yaml --mode train
python src/scripts/run_model.py --params src/task/contrastive_model/bioasq.sentence.test.yaml --mode test
```


### Contrastive Modeling (using custom classes)

Instead of using Sentence Transformers, we might want to create custom code to perform contrastive modeling.
The following custom code achieves this:
* `src.model.contrastive_model.custom_triplet_contrastive_learning.py`
* `src.model.contrastive_model.custom_triplet_contrastive_model.py`

I also coded a custom `CustomTripletContrastiveModel` class. 

Example use case (same as the "Contrastive Modeling (using Sentence Transformers)" Section above):
* Data from [BioASQ11](http://participants-area.bioasq.org/Tasks/11b/trainingDataset/), (registration is required to download the data).
* The data consists of English biomedical questions, an associated set of candidate snippets from PubMed articles, 
and a single "ideal answer".
We use the question, "ideal answer", and candidate snippets as `query`, `chosen`, and `rejected` respectively.

Modeling:
* Model: a custom `CustomTripletContrastiveModel` defined in `custom_triplet_contrastive_model.py`.
Internally, this uses [AutoModel](https://huggingface.co/transformers/v3.0.2/model_doc/auto.html#automodel)
to encode the `anchor`, `chosen`, and `rejected`. I then take their first subword embedding (`[CLS]` token), 
put them through a fully connected linear layer, then use `1 - cosine_similarity` as the distance metric
between the (`anchor`, `chosen`) and between the (`anchor`, `rejected`).
* Trainer: custom training code in `custom_triplet_contrastive_learning.py`.
* Data class: a custom `TripletDataset` class in `triplet_dataset.py` that subclasses [Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset).

Command for train:
```
python src/scripts/run_model.py --params src/task/contrastive_model/bioasq.triplet.train.yaml --mode train
```


### Instruction Fine-tuning

Example use case:
* Performs instruction fine-tuning of [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) with LoRA on the [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) dataset.
* Data: From the original [52K instruction-following data](https://github.com/tatsu-lab/stanford_alpaca#data-release) released by the Stanford Alpaca team,
I extracted the first 5K data to use as samples for this code base. 
Each datapoint consists of 3 fields: `instruction`, `input` (optional), `output`. 
Some example instruction following datapoints are:
    ```
    "instruction": "Give three tips for staying healthy.",
    "input": "",
    "output": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly ..."
            
    "instruction": "Generate a song using the following context and melody.",
    "input": "Context: A love song\nMelody:",
    "output": "Verse 1\nWhen I saw you in the room\nMy heart was ..."
    ```

Modeling:
* Model: The class `InstructionFineTuneLM` defined in `src.model.causal_model.instruction_ft.py`
internally uses [AutoModelForCausalLM](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForCausalLM) and uses [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf).
* Efficiency: the code loads the LLM in 8-bits and also leveraged LoRA from [PEFT](https://github.com/huggingface/peft).

Command for train, inference:
```
python src/scripts/run_model.py --params src/task/causal_model/instruction_ft.train.yaml --mode train
python src/scripts/run_model.py --params src/task/causal_model/instruction_ft.inference.yaml --mode inference
```


### Chat Fine-tuning

Example use case:
* Fine-tunes a turn-base dialog dataset that consists of interactions between "user" and "assistant".
* Perform chat fine-tuning of [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) with 4-bit quantization.
* Dataset used for fine-tuning is the [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k).

Modeling:
* The class `ChatFineTuneLM` defined in `src.model.causal_model.chat_ft.py`.
* Trainer: [SFTTrainer](https://huggingface.co/docs/trl/sft_trainer) for training
* Efficiency: uses qLoRA.

Command for train, inference:
```
python src/scripts/run_model.py --params src/task/causal_model/ultrachat.train.yaml --mode train
python src/scripts/run_model.py --params src/task/causal_model/ultrachat.inference.yaml --mode inference
```

### Direct Preference Optimization (DPO)

Example use case:
* Takes the above chat-fine-tuned model and performs DPO training for alingment to human preferences.
* Dataset used is the [HuggingFaceH4/ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized).

Modeling:
* The class `DPOFineTuneLM` defined in `src.model.causal_model.dpo_ft.py`.
* Trainer: [DPOTrainer](https://huggingface.co/docs/trl/main/en/dpo_trainer) for training
* Efficiency: uses qLoRA.

Command for train, inference:
```
python src/scripts/run_model.py --params src/task/causal_model/ultrafeedback.train.yaml --mode train
python src/scripts/run_model.py --params src/task/causal_model/ultrafeedback.inference.yaml --mode inference
```


### Retrieval Augmented Generation (RAG)

Example use case:
- Joint training of a retriever and generator in a RAG approach.
- By training on dataset from different domains, i.e. domain specific (query, passage, answer), we can adapt the pretrained LLM to different domains.  
- The data is [SQUAD-v2](https://huggingface.co/datasets/squad_v2).

Modeling:
* Retriever model: [AutoModel](https://huggingface.co/docs/transformers/v4.36.1/en/model_doc/auto#transformers.AutoModel) using [intfloat/e5-large-v2](https://huggingface.co/intfloat/e5-large-v2). 
Using (query, candidate passage) pairs for training retriever.
* Generator model: [AutoModelForCausalLM](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForCausalLM) using [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf). 
Generator's task is to predict answer given (query, passage).
* Training: custom training code that combines the loss from retriever and generator.
* Data class: a custom `RagDataset` defined in `src.data.rag_dataset.py` to manage examples.

Command for train:
```
python src/scripts/run_model.py --params src/task/rag_model/squad.train.yaml --mode train
```

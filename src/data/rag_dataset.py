import torch
from torch.utils.data import Dataset
from collections import namedtuple

from tqdm import tqdm

rag_fields = ["query_input_ids", "query_attention_mask", "passage_input_ids", "passage_attention_mask",
              "qpa_input_ids", "qpa_attention_mask",
              #"qp_input_ids", "qp_attention_mask",
              "qp_len"]

RagInstance = namedtuple("RagInstance", field_names=rag_fields)
RagBatch = namedtuple("RagBatch", field_names=rag_fields)


class RagDataset(Dataset):
    def __init__(self, data, config, retriever_tokenizer, generator_tokenizer):
        self.config = config
        self.data = data
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.retriever_tokenizer = retriever_tokenizer
        self.generator_tokenizer = generator_tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def numberize(self):
        instances = []
        for inst_index, datapoint in tqdm(enumerate(self.data), total=len(self.data), desc="Processing data"):
            query = datapoint["query"]
            passage = datapoint["passage"]
            answer = datapoint["answer"]

            query_tokenized = self.retriever_tokenizer(f"query: {query}", padding="max_length", max_length=self.config.hyperparams.query_max_seq_len, truncation=True)
            passage_tokenized = self.retriever_tokenizer(f"passage: {passage}", padding="max_length", max_length=self.config.hyperparams.passage_max_seq_len, truncation=True)

            # Here, input = "query: <query> passage: <passage> answer: <answer>", output = "<answer>"
            qpa_text = f"query: {query} passage: {passage} answer: {answer}"
            qpa_tokenized = self.generator_tokenizer(qpa_text, padding="max_length", max_length=self.config.hyperparams.generator_max_seq_len, truncation=True)

            qp_text = f"query: {query} passage: {passage} answer:"
            qp_tokenized = self.generator_tokenizer(qp_text, padding=False)

            qp_len = len(qp_tokenized["input_ids"])

            instance = RagInstance(query_input_ids=query_tokenized.input_ids,
                                   query_attention_mask=query_tokenized.attention_mask,
                                   passage_input_ids=passage_tokenized.input_ids,
                                   passage_attention_mask=passage_tokenized.attention_mask,
                                   qpa_input_ids=qpa_tokenized.input_ids,
                                   qpa_attention_mask=qpa_tokenized.attention_mask,
                                   #qp_input_ids=qp_tokenized.input_ids,
                                   #qp_attention_mask=qp_tokenized.attention_mask,
                                   qp_len=qp_len)

            instances.append(instance)

        self.data = instances

    def collate_fn(self, batch):
        query_input_ids = torch.tensor([instance.query_input_ids for instance in batch], dtype=torch.long, device=self.device)
        query_attention_mask = torch.tensor([instance.query_attention_mask for instance in batch], dtype=torch.float, device=self.device)

        passage_input_ids = torch.tensor([instance.passage_input_ids for instance in batch], dtype=torch.long, device=self.device)
        passage_attention_mask = torch.tensor([instance.passage_attention_mask for instance in batch], dtype=torch.float, device=self.device)

        qpa_input_ids = torch.tensor([instance.qpa_input_ids for instance in batch], dtype=torch.long, device=self.device)
        qpa_attention_mask = torch.tensor([instance.qpa_attention_mask for instance in batch], dtype=torch.float, device=self.device)

        # qp_input_ids = torch.tensor([instance.qp_input_ids for instance in batch], dtype=torch.long,
        #                              device=self.device)
        # qp_attention_mask = torch.tensor([instance.qp_attention_mask for instance in batch], dtype=torch.float,
        #                                   device=self.device)

        qp_len = [instance.qp_len for instance in batch]

        return RagBatch(query_input_ids=query_input_ids, query_attention_mask=query_attention_mask,
                            passage_input_ids=passage_input_ids, passage_attention_mask=passage_attention_mask,
                            qpa_input_ids=qpa_input_ids, qpa_attention_mask=qpa_attention_mask,
                            #qp_input_ids=qp_input_ids, qp_attention_mask=qp_attention_mask,
                            qp_len=qp_len)

import torch
from torch.utils.data import Dataset
from collections import namedtuple

from tqdm import tqdm

triplet_fields = ["anchor_input_ids", "anchor_attention_mask", "chosen_input_ids", "chosen_attention_mask",
                  "rejected_input_ids", "rejected_attention_mask", "anchor_start", "anchor_end"]

TripletInstance = namedtuple("TripletInstance", field_names=triplet_fields)
TripletBatch = namedtuple("TripletBatch", field_names=triplet_fields)


class TripletDataset(Dataset):
    def __init__(self, data, config):
        self.config = config
        self.data = data
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def numberize(self, tokenizer):
        instances = []
        for inst_index, datapoint in tqdm(enumerate(self.data), total=len(self.data), desc="Processing data"):
            texts = [datapoint["anchor"], datapoint["chosen"], datapoint["rejected"]]
            outputs = tokenizer(texts, padding="max_length", truncation=True,
                                max_length=self.config.hyperparams.max_seq_length, return_offsets_mapping=True)

            start = end = None
            for i, offset in enumerate(outputs.offset_mapping[0]):
                if offset[0] == datapoint["anchor_start"]:
                    start = i
                if offset[1] == datapoint["anchor_end"]:
                    end = i
                if start is not None and end is not None:
                    break

            assert start is not None and end is not None

            instance = TripletInstance(anchor_input_ids=outputs.input_ids[0],
                                       anchor_attention_mask=outputs.attention_mask[0],
                                       chosen_input_ids=outputs.input_ids[1],
                                       chosen_attention_mask=outputs.attention_mask[1],
                                       rejected_input_ids=outputs.input_ids[2],
                                       rejected_attention_mask=outputs.attention_mask[2],
                                       anchor_start=start, anchor_end=end)
            instances.append(instance)

        self.data = instances

    def collate_fn(self, batch):
        anchor_input_ids = torch.tensor([instance.anchor_input_ids for instance in batch], dtype=torch.long, device=self.device)
        anchor_attention_mask = torch.tensor([instance.anchor_attention_mask for instance in batch], dtype=torch.float, device=self.device)

        chosen_input_ids = torch.tensor([instance.chosen_input_ids for instance in batch], dtype=torch.long, device=self.device)
        chosen_attention_mask = torch.tensor([instance.chosen_attention_mask for instance in batch], dtype=torch.float, device=self.device)

        rejected_input_ids = torch.tensor([instance.rejected_input_ids for instance in batch], dtype=torch.long, device=self.device)
        rejected_attention_mask = torch.tensor([instance.rejected_attention_mask for instance in batch], dtype=torch.float, device=self.device)

        anchor_start = [instance.anchor_start for instance in batch]
        anchor_end = [instance.anchor_end for instance in batch]

        return TripletBatch(anchor_input_ids=anchor_input_ids, anchor_attention_mask=anchor_attention_mask,
                            chosen_input_ids=chosen_input_ids, chosen_attention_mask=chosen_attention_mask,
                            rejected_input_ids=rejected_input_ids, rejected_attention_mask=rejected_attention_mask,
                            anchor_start=anchor_start, anchor_end=anchor_end)

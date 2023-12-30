import os
import argparse
import yaml

from src.scripts.common_utils import DotDict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", required=True)
    parser.add_argument("--mode", required=True)
    args = parser.parse_args()

    os.environ["WANDB_DISABLED"] = "true"

    with open(args.params, "r") as f:
        config_data = yaml.safe_load(f)
    config = DotDict(config_data)

    task = None
    if config.task == "sequence_classification":
        from src.model.sequence_model.sequence_classification import SequenceClassification
        task = SequenceClassification(config)
    elif config.task == "custom_sequence_classification":
        from src.model.sequence_model.custom_sequence_classification import CustomSequenceClassification
        task = CustomSequenceClassification(config)
    elif config.task == "token_classification":
        from src.model.token_model.token_classification import TokenClassification
        task = TokenClassification(config)
    elif config.task == "seq2seq_lm":
        from src.model.seq2seq_model.seq2seq_lm import Seq2SeqLM
        task = Seq2SeqLM(config, config.prompt.context_prompt, config.prompt.response_prompt)
    elif config.task == "chat_ft":
        from src.model.causal_model.chat_ft import ChatFineTuneLM
        task = ChatFineTuneLM(config)
    elif config.task == "dpo_ft":
        from src.model.causal_model.dpo_ft import DPOFineTuneLM
        task = DPOFineTuneLM(config)
    elif config.task == "instruction_ft":
        from src.model.causal_model.instruction_ft import InstructionFineTuneLM
        task = InstructionFineTuneLM(config)
    elif config.task == "spanpair_classification":
        from src.model.spanpair_model.custom_spanpair_classification import CustomSpanPairClassification
        task = CustomSpanPairClassification(config)
    elif config.task == "sentence_contrastive_learning":
        from src.model.contrastive_model.sentence_contrastive_learning import SentenceTransformerContrastiveLearning
        task = SentenceTransformerContrastiveLearning(config)
    elif config.task == "triplet_contrastive_learning":
        from src.model.contrastive_model.custom_triplet_contrastive_learning import CustomTripletContrastiveLearning
        task = CustomTripletContrastiveLearning(config)
    elif config.task == "rag_e2e":
        from src.model.rag_model.rag_e2e_learning import RagEnd2End
        task = RagEnd2End(config)

    assert task is not None

    if args.mode == "train":
        task.train()
    elif args.mode == "test":
        task.test()
    elif args.mode == "inference":
        predictions = task.inference()


        
    
    

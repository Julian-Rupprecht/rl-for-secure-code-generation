import torch
import transformers
from transformers import T5ForConditionalGeneration, RobertaTokenizerFast, T5ForSequenceClassification, T5Config
from constants import BINARY_CRITIC_CHECKPOINT_PATH, BINARY_CRITIC_CONFIG_PATH


def train():
    return 0

def evaluate():
    return 0

def test():
    return 0

def main():

    critic_tokenizer = RobertaTokenizerFast.from_pretrained("Salesforce/codet5-base")
    critic_config = T5Config.from_pretrained(BINARY_CRITIC_CONFIG_PATH)
    critic_model = T5ForSequenceClassification(config=critic_config)
    critic_model._load_from_state_dict(torch.load(BINARY_CRITIC_CHECKPOINT_PATH))
    
    actor_tokenizer = RobertaTokenizerFast.from_pretrained("Salesforce/codet5-large")
    actor_model = T5ForConditionalGeneration("Salesforce/codet5-large")
    
    return 0


if __name__ == "__main__":
    main()
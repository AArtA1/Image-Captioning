from datasets import load_dataset
from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer
import json


if __name__ == "__main__":
    # Load flickr8k and split to train, test and val

    config_path = "config.json"

    with open(config_path, "r") as f:
        config = json.load(f)

    datasets = load_dataset("Naveengo/flickr8k")

    train_and_test = datasets['train'].train_test_split(test_size=0.2)
    train_test_val = train_and_test['train'].train_test_split(test_size=0.15)

    train_test_val['val'] = train_and_test['test']

    train_test_val.save_to_disk(config['hyperparams']['dataset_path'])

    corpus = datasets['train']['text']

    tokenizer = Tokenizer(models.WordPiece())

    special_tokens = ["[PAD]","[UNK]", "[CLS]", "[SEP]"]
    
    trainer = trainers.WordPieceTrainer(special_tokens=special_tokens)

    tokenizer.normalizer = normalizers.Lowercase()

    tokenizer.train_from_iterator(corpus, trainer=trainer)

    cls_token_id = tokenizer.token_to_id("[CLS]")
    
    sep_token_id = tokenizer.token_to_id("[SEP]")

    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"[CLS]:0 $A:0 [SEP]:0",
        pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)],
    )

    tokenizer.enable_padding(length=config['max_len'])

    tokenizer.save(config['pathes']['tokenizer'])

from transformers import BertTokenizer, BertModel
from local_io import LocalIO
import torch


class BertClinical(object):
    def __init__(self, bert_model_name: str, max_tokens_sentences: int, num_sentences: int):
        self.bert_model_name = bert_model_name
        self.num_sentences = num_sentences
        self.model = BertModel.from_pretrained(bert_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.max_tokens_sentences = max_tokens_sentences

    def tag(self, entry: str, is_file: bool) -> list:
        if is_file:
            local_io = LocalIO(entry)
            text = local_io.load_txt()
        else:
            text = entry

        tokens = self.__tokenize(text)
        num_tokens_by_sentence = int(len(tokens)/self.num_sentences)
        tokenized_text = []
        for i in range(self.num_sentences):
            sentence_tokens = tokens[i*num_tokens_by_sentence:(i+1)*num_tokens_by_sentence]\
                if i < self.num_sentences-1 else tokens[i*num_tokens_by_sentence:len(tokens)]
            padded_sentence_tokens = self.__tag_sentence(sentence_tokens, i == 0)
            tokenized_text.append(padded_sentence_tokens)
        return tokenized_text

    def embedding(self, entry: str, is_file: bool) -> list:
        sentences_tokens = self.tag(entry, is_file)
        [self.__embedding(sentence_tokens) for sentence_tokens in sentences_tokens]

    # ----------------  Supporting methods ----------------------

    def __embedding(self, tokens: list) -> (torch.tensor, torch.tensor):
        attention_mask = BertClinical.__get_attention_mask(tokens)
        token_ids = self.__get_token_ids(tokens)
        torch_token_ids = torch.tensor(token_ids).unsqueeze(0)
        torch_attention_mask = torch.tensor(attention_mask).unsqueeze(0)
        pooled_output, cls = self.model(torch_token_ids, torch_attention_mask)
        return pooled_output, cls

    @staticmethod
    def __get_attention_mask(tokens: list) ->list:
        return [0 if i == '[PAD]' else 1 for i in tokens]

    def __get_token_ids(self, tokens: list) -> list:
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def __tag_sentence(self, tokens: list, is_leading: bool) -> str:
        part_tokens = tokens[0:self.max_tokens_sentences] if len(tokens) > self.max_tokens_sentences else tokens + ['[PAD]']*(self.max_tokens_sentences - len(tokens))
        return ['[CLS]'] + part_tokens + ['[SEP]'] if is_leading else ['[SEP]'] + part_tokens + ['[SEP]']

    def __tokenize(self, text: str) -> list:
        return self.tokenizer.tokenize(text)

    def __str__(self) -> str:
        return self.bert_model_name + str(self.num_sentences)

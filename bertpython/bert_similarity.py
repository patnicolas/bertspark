

from bert_clinical import BertClinical
import torch

class BertSimilarity(object):
    def __init__(self, bert_clinical: BertClinical):
        self.bert_clinical = bert_clinical

    def similarity_cls(self, notes: list) -> float:
        assert (len(notes) > 1, 'number of notes should be > 1')
        embeddings = [self.bert_clinical.embedding(note) for note in notes]
        t_similarity = [torch.dot(embeddings[0], next_embedding) for next_embedding in embeddings[1:]]
        mean_similarity = torch.mean(t_similarity)
        return mean_similarity

from bert_clinical import BertClinical
from local_io import LocalIO
import os





if __name__ == '__main__':
    current_path = os.getcwd()
    localIO = LocalIO('notes/note-1.txt')
    content = localIO.load_txt()
    print(content)

    clinical_bert_name = 'emilyalsentzer/Bio_ClinicalBERT'
    bert_uncased_name = 'bert-base-uncased'

    bert_clinical = BertClinical(bert_uncased_name, 510, 2)
    res2 = bert_clinical.tag(content, False)
    print(str(res2))
    embedding_vec = bert_clinical.embedding(content, False)
    print(embedding_vec.shape())

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

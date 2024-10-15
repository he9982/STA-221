from contextualized_topic_models.models.ctm import CombinedTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessingStopwords
from sentence_transformers import SentenceTransformer
import pandas as pd
from nltk.corpus import stopwords as stop_words
import nltk
nltk.download('stopwords')

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# build topic model

def CTMTopic(filepath, num_topic = 10):
    df = pd.read_csv(filepath, encoding='latin-1')
    documents = [line.strip() for line in df['Full Text']]
    stopwords = list(stop_words.words("english"))
    sp = WhiteSpacePreprocessingStopwords(documents, stopwords_list=stopwords)
    preprocessed_documents, unpreprocessed_corpus, vocab, retained_indices = sp.preprocess()
    tp = TopicModelDataPreparation("all-mpnet-base-v2")

    training_dataset = tp.fit(text_for_contextual=unpreprocessed_corpus, text_for_bow=preprocessed_documents)
    ctm = CombinedTM(bow_size=len(tp.vocab), contextual_size=768, n_components=num_topic, num_epochs=10)
    ctm.fit(training_dataset) # run the model
    topics = ctm.get_topic_lists(num_topic)
    return topics



if __name__ == '__main__':
    CTMTopic('/Users/kuangli/Desktop/UCD/Fall_2024/STA_221/BA_Romdom.csv')
import pandas as pd
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel


class CoherenceComputation:
    def __init__(self, topic_model):
        self.topic_model = topic_model
    
    def compute_coherence(self, docs: list, include_outliers: bool = False) -> float:
        """Compute the coherence of topics in the BERTopic model.

        Args:
            docs (list): List of documents used to compute coherence.

        Returns:
            float: Coherence score of the topics.
        """
        topics = self.topic_model.get_document_info(docs)['Topic'].tolist()
        documents = pd.DataFrame({"Document": docs,
                                  "ID": range(len(docs)),
                                  "Topic": topics})
        documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
        cleaned_docs = self.topic_model._preprocess_text(documents_per_topic.Document.values)

        # Extract vectorizer and analyzer from BERTopic
        vectorizer = self.topic_model.vectorizer_model
        analyzer = vectorizer.build_analyzer()

        # Use .get_feature_names_out() if you get an error with .get_feature_names()
        words = vectorizer.get_feature_names_out()

        # Extract features for Topic Coherence evaluation
        tokens = [analyzer(doc) for doc in cleaned_docs]
        dictionary = corpora.Dictionary(tokens)
        corpus = [dictionary.doc2bow(token) for token in tokens]

        # Extract words in each topic if they are non-empty and exist in the dictionary
        topic_words = []
        if not include_outliers:
            for topic in range(len(set(topics)) - self.topic_model._outliers):
                words = list(zip(*self.topic_model.get_topic(topic)))[0]
                words = [word for word in words if word in dictionary.token2id]
                topic_words.append(words)
            topic_words = [words for words in topic_words if len(words) > 0]
        
        else:
            for topic in range(len(set(topics))):
                words = list(zip(*self.topic_model.get_topic(topic)))[0]
                words = [word for word in words if word in dictionary.token2id]
                topic_words.append(words)
            topic_words = [words for words in topic_words if len(words) > 0]

        # Evaluate Coherence
        coherence_model = CoherenceModel(topics=topic_words, 
                                         texts=tokens, 
                                         corpus=corpus,
                                         dictionary=dictionary, 
                                         coherence='c_v')
        coherence = coherence_model.get_coherence()        
        return coherence
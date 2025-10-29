import pandas as pd
import nltk
from nltk.corpus import wordnet
from tqdm import tqdm

class SymptomDataProcessor:
    """
    Processes symptom data for analysis, including cleaning, normalization,
    and transformation into a transactional format suitable for Apriori.
    """

    def __init__(self, data_path):
        """
        Initializes the processor with the path to the symptom dataset.

        Args:
            data_path (str): The file path to the symptom dataset (e.g., 'data/dataset.csv').
        """
        self.data_path = data_path
        self.df = None
        self.transactions = []
        self._download_nltk_data()

    def _download_nltk_data(self):
        """Downloads NLTK 'wordnet' if not already present."""
        try:
            wordnet.synsets('test')
        except nltk.downloader.DownloadError:
            nltk.download('wordnet')

    def _clean_symptom(self, symptom):
        """
        Cleans a single symptom string by removing extra spaces and underscores.

        Args:
            symptom (str): The symptom string to clean.

        Returns:
            str: The cleaned symptom string.
        """
        if pd.isna(symptom):
            return ""
        return symptom.strip().replace('_', ' ')

    def _normalize_symptom(self, symptom):
        """
        Normalizes a symptom to its canonical form using WordNet.
        If a canonical form is not found, the original cleaned symptom is returned.

        Args:
            symptom (str): The symptom string to normalize.

        Returns:
            str: The normalized (canonical) symptom string.
        """
        synsets = wordnet.synsets(symptom)
        if synsets:
            # Use the first lemma of the first synset as the canonical form
            return synsets[0].lemmas()[0].name().replace('_', ' ')
        return symptom

    def process_data(self):
        """
        Executes the full data processing pipeline:
        1. Loads the data.
        2. Cleans and normalizes symptoms.
        3. Transforms the data into a transactional list.

        Returns:
            list: A list of lists, where each inner list contains the symptoms for a transaction.
        """
        self.df = pd.read_csv(self.data_path)
        
        symptom_cols = [col for col in self.df.columns if 'Symptom' in col]
        
        transactions = []
        # Wrap the DataFrame iterator with tqdm for a progress bar
        for index, row in tqdm(self.df.iterrows(), total=self.df.shape[0], desc="Processing symptoms"):
            transaction = []
            for col in symptom_cols:
                symptom = self._clean_symptom(row[col])
                if symptom:
                    normalized_symptom = self._normalize_symptom(symptom)
                    transaction.append(normalized_symptom)
            if transaction:
                transactions.append(transaction)
        
        self.transactions = transactions
        return self.transactions


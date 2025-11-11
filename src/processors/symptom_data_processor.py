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
        self.symptom_mapping = {
            # --- Abdominal / Stomach ---
            'belly pain': 'abdominal pain',
            'stomach pain': 'abdominal pain',
            'swelling of stomach': 'distention of abdomen',

            # --- Pain & Discomfort ---
            'burning micturition': 'painful urination',
            'irritation in anus': 'anal discomfort',
            'pain in anal region': 'anal discomfort',
            # 'hip joint pain': 'joint pain',
            # 'knee pain': 'joint pain',

            # --- General Symptoms ---
            'lethargy': 'fatigue',
            'malaise': 'fatigue',
            'spinning movements': 'dizziness',
            'loss of balance': 'unsteadiness',
            'palpitation': 'fast heart rate',
            'perspiration': 'sweating',
            'chill': 'chills',
            'vomit': 'vomiting',
            'nausea': 'vomiting',

            # --- Skin ---
            'itch': 'itching',
            'internal itching': 'itching',

            # --- Swelling ---
            'swollen extremeties': 'swollen extremities',
            'swollen legs': 'swollen extremities',

            # --- Other Groups ---
            'excessive hunger': 'increased appetite',
            'weakness in limbs': 'muscle weakness',

            # --- Spelling/Typo Corrections ---
            'cold hands and feets': 'cold hands and feet',
            'dischromic  patches': 'dischromic patches',
            'scurring': 'skin scaling',  # Assumed typo of 'scurfing' or 'scaling'
            'spotting  urination': 'spotting during urination',
            'toxic look (typhos)': 'toxic look (typhus)'
        }
        self._download_nltk_data()

    def _download_nltk_data(self):
        """Downloads NLTK 'wordnet' if not already present."""
        try:
            wordnet.synsets('test')
        except Exception:
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
        Normalizes a symptom to its canonical form.
        First checks the symptom mapping dictionary, then falls back to WordNet.
        If a canonical form is not found, the original cleaned symptom is returned.

        Args:
            symptom (str): The symptom string to normalize.

        Returns:
            str: The normalized (canonical) symptom string.
        """
        # First check the mapping dictionary (case-insensitive)
        symptom_lower = symptom.lower()
        if symptom_lower in self.symptom_mapping:
            return self.symptom_mapping[symptom_lower]
        
        # # If not in specified dict, use WordNet
        # synsets = wordnet.synsets(symptom)
        # if synsets:
        #     # Use the first lemma of the first synset as the canonical form
        #     return synsets[0].lemmas()[0].name().replace('_', ' ')
        return symptom

    def normalize_symptom(self, symptom):
        """
        Public helper for normalizing symptoms that handles cleaning internally.

        Args:
            symptom (str): Raw symptom string.

        Returns:
            str: Normalized symptom string, or empty string if none.
        """
        cleaned_symptom = self._clean_symptom(symptom)
        if not cleaned_symptom:
            return ""
        return self._normalize_symptom(cleaned_symptom)

    def process_data(self, group_by_disease=True, min_transactions=10):
        """
        Executes the full data processing pipeline:
        1. Loads the data.
        2. Cleans and normalizes symptoms.
        3. Transforms the data into a transactional list.

        Args:
            group_by_disease (bool): If True, groups symptoms by disease (each disease is a basket).
                                   If False, each row is a transaction.
            min_transactions (int): Minimum number of transactions required. Raises warning if below.

        Returns:
            list: A list of lists, where each inner list contains the symptoms for a transaction.
        """
        self.df = pd.read_csv(self.data_path)
        
        symptom_cols = [col for col in self.df.columns if 'Symptom' in col]
        disease_col = 'Disease' if 'Disease' in self.df.columns else None
        
        if group_by_disease and disease_col:
            # Group by disease: each disease becomes a basket with all unique symptoms
            transactions = []
            disease_symptoms = {}
            
            for index, row in tqdm(self.df.iterrows(), total=self.df.shape[0], desc="Processing symptoms"):
                disease = row[disease_col] if pd.notna(row[disease_col]) else None
                if disease:
                    if disease not in disease_symptoms:
                        disease_symptoms[disease] = set()
                    
                    for col in symptom_cols:
                        symptom = self._clean_symptom(row[col])
                        if symptom:
                            normalized_symptom = self._normalize_symptom(symptom)
                            disease_symptoms[disease].add(normalized_symptom)
            
            # Convert sets to lists for transactions
            for disease, symptoms in disease_symptoms.items():
                if symptoms:  # Only add if disease has symptoms
                    transactions.append(list(symptoms))
        else:
            # Each row is a transaction (Not grouped by disease)
            transactions = []
            for index, row in tqdm(self.df.iterrows(), total=self.df.shape[0], desc="Processing symptoms"):
                transaction = []
                for col in symptom_cols:
                    symptom = self._clean_symptom(row[col])
                    if symptom:
                        normalized_symptom = self._normalize_symptom(symptom)
                        transaction.append(normalized_symptom)
                if transaction:
                    transactions.append(transaction)
        
        # Validate minimum transactions
        if len(transactions) < min_transactions:
            import warnings
            warnings.warn(
                f"Only {len(transactions)} transactions found. Minimum recommended: {min_transactions}. "
                "Consider augmenting data or lowering min_support threshold.",
                UserWarning
            )
        
        self.transactions = transactions
        return self.transactions


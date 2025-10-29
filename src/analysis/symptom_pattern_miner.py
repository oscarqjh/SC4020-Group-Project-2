import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

class SymptomPatternMiner:
    """
    Performs symptom pattern mining using the Apriori algorithm.

    This class uses the 'mlxtend' library, a valuable resource for frequent pattern
    mining in Python. For more information, see the official documentation:
    http://rasbt.github.io/mlxtend/
    """

    def __init__(self, transactions, min_support=0.01):
        """
        Initializes the miner with transactional data and a minimum support threshold.

        Args:
            transactions (list): A list of lists, where each inner list is a transaction.
            min_support (float): The minimum support threshold for the Apriori algorithm.
        """
        self.transactions = transactions
        self.min_support = min_support
        self.te = TransactionEncoder()
        self.df_encoded = None
        self.frequent_itemsets = None
        self.rules = None

    def _encode_transactions(self):
        """
        Encodes the transactional data into a one-hot encoded pandas DataFrame.
        """
        te_ary = self.te.fit(self.transactions).transform(self.transactions)
        self.df_encoded = pd.DataFrame(te_ary, columns=self.te.columns_)

    def mine_frequent_itemsets(self):
        """
        Mines for frequent itemsets using the Apriori algorithm.

        Returns:
            pandas.DataFrame: A DataFrame containing the frequent itemsets and their support.
        """
        if self.df_encoded is None:
            self._encode_transactions()
        
        # Citation: This implementation uses the apriori function from the mlxtend library.
        # Raschka, S. (2018). MLxtend: Providing machine learning and data science utilities 
        # and extensions to Python's scientific computing stack. Journal of Open Source Software, 3(24), 638.
        self.frequent_itemsets = apriori(self.df_encoded, min_support=self.min_support, use_colnames=True)
        return self.frequent_itemsets

    def generate_association_rules(self, metric="confidence", min_threshold=0.5):
        """
        Generates association rules from the frequent itemsets.

        Args:
            metric (str): The metric to evaluate the rules (e.g., 'confidence', 'lift').
            min_threshold (float): The minimum threshold for the given metric.

        Returns:
            pandas.DataFrame: A DataFrame containing the association rules.
        """
        if self.frequent_itemsets is None:
            self.mine_frequent_itemsets()
            
        self.rules = association_rules(self.frequent_itemsets, metric=metric, min_threshold=min_threshold)
        return self.rules



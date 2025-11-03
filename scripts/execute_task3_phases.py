#!/usr/bin/env python3
"""
Temporary script to execute Phase 1 and Phase 2 from task3_diagnostic_assistant.ipynb.
This script extracts and runs the code from the notebook cells.

Usage:
    python scripts/execute_task3_phases.py

Prerequisites:
    - Virtual environment activated
    - Dependencies installed: pip install -r requirements.txt
    - NumPy 1.26.4 (not 2.x) for ChromaDB compatibility
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

print("="*70)
print("Task 3: Phase 1 & Phase 2 Execution Script")
print("="*70)

# ============================================================================
# PHASE 1: ML Model Training & Vocabulary Export
# ============================================================================
print("\n" + "="*70)
print("PHASE 1: ML Model Training & Vocabulary Export")
print("="*70)

# Phase 1 - Cell 1: Imports
print("\n[Phase 1] Step 1: Importing libraries...")
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json

print("✓ Libraries imported successfully.")

# Phase 1 - Cell 2: Data Loading
print("\n[Phase 1] Step 2: Loading and preprocessing data...")
DATA_PATH = project_root / 'data' / 'dataset.csv'

df = pd.read_csv(DATA_PATH)
print(f"Dataset Shape: {df.shape}")

# Combine symptom columns into a single text field
symptom_cols = [f'Symptom_{i}' for i in range(1, 18)]

# Define function to clean string values only (treat non-strings as None)
def clean_symptom_value(value):
    """Clean symptom value: trim and normalize underscores for strings only, leave non-strings as None."""
    if not isinstance(value, str):
        return None
    # Strip whitespace and normalize underscores (remove spaces around underscores, normalize multiple underscores)
    import re
    cleaned = value.strip()
    cleaned = re.sub(r'\s+_\s+', '_', cleaned)  # Remove spaces around underscores
    cleaned = re.sub(r'\s+_', '_', cleaned)  # Remove spaces before underscores
    cleaned = re.sub(r'_\s+', '_', cleaned)  # Remove spaces after underscores
    cleaned = cleaned.strip('_')  # Remove leading/trailing underscores
    return cleaned if cleaned else None

# Apply cleaning function to each symptom column (only processes strings, leaves NaN as None)
for col in symptom_cols:
    df[col] = df[col].apply(clean_symptom_value)

# Build symptoms_text from symptom_cols by selecting only non-null entries (without prior astype(str))
def build_symptoms_text(row):
    """Build symptoms_text from symptom columns, selecting only non-null entries without converting NaN to strings."""
    symptoms = []
    for val in row[symptom_cols]:
        if pd.notna(val) and val is not None and isinstance(val, str) and val.strip():
            symptoms.append(val)
    return ' '.join(symptoms)

df['symptoms_text'] = df.apply(build_symptoms_text, axis=1)

# Clean up multiple spaces in symptoms_text
df['symptoms_text'] = df['symptoms_text'].str.replace(r'\s+', ' ', regex=True).str.strip()

# Drop rows with empty symptoms_text
df = df[df['symptoms_text'].notna() & (df['symptoms_text'].str.strip() != '')].copy()

# Normalize Disease column: strip whitespace
df['Disease'] = df['Disease'].str.strip()

# Prepare features and target
X = df['symptoms_text']
y = df['Disease']

print(f"✓ Data loaded. Number of unique diseases: {y.nunique()}")
print(f"  Sample symptoms_text: {X.iloc[0][:80]}...")

# Phase 1 - Cell 3: Model Training
print("\n[Phase 1] Step 3: Training model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the ML pipeline
disease_pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model
disease_pipeline.fit(X_train, y_train)
print("✓ Model training complete.")

# Evaluate the model
y_pred = disease_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✓ Model Accuracy: {accuracy:.4f}")

# Export the vocabulary
vocabulary = disease_pipeline.named_steps['vectorizer'].get_feature_names_out()
vocabulary_list = vocabulary.tolist()

vocab_path = project_root / 'outputs' / 'models' / 'symptom_vocabulary.json'
vocab_path.parent.mkdir(parents=True, exist_ok=True)
with open(vocab_path, 'w') as f:
    json.dump(vocabulary_list, f, indent=2)

print(f"✓ Vocabulary exported to: {vocab_path}")
print(f"✓ Vocabulary size: {len(vocabulary_list)} unique symptom tokens")

# Save the trained model
model_path = project_root / 'outputs' / 'models' / 'disease_model.pkl'
model_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(disease_pipeline, model_path)

print(f"✓ Trained model saved to: {model_path}")

# ============================================================================
# PHASE 2: Knowledge Base Setup with ChromaDB
# ============================================================================
print("\n" + "="*70)
print("PHASE 2: Knowledge Base Setup with ChromaDB")
print("="*70)

# Phase 2 - Cell 1: Setup
print("\n[Phase 2] Step 1: Setting up environment...")
try:
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer
    import re
except ImportError as e:
    print(f"✗ ERROR: Failed to import required libraries: {e}")
    print("  Please ensure dependencies are installed: pip install -r requirements.txt")
    print("  Also verify NumPy version: python -c 'import numpy; print(numpy.__version__)'")
    print("  Expected: NumPy 1.26.4 (not 2.x)")
    sys.exit(1)

# Initialize SentenceTransformer model
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✓ SentenceTransformer model loaded successfully.")
    print(f"✓ Embedding dimension: {embedding_model.get_sentence_embedding_dimension()}")
except Exception as e:
    print(f"✗ ERROR: Failed to load SentenceTransformer: {e}")
    print("  This may be due to NumPy compatibility issues.")
    print("  Please verify NumPy version: python -c 'import numpy; print(numpy.__version__)'")
    print("  Expected: NumPy 1.26.4 (not 2.x)")
    sys.exit(1)

# Phase 2 - Cell 2: Load Data
print("\n[Phase 2] Step 2: Loading disease information...")
DESCRIPTION_PATH = project_root / 'data' / 'symptom_Description.csv'
PRECAUTION_PATH = project_root / 'data' / 'symptom_precaution.csv'

# Load disease descriptions
descriptions_df = pd.read_csv(DESCRIPTION_PATH)
print(f"Descriptions shape: {descriptions_df.shape}")

# Load disease precautions
precautions_df = pd.read_csv(PRECAUTION_PATH)
print(f"Precautions shape: {precautions_df.shape}")

# Data cleaning: strip whitespace from Disease column
descriptions_df['Disease'] = descriptions_df['Disease'].str.strip()
precautions_df['Disease'] = precautions_df['Disease'].str.strip()

# Normalize disease names using correction map to fix typos and mismatches
def normalize_disease_name(name):
    """Normalize disease names to fix typos and inconsistencies."""
    correction_map = {
        'hemmorhoids': 'hemorrhoids',  # Fix typo: hemmorhoids -> hemorrhoids
        'Paroymsal': 'Paroxysmal',  # Fix typo: Paroymsal -> Paroxysmal
    }
    
    normalized = name
    for typo, correct in correction_map.items():
        if typo in normalized:
            normalized = normalized.replace(typo, correct)
    return normalized

# Apply normalization to both dataframes
descriptions_df['Disease'] = descriptions_df['Disease'].apply(normalize_disease_name)
precautions_df['Disease'] = precautions_df['Disease'].apply(normalize_disease_name)

# Verify data alignment after normalization
desc_diseases = set(descriptions_df['Disease'])
prec_diseases = set(precautions_df['Disease'])
mismatches_desc = desc_diseases - prec_diseases
mismatches_prec = prec_diseases - desc_diseases

print(f"After normalization:")
print(f"  Unique diseases in descriptions: {descriptions_df['Disease'].nunique()}")
print(f"  Unique diseases in precautions: {precautions_df['Disease'].nunique()}")

if mismatches_desc:
    print(f"⚠ WARNING: Diseases only in descriptions: {mismatches_desc}")
if mismatches_prec:
    print(f"⚠ WARNING: Diseases only in precautions: {mismatches_prec}")
if not mismatches_desc and not mismatches_prec:
    print("✓ All diseases match between descriptions and precautions files.")

# Phase 2 - Cell 3: Initialize ChromaDB
print("\n[Phase 2] Step 3: Initializing ChromaDB...")
VECTORSTORE_PATH = project_root / 'outputs' / 'vectorstore' / 'chroma_db'

# Create the vectorstore directory
VECTORSTORE_PATH.mkdir(parents=True, exist_ok=True)
print(f"Vectorstore directory: {VECTORSTORE_PATH}")

# Initialize ChromaDB persistent client
chroma_client = chromadb.PersistentClient(path=str(VECTORSTORE_PATH))
print("✓ ChromaDB persistent client initialized.")

# Define embedding function class for ChromaDB 1.3.0
class SentenceTransformerEmbeddingFunction:
    """Embedding function class that wraps SentenceTransformer for ChromaDB."""
    def __init__(self, model):
        self.model = model
    
    def __call__(self, input):
        """Embed input texts using SentenceTransformer.
        
        Args:
            input: Can be a single string or a list of strings
            
        Returns:
            List of embeddings (list of lists)
        """
        if isinstance(input, str):
            input = [input]
        embeddings = self.model.encode(input, show_progress_bar=False)
        return embeddings.tolist()
    
    def embed_query(self, input):
        """Embed a single query string or list of query strings.
        
        Args:
            input: Can be a single string or a list of strings
            
        Returns:
            Embedding as a list of floats (single query) or list of lists (multiple queries)
            Note: ChromaDB may expect list of lists even for single query
        """
        # Normalize to list - ChromaDB may call this with a single string or a list
        texts = [input] if isinstance(input, str) else input
        
        # Encode using SentenceTransformer
        embeddings = self.model.encode(texts, show_progress_bar=False)
        
        # Convert all embeddings to list of lists (each as list of Python floats)
        # ChromaDB expects this format even for single queries when using query_texts
        result = [[float(x) for x in emb.tolist()] for emb in embeddings]
        
        # For single query, ChromaDB might expect just the first embedding vector
        # But based on the error, it seems to expect list of lists
        # Return list of lists - ChromaDB will handle extracting what it needs
        return result

# Create embedding function instance
embed_function = SentenceTransformerEmbeddingFunction(embedding_model)

# Delete existing collection if it exists to avoid conflicts
try:
    chroma_client.delete_collection("disease_info")
    print("  Cleared existing 'disease_info' collection.")
except Exception:
    pass  # Collection doesn't exist, which is fine

# Create new collection with embedding function
collection = chroma_client.create_collection(
    name="disease_info",
    embedding_function=embed_function
)
print(f"✓ Collection 'disease_info' created. Current document count: {collection.count()}")

# Phase 2 - Cell 4: Populate Vectorstore
print("\n[Phase 2] Step 4: Populating vector database...")
# Merge descriptions and precautions
merged_df = pd.merge(descriptions_df, precautions_df, on='Disease', how='inner')
print(f"Merged dataframe shape: {merged_df.shape}")

# Assert that merged row count equals the number of unique diseases
unique_disease_count = descriptions_df['Disease'].nunique()
merged_row_count = len(merged_df)
assert merged_row_count == unique_disease_count, f"Merge failed: expected {unique_disease_count} rows, got {merged_row_count}."
print(f"✓ Merge successful: {merged_row_count} rows match {unique_disease_count} unique diseases.")

# Create combined documents
def create_document(row):
    description = row['Description']
    precautions = [row[f'Precaution_{i}'] for i in range(1, 5) if pd.notna(row[f'Precaution_{i}'])]
    precautions_text = ', '.join(precautions) if precautions else 'No specific precautions listed'
    return f"Disease: {row['Disease']}\n\nDescription: {description}\n\nPrecautions: {precautions_text}"

merged_df['document'] = merged_df.apply(create_document, axis=1)
print(f"✓ Created {len(merged_df)} combined documents.")

# Create stable IDs from disease names (slugify)
def slugify_disease_name(name):
    """Create a stable ID from disease name."""
    slug = name.lower()
    slug = re.sub(r'[^\w\s-]', '', slug)  # Remove special chars except spaces and hyphens
    slug = re.sub(r'[-\s]+', '_', slug)  # Replace spaces and hyphens with underscores
    slug = slug.strip('_')  # Remove leading/trailing underscores
    return f"disease_{slug}"

# Prepare data for ChromaDB
documents = merged_df['document'].tolist()
ids = [slugify_disease_name(disease) for disease in merged_df['Disease']]
metadatas = [{'disease': disease} for disease in merged_df['Disease']]
print(f"✓ Prepared {len(documents)} documents for ChromaDB.")

# Add documents to ChromaDB (embeddings will be generated automatically by the registered embedding_function)
collection.add(documents=documents, ids=ids, metadatas=metadatas)
print(f"✓ Successfully added {len(documents)} documents to ChromaDB collection.")
print(f"✓ Final collection count: {collection.count()}")

# Test the vectorstore using query_texts (works with embedding function)
print("\n[Phase 2] Testing vectorstore with sample query...")
test_query = 'fever and cough'
test_results = collection.query(query_texts=[test_query], n_results=3)
print(f"Test query results for '{test_query}':")
for i, (doc, metadata) in enumerate(zip(test_results['documents'][0], test_results['metadatas'][0])):
    print(f"  {i+1}. {metadata['disease']}: {doc[:100]}...")

# ============================================================================
# VALIDATION
# ============================================================================
print("\n" + "="*70)
print("VALIDATION: Checking Output Artifacts")
print("="*70)

# Check Phase 1 artifacts
vocab_path = project_root / 'outputs' / 'models' / 'symptom_vocabulary.json'
model_path = project_root / 'outputs' / 'models' / 'disease_model.pkl'

if vocab_path.exists():
    with open(vocab_path) as f:
        vocab_data = json.load(f)
    print(f"✓ symptom_vocabulary.json exists ({len(vocab_data)} tokens)")
else:
    print(f"✗ symptom_vocabulary.json NOT FOUND")

if model_path.exists():
    print(f"✓ disease_model.pkl exists")
else:
    print(f"✗ disease_model.pkl NOT FOUND")

# Check Phase 2 artifacts
vectorstore_dir = project_root / 'outputs' / 'vectorstore' / 'chroma_db'
if vectorstore_dir.exists() and any(vectorstore_dir.iterdir()):
    print(f"✓ ChromaDB vectorstore exists at: {vectorstore_dir}")
    print(f"  Collection count: {collection.count()} documents")
else:
    print(f"✗ ChromaDB vectorstore NOT FOUND or empty")

print("\n" + "="*70)
print("✓ EXECUTION COMPLETE")
print("="*70)


# Traycer Plan: Task 3 - AI Diagnostic Assistant (v5)

**Objective:** Implement a multi-tool AI assistant for Task 3.
* **Python Version:** This plan assumes a `python >= 3.11` environment, as specified in `pyproject.toml`.
* **Tool 1 (Symptom Checker):** A RAG + ML pipeline using the Disease Symptom dataset.
* **Tool 2 (Cancer Analysis Finder):** A "precontext" LLM tool using the findings from `outputs/analysis_summary.txt` and `outputs/feature_importance.txt`.

## Phase 0: Environment Setup (Clean Install)

**Goal:** Establish a stable environment with compatible libraries.

1.  **Verify Environment:** Ensure you are in a clean `venv` running Python 3.11 or 3.12.
2.  **Update `requirements.txt`:** Modify the `requirements.txt` file. Add the following new dependencies. Use these *exact* versions to ensure compatibility and avoid the previous mismatch issues.
    ```
    # --- Task 3 Dependencies ---
    chromadb==1.2.2
    sentence-transformers==3.0.1
    google-genai==1.47.0
    python-dotenv==1.0.1
    joblib==1.2.0
    ```
3.  **Install Dependencies:** Run `pip install -r requirements.txt` in your activated virtual environment.
4.  **Create Output Directories:** In the `outputs/` directory, create:
    * `outputs/models/`
    * `outputs/vectorstore/` (This will *only* be for the Symptom tool)
5.  **Path Resolution Strategy:** All scripts must resolve paths consistently. **Recommended approach:** Use `pathlib.Path(__file__).resolve().parent` to determine the script's directory, then resolve output paths relative to the project root. Alternatively, ensure scripts are run from the project root (validate with `os.getcwd()` or `Path.cwd()`). In `task3_app.py`, add a setup step that:
    * Validates required directories exist (`outputs/models/`, `outputs/vectorstore/`)
    * Exits with a clear error message if directories are missing (e.g., `raise FileNotFoundError("Required directory not found: outputs/models/. Please run Phase 0 step 4.")`)
    * Optionally creates missing directories or changes to project root if needed
    * Example validation at the start of `__init__`:
      ```python
      from pathlib import Path
      project_root = Path(__file__).resolve().parent.parent.parent
      required_dirs = [project_root / "outputs" / "models", project_root / "outputs" / "vectorstore"]
      for dir_path in required_dirs:
          if not dir_path.exists():
              raise FileNotFoundError(f"Required directory not found: {dir_path}. Please ensure Phase 0 step 4 is completed.")
      ```
6.  **Create Notebook:** In `notebooks/task3/`, create `task3_diagnostic_assistant.ipynb`.

## Phase 1: ML Model Training & Vocabulary Export (Tool 1)

**Goal:** Train the ML classifier for the Symptom Checker and export its vocabulary.
**Location:** `notebooks/task3/task3_diagnostic_assistant.ipynb`.

1.  **Imports:** Import `pandas`, `joblib`, `json`, `sklearn.pipeline.Pipeline`, `sklearn.ensemble.RandomForestClassifier`, `sklearn.feature_extraction.text.CountVectorizer`, and `sklearn.model_selection.train_test_split`.
2.  **Load Data:** Load `data/dataset.csv`.
3.  **Preprocess:**
    * Combine `Symptom_1`...`Symptom_17` into a single `symptoms_text` column (handle `NaN`s).
    * Define `X = df['symptoms_text']` and `y = df['Disease']`.
4.  **Define Pipeline:**
    * `disease_pipeline = Pipeline([('vectorizer', CountVectorizer()), ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])`
5.  **Train & Evaluate:** Split data, `fit` the `disease_pipeline`, and print the accuracy score on the test set.
6.  **Export Vocabulary:**
    * `vocabulary = disease_pipeline.named_steps['vectorizer'].get_feature_names_out()`
    * Save `vocabulary` (as a list) to `outputs/models/symptom_vocabulary.json`.
7.  **Save Model:** Save the trained `disease_pipeline` to `outputs/models/disease_model.pkl` using `joblib`.

## Phase 2: Knowledge Base Setup (Tool 1)

**Goal:** Build the ChromaDB vectorstore for the Symptom Checker.
**Location:** `notebooks/task3/task3_diagnostic_assistant.ipynb`.

1.  **Imports:** Import `chromadb` and `sentence_transformers.SentenceTransformer`.
2.  **Load Text Data:** Load `data/symptom_Description.csv` and `data/symptom_precaution.csv`.
3.  **Process Documents:** Merge precautions and descriptions into a single text document string for each disease.
4.  **Initialize Embeddings:** Load `embedding_model = SentenceTransformer('all-MiniLM-L6-v2')`.
5.  **Initialize ChromaDB:**
    * `client = chromadb.PersistentClient(path="outputs/vectorstore/chroma_db")`
    * `collection_symptoms = client.get_or_create_collection(name="disease_info")`
6.  **Populate Vectorstore:** Loop through your disease documents, generate embeddings, and add them to `collection_symptoms` with corresponding `metadata` (e.g., `{"disease": "Influenza"}`).

## Phase 3: The Multi-Tool AI Orchestrator Class

**Goal:** Create the "brain" of the application in a new script.
**Location:** Create a new file: `scripts/task3_app.py`.

1.  **Imports:** Import `joblib`, `json`, `chromadb`, `sentence_transformers.SentenceTransformer`, `google.genai`, `os`, and `dotenv`.
2.  **Define Class:** Create a class named `ProjectAssistant`.
3.  **`__init__` Method:**
    * `dotenv.load_dotenv()`
    * **Load API Key:** Load `GOOGLE_API_KEY` from `os.environ`. **Error Handling:** If the API key is missing or empty, immediately raise a `ValueError` with a clear message (e.g., `raise ValueError("GOOGLE_API_KEY environment variable is required but not set. Please set it in your .env file or environment.")`).
    * **Configure Gemini:** Wrap `genai.configure(api_key=...)` and any subsequent model initialization in a try/except block. **Error Handling Requirements:**
      * Detect missing API key and raise `ValueError` immediately (fail fast)
      * For non-transient errors (e.g., invalid API key, authentication errors), log the exception with full details and re-raise with a clear error message to the caller
      * For transient network failures (e.g., `ConnectionError`, `TimeoutError`, `google.api_core.exceptions.ServiceUnavailable`), implement optional retry logic with exponential backoff:
        * Configurable max attempts (e.g., 3 attempts)
        * Exponential backoff with jitter (e.g., `time.sleep(base_delay * (2 ** attempt) + random.uniform(0, 1))`)
        * Only retry transient errors; non-transient errors (e.g., `google.api_core.exceptions.InvalidArgument` for invalid API key) should fail fast
      * Example structure:
        ```python
        import logging
        import time
        import random
        from google.api_core import exceptions as google_exceptions
        
        logger = logging.getLogger(__name__)
        MAX_RETRIES = 3
        base_delay = 1  # Base delay in seconds for exponential backoff
        
        try:
            genai.configure(api_key=api_key)
            self.llm = genai.GenerativeModel('models/gemini-2.5-flash')
        except ValueError as e:
            logger.error(f"Configuration error: {e}")
            raise
        except (ConnectionError, TimeoutError, google_exceptions.ServiceUnavailable) as e:
            # Transient error - retry with backoff
            for attempt in range(MAX_RETRIES):
                try:
                    time.sleep(base_delay * (2 ** attempt) + random.uniform(0, 1))
                    genai.configure(api_key=api_key)
                    self.llm = genai.GenerativeModel('models/gemini-2.5-flash')
                    break
                except Exception as retry_error:
                    if attempt == MAX_RETRIES - 1:
                        logger.error(f"Failed after {MAX_RETRIES} attempts: {retry_error}")
                        raise
        except google_exceptions.InvalidArgument as e:
            # Invalid API key - fail fast
            logger.error(f"Invalid API key: {e}")
            raise ValueError(f"Invalid Google API key configuration: {e}")
        except Exception as e:
            # Other errors - log and re-raise
            logger.error(f"Unexpected error during GenAI initialization: {e}")
            raise
        ```
    * **Load Tool 1 (Symptoms):**
        * `self.symptom_model = joblib.load('outputs/models/disease_model.pkl')`
        * `with open('outputs/models/symptom_vocabulary.json') as f: self.symptom_vocabulary = json.load(f)`
        * `self.db_client = chromadb.PersistentClient(path="outputs/vectorstore/chroma_db")`
        * `self.collection_symptoms = self.db_client.get_collection("disease_info")`
        * `self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')`
    * **Load Tool 2 (Cancer):**
        * `with open('outputs/analysis_summary.txt') as f: self.cancer_summary = f.read()`
        * `with open('outputs/feature_importance.txt') as f: self.cancer_features = f.read()`
        * `self.cancer_context = self.cancer_summary + "\n\n" + self.cancer_features`

4.  **Method: `_route_query(self, user_query)`:**
    * Create a prompt to classify intent: 'symptom_check' or 'cancer_analysis'.
    * **Prompt Template:**
        ```
        You are a query router. Classify the query into one of two categories:
        1. 'symptom_check': The user is describing personal health symptoms.
        2. 'cancer_analysis': The user is asking for analysis, patterns, or features from the breast cancer study.
        Return *only* the category name.

        Query: "{user_query}"
        Category:
        ```
    * Call `self.llm.generate_content(prompt)` and return the cleaned text.

5.  **Method: `_run_symptom_checker(self, user_query)`:**
    * **Step 1 (The Fix):** LLM call to extract symptoms, constrained by `self.symptom_vocabulary`.
    * **Step 2: Handle Empty:** Check if the returned `symptoms_list` is empty.
    * **Step 3: Predict:** Call `self.symptom_model.predict()` to get `disease` and `confidence`.
    * **Step 4: Retrieve:** Query `self.collection_symptoms` using `metadata={"disease": disease}`.
    * **Step 5: Generate:** Call LLM with the final prompt, including all context and the medical disclaimer.
    * Return the final string.

6.  **Method: `_run_cancer_analysis(self, user_query)`:**
    * This is the simple "precontext" method.
    * **Prompt Template:**
        ```
        You are an AI assistant for a biomedical researcher. Answer the user's question using *only* the provided context, which contains the key findings from our Task 2 analysis.

        **Provided Context:**
        {self.cancer_context}

        **User Question:**
        {user_query}

        **Answer:**
        ```
    * Call `self.llm.generate_content(prompt)` and return its text response.

7.  **Main Method: `run(self, user_query)`:**
    * `intent = self._route_query(user_query).strip()`
    * `if intent == 'symptom_check':`
        * `return self._run_symptom_checker(user_query)`
    * `elif intent == 'cancer_analysis':`
        * `return self._run_cancer_analysis(user_query)`
    * `else:`
        * `return "I'm sorry, I can only assist with symptom checks or questions about our cancer analysis findings. How can I help?"`

**Running `task3_app.py` Checklist:**

Before running the script, ensure:
1. **Project Root Context:** Scripts should be run from the project root, or use absolute paths built with `pathlib.Path(__file__).resolve().parent`
2. **Validation Step:** The `__init__` method should validate required directories exist and exit with a clear error if missing (see Phase 0, step 5 for example code)
3. **Environment Setup:** Ensure `.env` file contains `GOOGLE_API_KEY=your_key_here` in the project root
4. **Dependencies:** All dependencies from `requirements.txt` are installed
5. **Prerequisites:** Phase 1 and Phase 2 must be completed (model files and vectorstore must exist)

**Example Command:**
```bash
# From project root:
cd /path/to/project
python scripts/task3_app.py

# Or as a module:
python -m scripts.task3_app
```

**Validation:** The script should exit with a clear error message if:
- Required directories (`outputs/models/`, `outputs/vectorstore/`) don't exist
- API key is missing or invalid
- Model files are not found

## Phase 4: Demonstration

**Goal:** Test the complete multi-tool system.
**Location:** `notebooks/task3/task3_diagnostic_assistant.ipynb`.

1.  **Import Class:** From `scripts.task3_app`, import `ProjectAssistant`. (You may need to add `src` to the path or turn `scripts` into a package).
2.  **Initialize:** `assistant = ProjectAssistant()`
3.  **Run Demos:**
    * **Test Tool 1:** `print(assistant.run("I have a bad cough, a high fever, and my whole body aches."))`
    * **Test Tool 2:** `print(assistant.run("What are the most discriminative patterns for benign tumors?"))`
    * **Test Tool 2:** `print(assistant.run("Which features are most important in malignant patterns?"))`
    * **Test Router:** `print(assistant.run("Hello, how are you?"))`
# Traycer Plan: Task 3 - AI Diagnostic Assistant (v3)

This plan details the creation of a "AI Diagnostic Assistant" for Project 2, Task 3. This is a multi-tool application.
* **Tool 1 (Symptom Checker):** Uses RAG + ML to diagnose symptoms.
* **Tool 2 (Cancer Analysis Finder):** Uses a "precontext" LLM call to answer questions about Task 2 findings.

## Phase 0: Environment Setup

**Goal:** Prepare the repository and environment for Task 3.

1.  **Update Dependencies:** Modify `requirements.txt`. Add the following new dependencies (check if they exist before adding):
    * `chromadb==0.5.3`
    * `sentence-transformers==3.0.1`
    * `google-generativeai==0.7.2`
    * `python-dotenv==1.0.1`
2.  **Install Dependencies:** Run `pip install -r requirements.txt`.
3.  **Create Output Directories:** In `outputs/`, create:
    * `outputs/models/`
    * `outputs/vectorstore/` (Note: This will *only* be for the Symptom tool)
4.  **Create Notebook:** In `notebooks/task3/`, create `task3_diagnostic_assistant.ipynb`.

## Phase 1: ML Model Training & Vocabulary Export (Tool 1)

**Goal:** Train the ML classifier for the Symptom Checker and export its vocabulary.
**Location:** `notebooks/task3/task3_diagnostic_assistant.ipynb`.

1.  **Load Data:** Load `data/dataset.csv`.
2.  **Preprocess:**
    * Combine `Symptom_1`...`Symptom_17` into a single `symptoms_text` column.
    * Define `X = df['symptoms_text']` and `y = df['Disease']`.
3.  **Define Pipeline:**
    * Create a `Pipeline` using `CountVectorizer` and `RandomForestClassifier(n_estimators=100, random_state=42)`.
4.  **Train & Evaluate:** Split data, fit the `disease_pipeline`, and print accuracy.
5.  **Export Vocabulary:**
    * `vocabulary = disease_pipeline.named_steps['vectorizer'].get_feature_names_out()`
    * Save `vocabulary` to `outputs/models/symptom_vocabulary.json`.
6.  **Save Model:** Save the trained `disease_pipeline` to `outputs/models/disease_model.pkl`.

## Phase 2: Knowledge Base Setup (Tool 1)

**Goal:** Build the ChromaDB vectorstore for the Symptom Checker.
**Location:** `notebooks/task3/task3_diagnostic_assistant.ipynb`.

1.  **Load Text Data:** Load `data/symptom_Description.csv` and `data/symptom_precaution.csv`.
2.  **Process Documents:** Merge precautions and descriptions into a single doc per disease.
3.  **Initialize Embeddings:** Load `embedding_model = SentenceTransformer('all-MiniLM-L6-v2')`.
4.  **Initialize ChromaDB:**
    * `client = chromadb.PersistentClient(path="outputs/vectorstore/chroma_db")`
    * `collection_symptoms = client.get_or_create_collection(name="disease_info")`
5.  **Populate Vectorstore:** Loop through disease documents, embed them, and add to `collection_symptoms` with `metadata={"disease": "Influenza"}`.

## Phase 3: The Multi-Tool AI Orchestrator Class

**Goal:** Create the "brain" of the application.
**Location:** Create a new file: `scripts/task3_app.py`.

1.  **Imports:** Import `joblib`, `json`, `chromadb`, `sentence_transformers`, `google.generativeai`, `os`, and `dotenv`.
2.  **Define Class:** Create a class named `ProjectAssistant`.
3.  **`__init__` Method:**
    * `dotenv.load_dotenv()`
    * **Load API Key:** Load `GOOGLE_API_KEY` from `os.environ` and configure `genai`.
    * **Load LLM:** `self.llm = genai.GenerativeModel('gemini-2.5-flash')`
    * **Load Tool 1 (Symptoms):**
        * `self.symptom_model = joblib.load('outputs/models/disease_model.pkl')`
        * `with open('outputs/models/symptom_vocabulary.json') as f: self.symptom_vocabulary = json.load(f)`
        * `self.db_client = chromadb.PersistentClient(...)`
        * `self.collection_symptoms = self.db_client.get_collection("disease_info")`
        * `self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')`
    * **Load Tool 2 (Cancer):**
        * `with open('outputs/analysis_summary.txt') as f: self.cancer_summary = f.read()`
        * `with open('outputs/feature_importance.txt') as f: self.cancer_features = f.read()`
        * `self.cancer_context = self.cancer_summary + "\n\n" + self.cancer_features`

4.  **Method: `_route_query(self, user_query)`:**
    * Create a prompt to classify the user's intent.
    * **Prompt Template:**
        ```
        You are a query router. Classify the query into one of two categories:
        1. 'symptom_check': The user is describing personal health symptoms.
        2. 'cancer_analysis': The user is asking for analysis, patterns, or features from the breast cancer study.
        Return *only* the category name.

        Query: "{user_query}"
        Category:
        ```
    * Call `self.llm.generate_content(prompt)` and return the cleaned text (e.g., `"symptom_check"`).

5.  **Method: `_run_symptom_checker(self, user_query)`:**
    * (This method contains the *entire* RAG+ML logic from the previous plan)
    * **Step 1: Extract Symptoms:** Call the LLM with a clear prompt to extract symptom phrases from `user_query` and map each phrase to the closest token(s) in `self.symptom_vocabulary`. The prompt should instruct the LLM to:
        * Identify symptom phrases in the user's natural language query
        * Map each phrase to the closest matching token(s) from the vocabulary
        * Return a normalized list of symptom tokens
        * Include any unmatched terms in a separate list for potential clarification
        * **Prompt Template for Extraction:**
            ```
            You are a medical symptom extraction assistant. Extract symptom phrases from the user's query and map them to the closest matching tokens from the provided vocabulary.

            **Available Symptom Vocabulary:** {self.symptom_vocabulary}

            **User Query:** {user_query}

            **Instructions:**
            1. Identify all symptom-related phrases in the user query
            2. For each phrase, find the closest matching token(s) from the vocabulary
            3. Return a normalized list of symptom tokens (one per line)
            4. If a phrase doesn't match any vocabulary term, include it in an "unmatched" section

            **Response Format:**
            Matched Symptoms:
            - [symptom_token_1]
            - [symptom_token_2]
            ...

            Unmatched Terms:
            - [term_1]
            ...
            ```
        * Parse the LLM response to extract a clean `symptoms_list` (normalized symptom tokens)
    * **Step 2: Handle Empty Symptoms:** If `symptoms_list` is empty or contains no valid vocabulary tokens, return a clarification request to the user:
        * `return "I couldn't identify any recognizable symptoms in your message. Could you please describe your symptoms in more detail? For example: 'I have a cough and fever.'"`
    * **Step 3: Predict Disease and Calculate Confidence:**
        * Vectorize the `symptoms_list` using the same vectorizer from the trained pipeline (or convert symptoms to a format compatible with the model's input)
        * Call `self.symptom_model.predict_proba(symptoms_list_vectorized)` to get class probabilities for all diseases
        * Get the predicted class by calling `self.symptom_model.predict(symptoms_list_vectorized)` or by using `probabilities.argmax()` to get the index of the highest probability
        * Extract the confidence score as the probability of the predicted class: `confidence = probabilities.max()` or `confidence = probabilities[predicted_index]`
        * **Confidence Threshold Handling:**
            * Define a confidence threshold (e.g., `CONFIDENCE_THRESHOLD = 0.5`)
            * If `confidence < CONFIDENCE_THRESHOLD`, mark this as a low-confidence prediction
            * For low-confidence cases, consider returning top-K diseases (e.g., top-3) with their probabilities to allow for disambiguation
            * Store `disease` (or `top_diseases` list) and `confidence` (or `confidence_scores` list) for use in subsequent steps
    * **Step 4: Retrieve Supporting Context:** 
        * Query `self.collection_symptoms.query(query_texts=[disease], where={"disease": disease}, n_results=3)` to retrieve supporting context documents (use `disease` from Step 3, or query for multiple diseases if using top-K approach). Alternatively, if you only need to fetch documents by metadata without semantic search, use `self.collection_symptoms.get(where={"disease": disease})`.
        * The `query()` method returns results with `ids`, `documents`, and `metadatas`; extract the `documents` from the results. The `get()` method returns results with the same structure.
        * If multiple diseases were identified, query for each: `contexts = []` then loop through diseases using `for disease in top_diseases: results = collection_symptoms.query(..., where={"disease": disease}, ...); contexts.extend(results['documents'])`
        * Combine the retrieved text passages into a single context string for use in the final prompt
    * **Step 5: Construct Final LLM Prompt:**
        * Build a complete prompt template that includes:
            * Normalized `symptoms_list` extracted from Step 1
            * Predicted disease(s) and confidence score(s) from Step 3
            * Retrieved context passages from Step 4
            * Relevant patient/context info if applicable
            * Instructions for answer structure (see template below)
            * The required medical disclaimer (see disclaimer below)
        * **Final Prompt Template:**
            ```
            You are a medical information assistant. You are helping a user understand their symptoms and possible conditions based on a machine learning prediction and supporting medical information. You must NOT provide definitive diagnoses or medical advice.

            **User's Reported Symptoms:**
            {symptoms_list}

            **ML Model Prediction:**
            Predicted Condition: {disease}
            Confidence Score: {confidence:.2%}

            {If low confidence, add:}
            Note: The model's confidence is below the threshold. Other possible conditions:
            {top_k_diseases_with_probabilities}

            **Supporting Medical Information:**
            {retrieved_context}

            **Instructions for Your Response:**
            1. **Concise Differential:** Provide a brief, non-definitive explanation of what the predicted condition might involve, based on the symptoms and context provided
            2. **Next Steps:** Suggest appropriate, non-prescriptive actions the user could consider (e.g., "Consider consulting with a healthcare provider", "Monitor symptoms and track changes")
            3. **Red Flags:** List any warning signs that would warrant immediate medical attention
            4. **When to Seek Urgent Care:** Clearly state situations that require emergency medical care

            **Critical Requirements:**
            - Do NOT state that you are diagnosing the user
            - Do NOT prescribe treatments or medications
            - Emphasize uncertainty, especially if confidence is low
            - Always recommend consulting with a qualified healthcare professional
            - Cite the supporting information when relevant
            - Make it clear that ML predictions are probabilistic and should be verified by medical professionals

            **Your Response:**
            ```
        * **Medical Disclaimer (to be appended verbatim):**
            ```
            {MEDICAL_DISCLAIMER}
            ```
            * Store the medical disclaimer as a class constant or module-level constant:
                ```python
                MEDICAL_DISCLAIMER = """DISCLAIMER: This AI assistant provides informational support only and does not constitute medical advice, diagnosis, or treatment. The predictions and information provided are based on machine learning models and general medical knowledge, which may not be accurate, complete, or applicable to your specific situation. Always consult with a qualified healthcare professional for proper diagnosis, treatment, and personalized medical advice. Do not disregard professional medical advice or delay seeking it because of information provided by this system. In case of a medical emergency, call emergency services immediately."""
            ```
    * **Step 6: Generate and Return Response:**
        * Call `self.llm.generate_content(final_prompt)` to generate the user-facing response
        * Ensure the response includes the medical disclaimer (either appended programmatically or included in the prompt so the LLM outputs it)
        * If confidence was low in Step 3, ensure the response prompts for clarification or explicitly recommends seeing a clinician for proper evaluation
        * Return the final generated string

6.  **Method: `_run_cancer_analysis(self, user_query)`:**
    * (This is the new, simple "precontext" method)
    * Create the final prompt for the LLM.
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
    * `intent = self._route_query(user_query)`
    * `if intent == 'symptom_check':`
        * `return self._run_symptom_checker(user_query)`
    * `elif intent == 'cancer_analysis':`
        * `return self._run_cancer_analysis(user_query)`
    * `else:`
        * `return "I'm sorry, I can only assist with symptom checks or questions about our cancer analysis findings. How can I help?"`

## Phase 4: Demonstration

**Goal:** Test the complete multi-tool system.
**Location:** `notebooks/task3/task3_diagnostic_assistant.ipynb`.

1.  **Import Class:** From `scripts.task3_app`, import `ProjectAssistant`.
2.  **Initialize:** `assistant = ProjectAssistant()`
3.  **Run Demos:**
    * **Test Tool 1:** `print(assistant.run("I have a bad cough, a high fever, and my whole body aches."))`
    * **Test Tool 2:** `print(assistant.run("What are the most discriminative patterns for benign tumors?"))`
    * **Test Tool 2:** `print(assistant.run("Which features are most important in malignant patterns?"))`
    * **Test Router:** `print(assistant.run("Hello, how are you?"))`
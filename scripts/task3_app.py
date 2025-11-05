"""
Multi-tool AI diagnostic assistant orchestrating symptom checker and cancer analysis tools.

This module implements the ProjectAssistant class, which provides a unified interface
for two AI-powered tools:
- Tool 1 (Symptom Checker): A RAG pipeline combining ML prediction with ChromaDB retrieval
  and LLM generation for symptom-based disease diagnosis
- Tool 2 (Cancer Analysis): A precontext LLM tool using cancer pattern mining findings
  from Task 2

Usage:
    from scripts.task3_app import ProjectAssistant
    
    assistant = ProjectAssistant()
    response = assistant.run("I have fever and cough")
    print(response)

Prerequisites:
    - Phase 1-2 completion (ML model, ChromaDB vectorstore, cancer analysis outputs)
    - .env file with GOOGLE_API_KEY in project root
    - All dependencies from requirements.txt installed

Reference:
    See docs/task3_implementation_plan.md Phase 3 for detailed implementation guide.
"""
import os
import sys
import json
import logging
import time
import random
import traceback
import re
from pathlib import Path

# Third-party imports
from dotenv import load_dotenv
import joblib
import chromadb
from sentence_transformers import SentenceTransformer
import google.genai as genai
from google.api_core import exceptions as google_exceptions

# Configure logging
logger = logging.getLogger(__name__)
# Ensure logger has a handler if used as a library (basicConfig only runs in __main__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Constants
MAX_RETRIES = 3
BASE_DELAY = 1  # Base delay in seconds for exponential backoff
GEMINI_MODEL = 'models/gemini-2.5-flash'
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
CHROMA_COLLECTION_NAME = 'disease_info'


class ProjectAssistant:
    """
    Multi-tool AI diagnostic assistant orchestrating symptom checker and cancer analysis.
    
    This class provides a unified interface for two AI-powered tools:
    - Tool 1 (Symptom Checker): RAG + ML pipeline for symptom-based disease prediction
    - Tool 2 (Cancer Analysis): Precontext LLM tool using cancer pattern mining findings
    
    Attributes:
        client (genai.Client): Initialized Gemini API client
        symptom_model: Trained ML model for disease prediction
        symptom_vocabulary (list): List of valid symptom tokens (134 items)
        db_client (chromadb.PersistentClient): ChromaDB client instance
        collection_symptoms (chromadb.Collection): ChromaDB collection for disease info
        embedding_model (SentenceTransformer): Embedding model for ChromaDB queries
        cancer_context (str): Combined cancer analysis context from Task 2 outputs
    
    Example:
        >>> assistant = ProjectAssistant()
        >>> response = assistant.run("I have fever and cough")
        >>> print(response)
    """
    
    def __init__(self):
        """
        Initialize all resources with robust error handling and path validation.
        
        Raises:
            FileNotFoundError: If required directories or files are missing
            ValueError: If GOOGLE_API_KEY is missing or invalid
            Exception: If initialization fails at any step
        """
        try:
            # Load environment variables
            project_root = Path(__file__).resolve().parent.parent
            env_path = project_root / '.env'
            if env_path.exists():
                load_dotenv(env_path)
                logger.info(f"Loaded .env file from {env_path}")
            else:
                load_dotenv()  # Try loading from default locations
                logger.warning(f".env file not found at {env_path}, trying default locations")
            
            # Path resolution and validation
            self.project_root = project_root
            
            # Define required paths
            models_dir = self.project_root / 'outputs' / 'models'
            vectorstore_dir = self.project_root / 'outputs' / 'vectorstore'
            model_path = models_dir / 'disease_model.pkl'
            vocab_path = models_dir / 'symptom_vocabulary.json'
            chroma_path = vectorstore_dir / 'chroma_db'
            cancer_summary_path = self.project_root / 'outputs' / 'analysis_summary.txt'
            cancer_features_path = self.project_root / 'outputs' / 'feature_importance.txt'
            
            # Validate all paths exist
            required_paths = {
                'models directory': models_dir,
                'vectorstore directory': vectorstore_dir,
                'disease model': model_path,
                'symptom vocabulary': vocab_path,
                'chroma database': chroma_path,
                'cancer summary': cancer_summary_path,
                'cancer features': cancer_features_path
            }
            
            missing_paths = []
            for name, path in required_paths.items():
                if not path.exists():
                    missing_paths.append(f"{name}: {path}")
            
            if missing_paths:
                error_msg = (
                    "Required files or directories are missing:\n" +
                    "\n".join(f"  - {path}" for path in missing_paths) +
                    "\n\nPlease ensure Phase 1-2 are completed before initializing ProjectAssistant."
                )
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            
            logger.info("All required paths validated successfully")
            for name, path in required_paths.items():
                logger.debug(f"  {name}: {path}")
            
            # Load API key and configure Gemini
            api_key = os.environ.get('GOOGLE_API_KEY')
            if not api_key or api_key.strip() == '':
                error_msg = (
                    "GOOGLE_API_KEY environment variable is required but not set. "
                    "Please set it in your .env file."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Configure Gemini with retry logic
            logger.info("Configuring Gemini API...")
            try:
                self.client = genai.Client(api_key=api_key)
                logger.info(f"Gemini API configured successfully with model {GEMINI_MODEL}")
            except ValueError as e:
                # Invalid API key - fail fast
                logger.error(f"Configuration error: {e}")
                raise
            except (ConnectionError, TimeoutError, google_exceptions.ServiceUnavailable) as e:
                # Transient error - retry with exponential backoff
                logger.warning(f"Transient error during Gemini initialization: {e}. Retrying...")
                for attempt in range(MAX_RETRIES):
                    try:
                        delay = BASE_DELAY * (2 ** attempt) + random.uniform(0, 1)
                        logger.debug(f"Retry attempt {attempt + 1}/{MAX_RETRIES} after {delay:.2f}s delay")
                        time.sleep(delay)
                        self.client = genai.Client(api_key=api_key)
                        logger.info(f"Gemini API configured successfully on retry {attempt + 1}")
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
                logger.error(traceback.format_exc())
                raise
            
            # Load Tool 1 resources (Symptom Checker)
            logger.info("Loading Tool 1 resources (Symptom Checker)...")
            self.symptom_model = joblib.load(model_path)
            logger.info(f"Loaded ML model from {model_path}")
            
            with open(vocab_path, 'r') as f:
                self.symptom_vocabulary = json.load(f)
            
            if not isinstance(self.symptom_vocabulary, list) or len(self.symptom_vocabulary) == 0:
                raise ValueError(f"Symptom vocabulary must be a non-empty list. Got: {type(self.symptom_vocabulary)}")
            
            logger.info(f"Loaded symptom vocabulary with {len(self.symptom_vocabulary)} tokens")
            
            self.db_client = chromadb.PersistentClient(path=str(chroma_path))
            
            # Initialize embedding model first (needed for collection retrieval)
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            logger.info(f"Initialized embedding model: {EMBEDDING_MODEL_NAME}")
            
            # Get collection with graceful handling for missing collections
            try:
                self.collection_symptoms = self.db_client.get_collection(CHROMA_COLLECTION_NAME)
            except (ValueError, AttributeError) as e:
                # ChromaDB may raise ValueError or AttributeError for missing collections
                # Check if the error message indicates collection not found
                error_str = str(e).lower()
                if 'not found' in error_str or 'collection' in error_str or 'does not exist' in error_str:
                    error_msg = (
                        f"ChromaDB collection '{CHROMA_COLLECTION_NAME}' not found. "
                        "Please run Phase 2 to build the vectorstore."
                    )
                    logger.error(error_msg)
                    raise FileNotFoundError(error_msg) from e
                else:
                    # Re-raise if it's a different ValueError
                    raise
            except Exception as e:
                # Catch any other ChromaDB exceptions (e.g., chromadb.errors.InvalidCollectionException if available)
                error_str = str(e).lower()
                if 'not found' in error_str or 'collection' in error_str or 'does not exist' in error_str:
                    error_msg = (
                        f"ChromaDB collection '{CHROMA_COLLECTION_NAME}' not found. "
                        "Please run Phase 2 to build the vectorstore."
                    )
                    logger.error(f"{error_msg}: {e}")
                    raise FileNotFoundError(error_msg) from e
                else:
                    # Re-raise if it's an unexpected error
                    raise
            
            doc_count = self.collection_symptoms.count()
            if doc_count == 0:
                raise ValueError(f"ChromaDB collection '{CHROMA_COLLECTION_NAME}' is empty. Please run Phase 2.")
            
            logger.info(f"Connected to ChromaDB collection '{CHROMA_COLLECTION_NAME}' with {doc_count} documents")
            logger.info("Tool 1 (Symptom Checker) initialization complete")
            
            # Load Tool 2 resources (Cancer Analysis)
            logger.info("Loading Tool 2 resources (Cancer Analysis)...")
            with open(cancer_summary_path, 'r') as f:
                self.cancer_summary = f.read()
            
            with open(cancer_features_path, 'r') as f:
                self.cancer_features = f.read()
            
            self.cancer_context = self.cancer_summary + '\n\n' + self.cancer_features
            
            if not self.cancer_context or len(self.cancer_context.strip()) == 0:
                raise ValueError("Cancer context is empty. Please ensure Phase 0-2 outputs are generated.")
            
            logger.info(f"Loaded cancer context ({len(self.cancer_context)} characters)")
            logger.info("Tool 2 (Cancer Analysis) initialization complete")
            
            logger.info("ProjectAssistant initialized successfully")
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _retry_llm_call(self, func, max_retries: int = 3):
        """
        Helper method to retry LLM API calls with exponential backoff for transient errors.
        
        This method is specifically designed for LLM API calls (e.g., client.models.generate_content).
        It handles transient errors like ConnectionError, TimeoutError, and ServiceUnavailable.
        
        Args:
            func: Callable function that returns a GenAI response or string (LLM API call)
            max_retries: Maximum number of retry attempts (default: 3)
            
        Returns:
            Result of function call (typically a GenAI response object)
            
        Raises:
            Exception: Re-raises non-transient exceptions or final exception after retries
        """
        for attempt in range(max_retries):
            try:
                return func()
            except (ConnectionError, TimeoutError, google_exceptions.ServiceUnavailable) as e:
                # Transient error - retry with exponential backoff
                if attempt < max_retries - 1:
                    delay = BASE_DELAY * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(
                        f"Transient error during LLM call (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying after {delay:.2f}s..."
                    )
                    time.sleep(delay)
                else:
                    # Final attempt failed
                    logger.error(f"Failed after {max_retries} attempts: {e}")
                    raise
            except (google_exceptions.InvalidArgument, ValueError) as e:
                # Non-transient error - fail fast
                logger.error(f"Non-transient error (invalid request): {e}")
                raise
            except Exception as e:
                # Other errors - log and re-raise (don't retry)
                logger.error(f"Unexpected error during LLM call: {e}")
                raise
    
    def _route_query(self, user_query: str) -> str:
        """
        Classify user intent using LLM to route to appropriate tool.
        
        Args:
            user_query: User's input query string
            
        Returns:
            Intent classification: 'symptom_check', 'cancer_analysis', or 'unknown'
        """
        try:
            # Create classification prompt
            prompt = f"""You are a query router. Classify the query into one of two categories:
1. 'symptom_check': The user is describing personal health symptoms.
2. 'cancer_analysis': The user is asking for analysis, patterns, or features from the breast cancer study.
Return *only* the category name.

Query: "{user_query}"
Category:
"""
            
            # Call LLM with retry
            def generate():
                response = self.client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
                return response.text.strip().lower()
            
            intent = self._retry_llm_call(generate)
            
            logger.info(f"Query routed to: {intent}")
            return intent
            
        except Exception as e:
            logger.error(f"Error during query routing: {e}")
            logger.error(traceback.format_exc())
            return 'unknown'
    
    def _run_symptom_checker(self, user_query: str) -> str:
        """
        Execute 5-step RAG pipeline for symptom-based disease prediction.
        
        Steps:
        1. Extract symptoms using vocabulary constraint
        2. Validate extracted symptoms
        3. Predict disease using ML model
        4. Retrieve disease information from ChromaDB
        5. Generate final response with LLM
        
        Args:
            user_query: User's symptom description
            
        Returns:
            Generated response with medical disclaimer
        """
        try:
            # Step 1: Extract symptoms using vocabulary constraint
            vocabulary_list = ', '.join(self.symptom_vocabulary)
            extraction_prompt = f"""You are a medical symptom extractor. Extract ONLY the symptoms mentioned in the user's query.
You must ONLY use symptoms from this vocabulary list: {vocabulary_list}

Return a JSON array of symptom names, e.g., ["fever", "cough", "headache"]
If no valid symptoms found, return an empty array: []

User query: "{user_query}"
Extracted symptoms:
"""
            
            # Call LLM with retry
            def generate_extraction():
                return self.client.models.generate_content(model=GEMINI_MODEL, contents=extraction_prompt)
            
            extraction_response = self._retry_llm_call(generate_extraction)
            
            # Parse response as JSON - sanitize to handle code fences and prose
            response_text = extraction_response.text.strip()
            
            # Remove code fences (```json, ```, etc.)
            response_text = re.sub(r'^```(?:json)?\s*', '', response_text, flags=re.MULTILINE)
            response_text = re.sub(r'\s*```$', '', response_text, flags=re.MULTILINE)
            response_text = response_text.strip()
            
            # Extract JSON array from response (find first [ and matching ])
            json_match = None
            bracket_count = 0
            start_idx = -1
            
            for i, char in enumerate(response_text):
                if char == '[':
                    if start_idx == -1:
                        start_idx = i
                    bracket_count += 1
                elif char == ']':
                    bracket_count -= 1
                    if bracket_count == 0 and start_idx != -1:
                        json_match = response_text[start_idx:i+1]
                        break
            
            # Parse JSON
            try:
                if json_match:
                    symptoms_list = json.loads(json_match)
                else:
                    # Fallback: try parsing entire response
                    symptoms_list = json.loads(response_text)
                
                if not isinstance(symptoms_list, list):
                    symptoms_list = []
            except (json.JSONDecodeError, AttributeError) as e:
                logger.warning(f"Failed to parse symptoms as JSON: {e}. Using empty list.")
                symptoms_list = []
            
            logger.info(f"Extracted symptoms: {symptoms_list}")
            
            # Step 2: Validate extracted symptoms
            if not symptoms_list or len(symptoms_list) == 0:
                return (
                    "I couldn't identify any valid symptoms from our database in your query. "
                    "Please describe your symptoms using common medical terms like 'fever', 'cough', "
                    "'headache', etc."
                )
            
            # Filter symptoms to ensure they're in vocabulary
            valid_symptoms = [s for s in symptoms_list if s in self.symptom_vocabulary]
            
            if not valid_symptoms or len(valid_symptoms) == 0:
                return (
                    "I couldn't identify any valid symptoms from our database in your query. "
                    "Please describe your symptoms using common medical terms like 'fever', 'cough', "
                    "'headache', etc."
                )
            
            logger.info(f"Validated symptoms: {valid_symptoms}")
            
            # Step 3: Predict disease using ML model
            symptoms_text = ' '.join(valid_symptoms)
            prediction = self.symptom_model.predict([symptoms_text])[0]
            probabilities = self.symptom_model.predict_proba([symptoms_text])[0]
            confidence = max(probabilities)
            disease = prediction
            
            logger.info(f"Predicted disease: {disease} (confidence: {confidence:.2%})")
            
            # Step 4: Retrieve disease information from ChromaDB
            try:
                # Use query_texts to let the collection's embedding function handle embeddings
                # This ensures consistency with the embedding model used at index time
                results = self.collection_symptoms.query(
                    query_texts=[disease],
                    n_results=1,
                    where={"disease": disease}
                )
                
                if results['documents'] and len(results['documents'][0]) > 0:
                    disease_info = results['documents'][0][0]
                    logger.info(f"Retrieved disease info from ChromaDB for: {disease}")
                else:
                    disease_info = f"Disease: {disease} (No additional information available)"
                    logger.warning(f"No ChromaDB results found for disease: {disease}")
            except Exception as e:
                logger.error(f"Error querying ChromaDB: {e}")
                disease_info = f"Disease: {disease} (No additional information available)"
            
            # Step 5: Generate final response with LLM
            generation_prompt = f"""You are a medical AI assistant. Based on the symptoms and disease prediction, provide a helpful response.

User's symptoms: {valid_symptoms}
Predicted disease: {disease}
Confidence: {confidence:.2%}

Disease information from knowledge base:
{disease_info}

Provide a clear, empathetic response that:
1. Acknowledges the symptoms
2. Explains the predicted condition
3. Includes relevant information from the knowledge base
4. Emphasizes this is AI-generated and not a substitute for professional medical advice

IMPORTANT: End your response with this medical disclaimer:
"⚠️ Medical Disclaimer: This is an AI-generated assessment based on limited information. Please consult a qualified healthcare professional for proper diagnosis and treatment. Do not use this as a substitute for professional medical advice."

Response:
"""
            
            # Call LLM with retry
            def generate_response():
                return self.client.models.generate_content(model=GEMINI_MODEL, contents=generation_prompt)
            
            generation_response = self._retry_llm_call(generate_response)
            final_response = generation_response.text.strip()
            
            logger.info("Generated final symptom checker response")
            return final_response
            
        except Exception as e:
            logger.error(f"Error in symptom checker: {e}")
            logger.error(traceback.format_exc())
            return (
                "I encountered an error processing your symptom query. "
                "Please try again or rephrase your symptoms."
            )
    
    def _run_cancer_analysis(self, user_query: str) -> str:
        """
        Answer cancer analysis questions using preloaded context.
        
        Args:
            user_query: User's question about cancer analysis
            
        Returns:
            LLM-generated response based on cancer context
        """
        try:
            # Create precontext prompt
            prompt = f"""You are an AI assistant for a biomedical researcher. Answer the user's question using *only* the provided context, which contains the key findings from our Task 2 analysis.

**Provided Context:**

{self.cancer_context}

**User Question:**

{user_query}

**Answer:**

"""
            
            # Call LLM with retry
            def generate():
                return self.client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            
            response = self._retry_llm_call(generate)
            answer = response.text.strip()
            
            logger.info("Generated cancer analysis response")
            return answer
            
        except Exception as e:
            logger.error(f"Error in cancer analysis: {e}")
            logger.error(traceback.format_exc())
            return (
                "I encountered an error accessing the cancer analysis findings. "
                "Please try again."
            )
    
    def run(self, user_query: str) -> str:
        """
        Main orchestrator method that routes queries and returns responses.
        
        Args:
            user_query: User's input query string
            
        Returns:
            Response string from appropriate tool or error message
        """
        try:
            # Input validation
            if not user_query or not user_query.strip():
                return "Please provide a valid query."
            
            user_query = user_query.strip()
            
            # Log incoming query (truncated if very long)
            query_preview = user_query[:100] + "..." if len(user_query) > 100 else user_query
            logger.info(f"Processing query: {query_preview}")
            
            # Route query
            intent = self._route_query(user_query)
            intent = intent.strip().lower()
            
            logger.info(f"Routing decision: {intent}")
            
            # Execute appropriate tool
            if intent == 'symptom_check':
                return self._run_symptom_checker(user_query)
            elif intent == 'cancer_analysis':
                return self._run_cancer_analysis(user_query)
            else:
                return (
                    "I'm sorry, I can only assist with symptom checks or questions about "
                    "our cancer analysis findings. For example: 'I have fever and cough' "
                    "or 'Which features are most important in malignant patterns?'"
                )
                
        except Exception as e:
            logger.error(f"Unexpected error in run() method: {e}")
            logger.error(traceback.format_exc())
            return "I encountered an unexpected error. Please try again."


if __name__ == '__main__':
    """Test the ProjectAssistant with sample queries."""
    # Configure logging only in main block to avoid global side effects
    if not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    try:
        print("=" * 60)
        print("Initializing ProjectAssistant...")
        print("=" * 60)
        assistant = ProjectAssistant()
        
        print("\n" + "=" * 60)
        print("Test 1: Symptom Check")
        print("=" * 60)
        query1 = "I have a bad cough, a high fever, and my whole body aches."
        print(f"Query: {query1}")
        print("\nResponse:")
        response1 = assistant.run(query1)
        print(response1)
        
        print("\n" + "=" * 60)
        print("Test 2: Cancer Analysis")
        print("=" * 60)
        query2 = "What are the most discriminative patterns for benign tumors?"
        print(f"Query: {query2}")
        print("\nResponse:")
        response2 = assistant.run(query2)
        print(response2)
        
        print("\n" + "=" * 60)
        print("Test 3: Out-of-Scope Query")
        print("=" * 60)
        query3 = "Hello, how are you?"
        print(f"Query: {query3}")
        print("\nResponse:")
        response3 = assistant.run(query3)
        print(response3)
        
        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        traceback.print_exc()
        sys.exit(1)


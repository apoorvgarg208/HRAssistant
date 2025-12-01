"""
AI-Powered HR Assistant with RAG (Retrieval Augmented Generation)
Answers questions about ACME HR policies using semantic search
"""

import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
import os
from typing import List, Dict, Optional
import re
import pickle
import hashlib
from pathlib import Path


class HRPolicyAssistant:
    """AI-powered assistant for HR policy questions with optimized RAG pipeline"""
    
    def __init__(self, policy_file_path: str, cache_dir: str = ".cache"):
        self.policy_file_path = policy_file_path
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.vectorstore = None
        self.qa_chain = None
        self.embeddings = None
        self._answer_cache = {}  # Cache for frequent questions
        self.setup_assistant()
    
    def load_policy_document(self) -> str:
        """Load HR policy document efficiently"""
        with open(self.policy_file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _get_document_hash(self, text: str) -> str:
        """Generate hash for document to detect changes"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _get_cache_path(self, doc_hash: str) -> Path:
        """Get path for cached vectorstore"""
        return self.cache_dir / f"vectorstore_{doc_hash}.pkl"
    
    def setup_assistant(self):
        """Setup the RAG pipeline with caching for scalability"""
        # Load policy document
        policy_text = self.load_policy_document()
        doc_hash = self._get_document_hash(policy_text)
        cache_path = self._get_cache_path(doc_hash)
        
        # Try to load from cache first (massive speedup for large documents)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.vectorstore = cached_data['vectorstore']
                    self.embeddings = cached_data['embeddings']
                    print(f"Loaded vectorstore from cache ({len(cached_data.get('chunk_count', 0))} chunks)")
                    self._setup_retriever()
                    return
            except Exception as e:
                print(f"Cache load failed: {e}. Rebuilding...")
        
        # Optimized text splitting for large documents
        # Paragraph-based chunking with semantic boundaries
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,  # Increased for better context
            chunk_overlap=100,  # Increased overlap for continuity
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ": ", " ", ""],  # Sentence-aware
            keep_separator=True  # Maintain document structure
        )
        
        chunks = text_splitter.split_text(policy_text)
        print(f"Document split into {len(chunks)} chunks")
        
        # Create embeddings with optimized settings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={
                'device': 'cpu'
            },
            encode_kwargs={
                'batch_size': 32,  # Batch processing for large documents
                'normalize_embeddings': True  # Faster cosine similarity
            }
        )
        
        # Create vector store with batch embedding (efficient for 100+ chunks)
        print("Creating vector embeddings...")
        self.vectorstore = FAISS.from_texts(
            chunks, 
            self.embeddings,
            metadatas=[{"chunk_id": i, "chunk_size": len(chunk)} for i, chunk in enumerate(chunks)]
        )
        
        # Cache the vectorstore for future use
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'vectorstore': self.vectorstore,
                    'embeddings': self.embeddings,
                    'chunk_count': len(chunks)
                }, f)
            print(f"Vectorstore cached at {cache_path}")
        except Exception as e:
            print(f"Failed to cache vectorstore: {e}")
        
        self._setup_retriever()
    
    def _setup_retriever(self):
        """Setup retriever with optimized search parameters"""
        # Setup retriever with MMR for diversity (better for large document sets)
        retriever = self.vectorstore.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance for diverse results
            search_kwargs={
                "k": 4,  # Retrieve top 4 chunks
                "fetch_k": 20,  # Fetch 20 candidates before MMR
                "lambda_mult": 0.7  # Balance relevance vs diversity
            }
        )
        
        # Create prompt template
        prompt_template = """You are ACME Corp's helpful HR assistant. Use the following HR policy information to answer the question accurately and professionally.

Context from HR Policy:
{context}

Question: {question}

Instructions:
- Provide accurate information based on the policy context
- Be professional and friendly
- If the information is not in the context, say so politely
- Include specific details like numbers of days, amounts, etc.
- Format your answer clearly

Answer:"""
        
        PROMPT = PromptTemplate.from_template(prompt_template)
        
        # Setup LLM - Using a simple approach that works without API keys
        # For production, replace with proper LLM
        self.retriever = retriever
        self.prompt = PROMPT
    
    def answer_question(self, question: str) -> Dict[str, any]:
        """Answer a question using RAG with caching for performance"""
        # Check cache first (instant response for repeated questions)
        question_hash = hashlib.md5(question.lower().strip().encode()).hexdigest()
        if question_hash in self._answer_cache:
            return self._answer_cache[question_hash]
        
        try:
            # Use similarity search (more reliable than relevance scores)
            docs_with_scores = self.vectorstore.similarity_search_with_score(
                question, 
                k=4  # Retrieve top 4
            )
            
            # Check if any relevant documents found
            if not docs_with_scores:
                result = {
                    "question": question,
                    "answer": "Sorry, I am unable to find a suitable response for you. Please contact the HR helpdesk at hr@acmecorp.com.",
                    "source_chunks": []
                }
                self._answer_cache[question_hash] = result
                return result
            
            # Get the best (minimum) distance score (L2 distance - lower is better)
            best_score = min(score for _, score in docs_with_scores)
            
            # If best match has high distance (> 2.0), likely out of context
            # Relaxed threshold to allow more matches
            if best_score > 2.0:
                result = {
                    "question": question,
                    "answer": "Sorry, I am unable to find a suitable response for you. Please contact the HR helpdesk at hr@acmecorp.com.",
                    "source_chunks": []
                }
                self._answer_cache[question_hash] = result
                return result
            
            # Extract documents (sorted by distance score - ascending for L2 distance)
            docs = [doc for doc, score in sorted(docs_with_scores, key=lambda x: x[1])]
            
            # Combine context with relevance-based ordering
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Generate answer using rule-based approach + context
            # In production, use actual LLM here (e.g., OpenAI, Anthropic, etc.)
            answer = self.generate_answer_from_context(question, context)
            
            result = {
                "question": question,
                "answer": answer,
                "source_chunks": [doc.page_content for doc in docs],
                "relevance_scores": [score for _, score in docs_with_scores]
            }
            
            # Cache the result (LRU-style cache with size limit)
            if len(self._answer_cache) > 100:  # Keep cache manageable
                # Remove oldest entry
                self._answer_cache.pop(next(iter(self._answer_cache)))
            self._answer_cache[question_hash] = result
            
            return result
        
        except Exception as e:
            error_result = {
                "question": question,
                "answer": f"I apologize, but I encountered an error: {str(e)}. Please try rephrasing your question.",
                "source_chunks": []
            }
            return error_result
    
    def generate_answer_from_context(self, question: str, context: str) -> str:
        """Generate answer from context (simplified for demo without LLM API)"""
        # This is a simplified version. In production, use actual LLM
        question_lower = question.lower()
        
        # Extract relevant information based on keywords
        if any(word in question_lower for word in ['leave', 'annual leave', 'vacation']):
            if 'annual' in question_lower:
                match = re.search(r'Annual Leave:.*?(\d+)\s*days', context, re.IGNORECASE)
                if match:
                    return f"According to ACME Corp HR Policy, employees are entitled to {match.group(1)} days of paid annual leave each calendar year."
        
        if any(word in question_lower for word in ['sick leave', 'sick', 'illness']):
            match = re.search(r'Sick Leave:.*?(\d+)\s*days', context, re.IGNORECASE)
            if match:
                return f"Employees are allowed up to {match.group(1)} days of sick leave annually as per company policy."
        
        if any(word in question_lower for word in ['maternity', 'maternity leave']):
            match = re.search(r'Maternity Leave:.*?(\d+)\s*weeks', context, re.IGNORECASE)
            if match:
                return f"Female employees are entitled to {match.group(1)} weeks of paid maternity leave."
        
        if any(word in question_lower for word in ['paternity', 'paternity leave']):
            match = re.search(r'Paternity Leave:.*?(\d+)\s*weeks', context, re.IGNORECASE)
            if match:
                return f"Male employees are entitled to {match.group(1)} weeks of paternity leave."
        
        if any(word in question_lower for word in ['working hours', 'work hours', 'office hours']):
            if 'standard working hours' in context.lower():
                return "Standard working hours at ACME Corp are 9:00 AM to 6:00 PM, Monday to Friday. Employees are allowed flexible start times between 8:00 AM and 10:00 AM."
        
        if any(word in question_lower for word in ['remote', 'work from home', 'wfh']):
            if 'remote work' in context.lower():
                match = re.search(r'remote work.*?(\d+)\s*days?\s*per\s*week', context, re.IGNORECASE)
                if match:
                    return f"Remote work is allowed up to {match.group(1)} days per week, pending manager approval."
        
        if any(word in question_lower for word in ['notice period', 'resignation', 'notice']):
            if 'notice period' in context.lower():
                return "The notice period is 30 days for junior employees and 60 days for mid-to-senior roles. HR will conduct an exit interview before final clearance."
        
        if any(word in question_lower for word in ['training', 'learning', 'development', 'l&d']):
            if 'learning' in context.lower() or 'development' in context.lower():
                return "Employees can enroll in up to 2 paid learning programs annually. Every employee has access to a personal development budget of INR 25,000 per year. Employees must complete a minimum of one L&D course every two quarters."
        
        if any(word in question_lower for word in ['performance', 'review', 'appraisal']):
            if 'performance' in context.lower():
                return "Annual performance reviews occur in April and October. Ratings are on a scale of 1 to 4, where 4 denotes 'Exceeds Expectations'. Promotions are considered during the April cycle."
        
        if any(word in question_lower for word in ['travel', 'reimbursement', 'expense']):
            if 'travel' in context.lower() or 'reimbursement' in context.lower():
                return "Travel reimbursements require submission of original bills and pre-approval from the reporting manager. Per diem for domestic travel is INR 1500/day. International rates vary by region. All expense claims must be submitted within 30 days of the transaction."
        
        if any(word in question_lower for word in ['health insurance', 'insurance', 'benefits']):
            if 'health insurance' in context.lower():
                return "Health Insurance is provided to all full-time employees. Additionally, employees have access to the Employee Assistance Program (EAP) for confidential counseling services."
        
        if any(word in question_lower for word in ['dress code', 'attire', 'clothing']):
            if 'dress code' in context.lower():
                return "Business casual attire is required on all working days except Fridays, which are designated as casual wear days. Client-facing roles may have specific dress code requirements outlined by their department head."
        
        if any(word in question_lower for word in ['internal transfer', 'transfer', 'job posting']):
            if 'internal transfer' in context.lower():
                return "Employees can apply for internal job postings after completing 12 months in their current role. Transfers are subject to business need, performance history, and leadership approval."
        
        if any(word in question_lower for word in ['grievance', 'complaint', 'issue']):
            if 'grievance' in context.lower():
                return "Concerns related to workplace behavior or policy violations can be submitted via the confidential grievance portal. Each case will be reviewed within 7 business days and an impartial HR representative will be assigned."
        
        # Default response with context
        return f"Based on ACME Corp HR Policy:\n\n{context}\n\nFor more specific information, please contact the HR helpdesk at hr@acmecorp.com."
    
    def get_policy_summary(self) -> List[Dict[str, str]]:
        """Get summary of main policy categories"""
        categories = [
            {
                "category": "Leave Policy",
                "details": "Annual Leave: 20 days | Sick Leave: 12 days | Maternity: 26 weeks | Paternity: 2 weeks"
            },
            {
                "category": "Working Hours",
                "details": "9:00 AM - 6:00 PM (Mon-Fri) | Flexible start: 8-10 AM | Remote: 2 days/week"
            },
            {
                "category": "Travel & Expenses",
                "details": "Domestic per diem: INR 1500/day | Submit claims within 30 days"
            },
            {
                "category": "Performance",
                "details": "Reviews: April & October | Scale: 1-4 | Promotions: April cycle"
            },
            {
                "category": "Learning & Development",
                "details": "2 paid programs/year | Budget: INR 25,000 | Min 1 course per 2 quarters"
            },
            {
                "category": "Benefits",
                "details": "Health Insurance for all | EAP counseling services available"
            },
            {
                "category": "Exit Policy",
                "details": "Notice: 30 days (junior) | 60 days (mid-senior) | Exit interview required"
            },
            {
                "category": "Dress Code",
                "details": "Business casual (Mon-Thu) | Casual Fridays"
            },
            {
                "category": "Internal Transfers",
                "details": "Apply after 12 months in current role"
            },
            {
                "category": "Grievance",
                "details": "Confidential portal available | Response within 7 business days"
            }
        ]
        
        return categories


def render_hr_assistant():
    """Render HR Assistant interface"""
    
    # Custom CSS for compact layout
    st.markdown("""
        <style>
        .block-container {padding-top: 0.5rem !important; padding-bottom: 0.5rem !important;}
        h1 {font-size: 1.6rem !important; margin-bottom: 0.2rem !important; margin-top: 0 !important;}
        h2 {font-size: 1.1rem !important; margin-bottom: 0.3rem !important; margin-top: 0.3rem !important;}
        h3 {font-size: 0.95rem !important; margin-bottom: 0.2rem !important; margin-top: 0 !important;}
        p {font-size: 0.85rem !important; margin-bottom: 0.3rem !important;}
        .stButton > button {font-size: 0.7rem !important; padding: 0.2rem 0.4rem !important; height: auto !important; margin-bottom: 0.2rem !important;}
        .stTextArea textarea {font-size: 0.85rem !important;}
        .stMarkdown {font-size: 0.85rem !important; margin-bottom: 0.3rem !important;}
        hr {margin: 0.3rem 0 !important;}
        div[data-testid="stExpander"] {font-size: 0.85rem !important; margin-bottom: 0.3rem !important;}
        .stAlert {padding: 0.4rem !important; font-size: 0.85rem !important; margin-bottom: 0.3rem !important;}
        div[data-testid="stHorizontalBlock"] {gap: 0.3rem !important;}
        div[data-testid="column"] {padding: 0 0.2rem !important;}
        </style>
    """, unsafe_allow_html=True)
    
    st.title("Ask HR Assistant")
    st.markdown("###### Ask me anything about ACME Corp HR policies!")
    
    # Initialize assistant
    if 'hr_assistant' not in st.session_state:
        with st.spinner("Initializing HR Assistant..."):
            st.session_state.hr_assistant = HRPolicyAssistant('acme_hr_policy.txt')
            st.session_state.chat_history = []
    
    assistant = st.session_state.hr_assistant
    
    # Display policy summary
    with st.expander("Quick Policy Reference", expanded=False):
        policy_summary = assistant.get_policy_summary()
        
        cols = st.columns(2)
        for idx, item in enumerate(policy_summary):
            with cols[idx % 2]:
                st.markdown(f"**{item['category']}**")
                st.info(item['details'])
    
    # Common questions
    st.markdown("**Common Questions**")
    
    common_questions = [
        "Annual leave days?",
        "Notice period?",
        "Work from home?",
        "Working hours?",
        "L&D budget?",
        "Sick leave?",
        "Submit grievance?",
        "Dress code?",
        "Performance reviews?",
        "Benefits?"
    ]
    
    full_questions = [
        "How many days of annual leave am I entitled to?",
        "What is the notice period for resignation?",
        "Can I work from home?",
        "What are the working hours?",
        "How much is the learning and development budget?",
        "What is the sick leave policy?",
        "How do I submit a grievance?",
        "What is the dress code?",
        "When do performance reviews happen?",
        "What benefits does the company provide?"
    ]
    
    # Initialize current_question if not exists
    if 'current_question' not in st.session_state:
        st.session_state.current_question = ""
    
    # Handle common question clicks (before form to set state)
    common_question_clicked = False
    cols = st.columns(5)
    for idx, question in enumerate(common_questions):
        with cols[idx % 5]:
            if st.button(question, key=f"common_q_{idx}", use_container_width=True):
                st.session_state.current_question = full_questions[idx]
                st.session_state.should_ask = True
                common_question_clicked = True
    
    # If common question was clicked, process immediately
    if common_question_clicked:
        st.rerun()
    
    # Chat interface
    st.markdown("**Ask Your Question**")
    
    # User input with form for Enter key support
    with st.form(key="question_form", clear_on_submit=False):
        user_question = st.text_area(
            "Type your question here:",
            value=st.session_state.current_question,
            height=70,
            placeholder="e.g., How many days of sick leave do I get? (Press Ctrl+Enter to submit)",
            key="question_input"
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            ask_button = st.form_submit_button("Ask", type="primary", use_container_width=True)
        with col2:
            # Clear button needs to be outside form
            pass
    
    # Clear button outside form
    if st.button("Clear History", use_container_width=True, key="clear_btn"):
        st.session_state.chat_history = []
        st.session_state.current_question = ""
        st.rerun()
    
    # Process question from Ask button or common question button
    should_process = (ask_button and user_question.strip()) or st.session_state.get('should_ask', False)
    
    if should_process:
        question_to_ask = user_question if user_question.strip() else st.session_state.current_question
        
        if question_to_ask.strip():
            with st.spinner("Thinking..."):
                # Get answer
                result = assistant.answer_question(question_to_ask)
                
                # Add to chat history
                st.session_state.chat_history.append(result)
                
                # Clear the question and flag
                st.session_state.current_question = ""
                st.session_state.should_ask = False
                st.rerun()
    
    # Display latest response only (most recent)
    if st.session_state.chat_history:
        st.markdown("**Latest Response**")
        
        latest_chat = st.session_state.chat_history[-1]
        st.markdown(f"**Q:** {latest_chat['question']}")
        st.success(latest_chat['answer'])
        
        # Show source chunks in expander
        if latest_chat.get('source_chunks'):
            with st.expander("View Source Context"):
                for i, chunk in enumerate(latest_chat['source_chunks'], 1):
                    st.markdown(f"**Source {i}:**")
                    st.text(chunk)
    
    st.divider()
    
    # Full conversation history in expander
    if len(st.session_state.chat_history) > 1:
        with st.expander(f"View Full Conversation History ({len(st.session_state.chat_history)} questions)"):
            for idx, chat in enumerate(reversed(st.session_state.chat_history)):
                st.markdown(f"**Q{len(st.session_state.chat_history) - idx}:** {chat['question']}")
                st.info(chat['answer'])
                st.markdown("---")
    
if __name__ == "__main__":
    render_hr_assistant()
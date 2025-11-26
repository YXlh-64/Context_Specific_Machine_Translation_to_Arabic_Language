"""
Streamlit Demo Application for RAG Translation Agent

This is an interactive web interface for the RAG-based translation system.

Usage:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import json
from datetime import datetime
import os

# Import our modules
from rag_agent import RAGTranslationAgent
from vector_db import VectorDBManager
from utils import load_config, setup_logging

# Page configuration
st.set_page_config(
    page_title="RAG Translation Agent",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .example-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .metric-card {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models(config_path="config.ini"):
    """Load models and vector database (cached)"""
    config = load_config(config_path)
    
    # Setup logging
    logs_dir = config.get("GENERAL", "logs_dir", fallback="./logs")
    setup_logging(logs_dir)
    
    # Initialize Vector DB
    db_type = config.get("VECTOR_DB", "db_type", fallback="chromadb")
    embedding_model = config.get("EMBEDDINGS", "model_name")
    db_path = config.get("GENERAL", "vector_db_dir", fallback="./vector_db")
    collection_name = config.get("VECTOR_DB", "collection_name", fallback="translation_corpus")
    device = config.get("EMBEDDINGS", "device", fallback="cuda")
    
    vector_db = VectorDBManager(
        db_type=db_type,
        embedding_model=embedding_model,
        db_path=db_path,
        collection_name=collection_name,
        device=device
    )
    
    # Initialize RAG Agent with Local Llama
    translation_model = config.get("TRANSLATION", "model_name")
    max_length = config.getint("TRANSLATION", "max_length", fallback=256)
    temperature = config.getfloat("TRANSLATION", "temperature", fallback=0.3)
    top_p = config.getfloat("TRANSLATION", "top_p", fallback=0.9)
    top_k_retrieval = config.getint("VECTOR_DB", "top_k", fallback=3)
    use_4bit = config.getboolean("TRANSLATION", "use_4bit", fallback=True)
    
    agent = RAGTranslationAgent(
        model_name=translation_model,
        vector_db_manager=vector_db,
        device=device,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        top_k_retrieval=top_k_retrieval,
        use_4bit=use_4bit
    )
    
    return agent, vector_db, config


def main():
    """Main application"""
    
    # Header
    st.markdown('<div class="main-header">🌐 RAG Translation Agent</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Context-Aware English to Arabic Translation</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Load models
        with st.spinner("Loading models..."):
            try:
                agent, vector_db, config = load_models()
                st.success("✅ Models loaded successfully!")
            except Exception as e:
                st.error(f"❌ Error loading models: {str(e)}")
                st.stop()
        
        # Database stats
        st.subheader("📊 Database Statistics")
        stats = vector_db.get_stats()
        st.metric("Total Entries", stats["total_entries"])
        st.metric("DB Type", stats["db_type"])
        
        st.divider()
        
        # Translation settings
        st.subheader("🔧 Translation Settings")
        
        use_context = st.checkbox("Use RAG Context", value=True, help="Use retrieved examples to improve translation")
        top_k = st.slider("Number of Retrieved Examples", min_value=1, max_value=10, value=3)
        
        st.divider()
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🔤 Translate", "📝 Batch Translation", "🔍 About"])
    
    # Tab 1: Single Translation
    with tab1:
        st.header("Single Text Translation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("English Input")
            source_text = st.text_area(
                "Enter English text to translate:",
                height=200,
                placeholder="Enter your text here...",
                key="source_input"
            )
            
            translate_btn = st.button("🔄 Translate", type="primary", use_container_width=True)
        
        with col2:
            st.subheader("Arabic Translation")
            translation_output = st.empty()
        
        if translate_btn and source_text:
            with st.spinner("Translating..."):
                try:
                    result = agent.translate(
                        source_text=source_text,
                        use_context=use_context,
                        top_k=top_k,
                        return_context=True
                    )
                    
                    # Display translation
                    translation_output.text_area(
                        "Translation:",
                        value=result["translation"],
                        height=200,
                        key="translation_output"
                    )
                    
                    # Show retrieved context
                    if use_context and result.get("used_context"):
                        st.success(f"✅ Used {result.get('num_retrieved', 0)} retrieved examples")
                        
                        with st.expander("📚 Retrieved Context Examples"):
                            for i, example in enumerate(result.get("retrieved_examples", []), 1):
                                st.markdown(f"**Example {i}** (Similarity: {example['similarity']:.3f})")
                                st.markdown(f"🇬🇧 **EN:** {example['en']}")
                                st.markdown(f"🇸🇦 **AR:** {example['ar']}")
                                st.divider()
                    else:
                        st.info("ℹ️ Translated without context")
                        
                except Exception as e:
                    st.error(f"❌ Translation error: {str(e)}")
        
        elif translate_btn:
            st.warning("⚠️ Please enter text to translate")
    
    # Tab 2: Batch Translation
    with tab2:
        st.header("Batch Translation")
        
        st.info("📄 Upload a CSV file with an 'en' column containing English texts to translate")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(df)} rows")
                
                # Show preview
                st.subheader("Data Preview")
                st.dataframe(df.head())
                
                # Column selection
                col1, col2 = st.columns(2)
                with col1:
                    source_col = st.selectbox("Source column", df.columns.tolist(), index=0)
                with col2:
                    sample_size = st.number_input("Number of rows to translate", min_value=1, max_value=len(df), value=min(10, len(df)))
                
                if st.button("🔄 Translate Batch", type="primary"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    translations = []
                    sample_df = df.head(sample_size)
                    
                    for i, row in enumerate(sample_df.iterrows()):
                        idx, data = row
                        source = str(data[source_col])
                        
                        status_text.text(f"Translating {i+1}/{sample_size}...")
                        progress_bar.progress((i + 1) / sample_size)
                        
                        try:
                            result = agent.translate(
                                source_text=source,
                                use_context=use_context,
                                top_k=top_k
                            )
                            translations.append(result["translation"])
                        except Exception as e:
                            translations.append(f"Error: {str(e)}")
                    
                    # Add translations to dataframe
                    sample_df['translation'] = translations
                    
                    st.success("✅ Batch translation complete!")
                    st.dataframe(sample_df[[source_col, 'translation']])
                    
                    # Download button
                    csv = sample_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name=f"translations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
    
    # Tab 3: About
    with tab3:
        st.header("About This System")
        
        st.markdown("""
        ### 🎯 What is RAG Translation?
        
        **RAG (Retrieval-Augmented Generation)** combines the power of:
        - **Semantic Search**: Finding similar translation examples from a large corpus
        - **Neural Machine Translation**: Generating high-quality translations
        
        ### 🔄 How It Works
        
        1. **Retrieval**: When you provide English text, the system searches a vector database 
           to find similar sentences that have been translated before.
        
        2. **Context Formation**: The retrieved examples serve as context, helping the model 
           understand domain-specific terminology and style.
        
        3. **Translation**: The translation model uses these examples to generate more accurate, 
           context-aware translations.
        
        ### ✨ Key Features
        
        - 🎯 **Domain-Specific**: Better translations for specialized fields (economic, technical, etc.)
        - 🧠 **Context-Aware**: Uses similar examples to maintain consistency
        - 📊 **Transparent**: Shows which examples influenced the translation
        - 🚀 **Fast**: Efficient vector search for real-time retrieval
        
        ### 🛠️ Technical Stack
        
        - **Embedding Model**: Multilingual sentence transformers for semantic search
        - **Vector Database**: ChromaDB/FAISS for efficient similarity search
        - **Translation Model**: State-of-the-art neural MT models
        - **Framework**: Streamlit for interactive UI
        
        ### 📚 Data Source
        
        The system is trained on the **UN Parallel Corpus**, a high-quality collection 
        of professional translations covering various domains.
        """)
        
        # Model info
        st.divider()
        st.subheader("🤖 Current Model Configuration")
        model_info = agent.get_model_info()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Model", model_info["model_name"].split("/")[-1])
            st.metric("Device", model_info["device"])
            st.metric("VRAM Allocated", f"{model_info['vram_allocated_gb']:.2f} GB")
        with col2:
            st.metric("Parameters", f"{model_info['parameters']/1e9:.2f}B")
            st.metric("VRAM Reserved", f"{model_info['vram_reserved_gb']:.2f} GB")


if __name__ == "__main__":
    main()

"""
Streamlit UI for QWEN Translation Models
Interactive interface for testing domain-specific English-Arabic translation
"""

import streamlit as st
import torch
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
from typing import Dict, List
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from evaluate import QWENTranslator

# Page configuration
st.set_page_config(
    page_title="QWEN Translation Studio",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTextArea textarea {
        font-size: 16px;
    }
    .translation-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

class TranslationApp:
    def __init__(self):
        self.domains = ["technology", "economic", "education"]
        self.base_model = "Qwen/Qwen2-1.5B-Instruct"
        self.models_cache = {}
        
    @st.cache_resource
    def load_model(_self, domain: str, model_path: str):
        """Load model with caching"""
        try:
            translator = QWENTranslator(
                model_path=model_path,
                base_model=_self.base_model,
                domain=domain,
                use_4bit=True
            )
            return translator
        except Exception as e:
            st.error(f"Error loading {domain} model: {str(e)}")
            return None
    
    def load_evaluation_results(self, domain: str) -> Dict:
        """Load evaluation results if available"""
        results_path = Path(f"results/{domain}_results.json")
        if results_path.exists():
            with open(results_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def render_header(self):
        """Render application header"""
        st.title("🌐 QWEN Translation Studio")
        st.markdown("### Domain-Specific English-Arabic Translation")
        st.markdown("---")
    
    def render_sidebar(self):
        """Render sidebar with model selection and settings"""
        st.sidebar.header("⚙️ Configuration")
        
        # Domain selection
        domain = st.sidebar.selectbox(
            "Select Domain",
            self.domains,
            format_func=lambda x: x.capitalize()
        )
        
        # Model path
        model_path = st.sidebar.text_input(
            "Model Path",
            value=f"models/{domain}_model",
            help="Path to the fine-tuned model directory"
        )
        
        st.sidebar.markdown("---")
        
        # Generation parameters
        st.sidebar.subheader("Generation Settings")
        max_tokens = st.sidebar.slider("Max Tokens", 64, 1024, 512)
        temperature = st.sidebar.slider("Temperature", 0.1, 1.0, 0.7, 0.1)
        
        st.sidebar.markdown("---")
        
        # Model info
        st.sidebar.subheader("📊 Model Information")
        st.sidebar.info(f"""
        **Base Model:** QWEN 2 (1.5B)
        **Domain:** {domain.capitalize()}
        **Fine-tuning:** LoRA/QLoRA
        **Languages:** English → Arabic
        """)
        
        return domain, model_path, max_tokens, temperature
    
    def render_translation_tab(self, domain: str, model_path: str, max_tokens: int, temperature: float):
        """Render translation interface"""
        st.header("✍️ Translate Text")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("English Input")
            input_text = st.text_area(
                "Enter text to translate:",
                height=200,
                placeholder=f"Enter {domain} domain text in English..."
            )
            
            # Sample texts
            samples = {
                "technology": "The application provides a user-friendly interface for accessing settings and configurations.",
                "economic": "The gross domestic product increased by 3.5% in the last quarter.",
                "education": "The research methodology includes both qualitative and quantitative analysis."
            }
            
            if st.button("Load Sample Text"):
                input_text = samples.get(domain, "")
                st.rerun()
        
        with col2:
            st.subheader("Arabic Translation")
            
            if input_text and st.button("🔄 Translate", type="primary"):
                with st.spinner("Translating..."):
                    try:
                        # Load model
                        translator = self.load_model(domain, model_path)
                        
                        if translator:
                            # Translate
                            start_time = time.time()
                            translation = translator.translate(
                                input_text,
                                max_new_tokens=max_tokens,
                                temperature=temperature
                            )
                            translation_time = time.time() - start_time
                            
                            # Display translation
                            st.markdown(f"""
                            <div class="translation-box" dir="rtl">
                                {translation}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Metrics
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("Input Length", f"{len(input_text)} chars")
                            with col_b:
                                st.metric("Output Length", f"{len(translation)} chars")
                            with col_c:
                                st.metric("Time", f"{translation_time:.2f}s")
                            
                            # Save to history
                            if 'translation_history' not in st.session_state:
                                st.session_state.translation_history = []
                            
                            st.session_state.translation_history.append({
                                'domain': domain,
                                'source': input_text,
                                'translation': translation,
                                'time': translation_time
                            })
                            
                    except Exception as e:
                        st.error(f"Translation error: {str(e)}")
    
    def render_comparison_tab(self, model_paths: Dict[str, str], max_tokens: int, temperature: float):
        """Render domain comparison interface"""
        st.header("🔍 Compare Domains")
        st.markdown("Translate the same text using different domain-specific models")
        
        input_text = st.text_area(
            "Enter text to translate:",
            height=150,
            placeholder="Enter text that could belong to multiple domains..."
        )
        
        if input_text and st.button("🔄 Translate with All Models", type="primary"):
            cols = st.columns(len(self.domains))
            
            for idx, domain in enumerate(self.domains):
                with cols[idx]:
                    st.subheader(f"{domain.capitalize()}")
                    
                    with st.spinner(f"Translating with {domain} model..."):
                        try:
                            model_path = model_paths.get(domain, f"models/{domain}_model")
                            translator = self.load_model(domain, model_path)
                            
                            if translator:
                                translation = translator.translate(
                                    input_text,
                                    max_new_tokens=max_tokens,
                                    temperature=temperature
                                )
                                
                                st.markdown(f"""
                                <div class="translation-box" dir="rtl">
                                    {translation}
                                </div>
                                """, unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
    
    def render_evaluation_tab(self):
        """Render evaluation results"""
        st.header("📊 Model Evaluation")
        
        # Load results for all domains
        results = {}
        for domain in self.domains:
            result = self.load_evaluation_results(domain)
            if result:
                results[domain] = result
        
        if not results:
            st.warning("No evaluation results found. Please run evaluation first.")
            st.code("python evaluate.py --model_path models/technology_model --domain technology")
            return
        
        # Display metrics
        st.subheader("Performance Metrics")
        
        metrics_data = []
        for domain, result in results.items():
            metrics_data.append({
                'Domain': domain.capitalize(),
                'BLEU Score': result.get('bleu_score', 0),
                'chrF Score': result.get('chrf_score', 0),
                'Samples': result.get('num_samples', 0)
            })
        
        df_metrics = pd.DataFrame(metrics_data)
        
        # Display table
        st.dataframe(df_metrics, use_container_width=True)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # BLEU scores
            fig_bleu = px.bar(
                df_metrics,
                x='Domain',
                y='BLEU Score',
                title='BLEU Scores by Domain',
                color='Domain',
                text='BLEU Score'
            )
            fig_bleu.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            st.plotly_chart(fig_bleu, use_container_width=True)
        
        with col2:
            # chrF scores
            fig_chrf = px.bar(
                df_metrics,
                x='Domain',
                y='chrF Score',
                title='chrF Scores by Domain',
                color='Domain',
                text='chrF Score'
            )
            fig_chrf.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            st.plotly_chart(fig_chrf, use_container_width=True)
        
        # Example translations
        st.subheader("Example Translations")
        
        selected_domain = st.selectbox("Select domain to view examples:", list(results.keys()))
        
        if selected_domain and 'examples' in results[selected_domain]:
            examples = results[selected_domain]['examples']
            
            for i, example in enumerate(examples):
                with st.expander(f"Example {i+1}"):
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.markdown("**Source (English):**")
                        st.write(example['source'])
                        
                    with col_b:
                        st.markdown("**Reference (Arabic):**")
                        st.markdown(f"<div dir='rtl'>{example['reference']}</div>", unsafe_allow_html=True)
                    
                    st.markdown("**Model Translation (Arabic):**")
                    st.markdown(f"<div dir='rtl'>{example['translation']}</div>", unsafe_allow_html=True)
    
    def render_history_tab(self):
        """Render translation history"""
        st.header("📜 Translation History")
        
        if 'translation_history' not in st.session_state or not st.session_state.translation_history:
            st.info("No translations yet. Start translating to see history!")
            return
        
        history = st.session_state.translation_history
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Translations", len(history))
        with col2:
            avg_time = sum(h['time'] for h in history) / len(history)
            st.metric("Avg. Translation Time", f"{avg_time:.2f}s")
        with col3:
            domains_used = set(h['domain'] for h in history)
            st.metric("Domains Used", len(domains_used))
        
        st.markdown("---")
        
        # Display history
        for i, entry in enumerate(reversed(history[-10:])):  # Show last 10
            with st.expander(f"Translation {len(history) - i} - {entry['domain'].capitalize()}"):
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown("**Source:**")
                    st.write(entry['source'])
                
                with col_b:
                    st.markdown("**Translation:**")
                    st.markdown(f"<div dir='rtl'>{entry['translation']}</div>", unsafe_allow_html=True)
                
                st.caption(f"Time: {entry['time']:.2f}s")
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.translation_history = []
            st.rerun()
    
    def run(self):
        """Main application loop"""
        self.render_header()
        
        # Sidebar
        domain, model_path, max_tokens, temperature = self.render_sidebar()
        
        # Main content tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "🔤 Translate",
            "🔍 Compare Domains",
            "📊 Evaluation",
            "📜 History"
        ])
        
        with tab1:
            self.render_translation_tab(domain, model_path, max_tokens, temperature)
        
        with tab2:
            model_paths = {d: f"models/{d}_model" for d in self.domains}
            self.render_comparison_tab(model_paths, max_tokens, temperature)
        
        with tab3:
            self.render_evaluation_tab()
        
        with tab4:
            self.render_history_tab()

def main():
    app = TranslationApp()
    app.run()

if __name__ == "__main__":
    main()

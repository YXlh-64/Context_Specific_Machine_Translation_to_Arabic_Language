import os
import re
import json
import requests
from typing import List, Dict, Tuple


# Load environment variables (Make sure .env has OPENROUTER_API_KEY)


# ==========================================
# 1. THE JUDGE AGENT CLASS (Using requests)
# ==========================================
class RAGJudge:
    def __init__(self, model_name: str = "nex-agi/deepseek-v3.1-nex-n1:free"):
        """Initialize with OpenRouter API Key."""
        self.api_key = "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"
        self.model_name = model_name
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        
        if not self.api_key:
            raise ValueError("❌ Missing OPENROUTER_API_KEY in .env file")

    def evaluate(self, query: str, results: List[Dict]) -> Dict:
        """Sends query and results to LLM via requests."""
        
        # 1. Format the results for the LLM
        context_text = ""
        for res in results:
            context_text += f"""
            [Rank #{res['rank']}]
            - Source: "{res['source']}"
            - Target: "{res['target']}"
            """

        # 2. Construct Prompts
        system_prompt = """
        You are an AI Judge for a Retrieval System. Evaluate the quality of the retrieved results based on the User Query.
        
        Criteria:
        1. Relevance (1-10): Did we find the exact answer?
        2. Ranking (1-10): Is the best answer at Rank #1?
        
        You MUST return ONLY raw JSON (no markdown formatting, no backticks).
        JSON Structure:
        {
            "relevance_score": int,
            "ranking_score": int,
            "best_rank": int,
            "reasoning": "string",
            "verdict": "PERFECT" | "ACCEPTABLE" | "POOR"
        }
        """

        user_prompt = f"Query: {query}\n\nRetrieved Results:{context_text}"

        # 3. Prepare Request Payload
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            # Optional: Force JSON object if supported by the model, 
            # otherwise prompt engineering (above) handles it.
            "response_format": {"type": "json_object"} 
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000", # Optional
            "X-Title": "RAG Judge Agent",            # Optional
        }

        try:
            # 4. Send Request
            response = requests.post(
                url=self.api_url,
                headers=headers,
                data=json.dumps(payload)
            )
            response.raise_for_status()
            
            # 5. Parse Response
            result_json = response.json()
            content = result_json['choices'][0]['message']['content']
            
            return self._clean_and_parse_json(content)

        except Exception as e:
            return {"error": f"API Request failed: {str(e)}"}

    def _clean_and_parse_json(self, content: str) -> Dict:
        """Helper to handle models that wrap JSON in markdown blocks."""
        try:
            # Remove ```json and ``` if present
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            
            return json.loads(content)
        except json.JSONDecodeError:
            return {"error": "Failed to parse JSON response from model", "raw_content": content}

# ==========================================
# 2. MARKDOWN PARSER (Reads your file)
# ==========================================
def parse_markdown_report(file_path: str) -> Tuple[str, List[Dict]]:
    """Reads the MD file and extracts Query + Hybrid Results."""
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None, []

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Extract Query
    query_match = re.search(r"\*\*Query:\*\*\s*`([^`]+)`", content)
    if not query_match:
        print("❌ Could not find Query in Markdown.")
        return None, []
    query = query_match.group(1)

    # 2. Extract Hybrid Section
    hybrid_section_match = re.search(r"## 🔍 3\. Hybrid RRF Search(.*?)(?=\n## |\Z)", content, re.DOTALL)
    if not hybrid_section_match:
        print("❌ Could not find Hybrid Search section.")
        return query, []
    
    hybrid_text = hybrid_section_match.group(1)

    # 3. Extract Results
    results = []
    blocks = hybrid_text.split("### ")[1:] 
    
    for block in blocks:
        try:
            lines = block.strip().split('\n')
            
            # Get Rank line (e.g., "1. 📄 [EXISTING]")
            rank_line = lines[0]
            # Extract just the number
            rank = int(re.search(r"(\d+)\.", rank_line).group(1))
            
            source = "N/A"
            target = "N/A"
            
            for line in lines:
                if "**Source:**" in line:
                    source = line.split("**Source:**")[1].strip()
                if "**Target:**" in line:
                    target = line.split("**Target:**")[1].strip()
            
            results.append({
                "rank": rank,
                "source": source,
                "target": target
            })
        except Exception as e:
            print(f"⚠️ Error parsing block: {e}")
            continue

    return query, results

# ==========================================
# 3. APPEND RESULT TO FILE
# ==========================================
def append_verdict(file_path: str, evaluation: Dict):
    """Writes the AI Judge's verdict to the bottom of the MD file."""
    
    md_output = f"""
    
---

## 👨‍⚖️ AI Judge Evaluation (DeepSeek v3)
| Metric | Score |
| :--- | :--- |
| **Relevance** | `{evaluation.get('relevance_score')}/10` |
| **Ranking** | `{evaluation.get('ranking_score')}/10` |
| **Best Result** | `Rank #{evaluation.get('best_rank')}` |
| **Verdict** | **{evaluation.get('verdict')}** |

**Reasoning:**
> {evaluation.get('reasoning')}
"""
    
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(md_output)
    
    print(f"✅ Verdict appended to {file_path}")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    TARGET_FILE = "search_results_new4.md"  # <--- Verify this filename
    
    print(f"📂 Reading {TARGET_FILE}...")
    query, results = parse_markdown_report(TARGET_FILE)
    
    if query and results:
        print(f"❓ Query: {query}")
        print(f"📊 Found {len(results)} Hybrid results to judge.")
        
        print("🤖 Asking Judge Agent (DeepSeek)...")
        # Initialize with the specific model requested
        judge = RAGJudge(model_name="nex-agi/deepseek-v3.1-nex-n1:free")
        
        verdict = judge.evaluate(query, results)
        
        if "error" not in verdict:
            append_verdict(TARGET_FILE, verdict)
            print("🎉 Done!")
        else:
            print(f"❌ Judge Error: {verdict['error']}")
            if 'raw_content' in verdict:
                print(f"Raw response: {verdict['raw_content']}")
    else:
        print("❌ Failed to parse data.")
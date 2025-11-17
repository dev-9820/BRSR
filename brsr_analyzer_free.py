# BRSR Faithfulness Analyzer - 100% FREE VERSION
# Using: Groq (Free LLaMA 3), Hugging Face, or Local LLMs

import os
import json
from typing import List, Dict
from pypdf import PdfReader
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from google import genai
from google.genai import types


# =============================================================================
# OPTION 1: GROQ (FREE - FASTEST) - Recommended!
# Get free API key from: https://console.groq.com
# =============================================================================

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("Groq not installed. Install with: pip install groq")

# =============================================================================
# OPTION 2: HUGGING FACE (FREE - Good for fallback)
# Get free API key from: https://huggingface.co/settings/tokens
# =============================================================================

try:
    from huggingface_hub import InferenceClient
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Hugging Face not installed. Install with: pip install huggingface_hub")

# =============================================================================
# OPTION 3: OLLAMA (COMPLETELY OFFLINE - No API key needed!)
# Download from: https://ollama.ai
# =============================================================================

try:
    import requests
    OLLAMA_AVAILABLE = True
except:
    OLLAMA_AVAILABLE = False

# =============================================================================
# FREE LLM CLIENT - Supports multiple backends
# =============================================================================

class FreeLLMClient:
    """Unified client supporting multiple free LLM providers"""
    
    def __init__(self, provider="groq"):
        """
        Initialize with preferred provider
        
        Options:
        - "groq": Fast, free, recommended (LLaMA 3 70B)
        - "huggingface": Free API, good fallback
        - "ollama": Completely offline, no API key
        """
        self.provider = provider
        
        if provider == "groq":
            if not GROQ_AVAILABLE:
                raise ImportError("Install groq: pip install groq")
            api_key = os.getenv("GROQ_API_KEY", "")
            if not api_key:
                print("\n⚠️  Get FREE Groq API key from: https://console.groq.com")
                print("Then set: export GROQ_API_KEY='your-key-here'\n")
            self.client = Groq(api_key=api_key)
            self.model = "meta-llama/Meta-Llama-3-70B-Instruct"

        elif provider == "gemini":
            API_KEY = "AIzaSyA0gpYzWGMBnyQk4JaUz8FB0pDzKfJ04Dc"
            self.client = genai.Client(api_key=API_KEY)
            genai.Client(api_key="AIzaSyA0gpYzWGMBnyQk4JaUz8FB0pDzKfJ04Dc")
            self.model = "gemini-2.5-flash"  # or "gemini-1" depending on availability
    
    def generate(self, prompt: str, max_tokens: int = 2000) -> str:
        """Generate response using selected provider"""
        
        if self.provider == "groq":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0
            )
            return response.choices[0].message.content
        
        elif self.provider == "huggingface":
            response = self.client.chat_completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            # The text is in response["message"]["content"]
            return response["message"]["content"]
        
        elif self.provider == "gemini":
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0,
                        max_output_tokens=max_tokens
                    )
                )
                return response.text if response.text is not None else "" 
            except Exception as e:
                print(f"Gemini API generation failed: {e}")
                return "" 

        
        elif self.provider == "ollama":
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': self.model,
                    'prompt': prompt,
                    'stream': False
                }
            )
            return response.json()['response']

# =============================================================================
# PDF EXTRACTION (Same as before)
# =============================================================================

class PDFExtractor:
    """Extract and preprocess text from PDF files"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> Dict:
        """Extract text from PDF with page-level granularity"""
        reader = PdfReader(pdf_path)
        pages = []
        
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            pages.append({
                'page_num': page_num + 1,
                'content': text,
                'char_count': len(text)
            })
        
        return {
            'total_pages': len(pages),
            'pages': pages,
            'full_text': '\n\n'.join([p['content'] for p in pages])
        }
    
    @staticmethod
    def extract_principle6_section(full_text: str) -> str:
        """Extract Principle 6 specific content"""
        markers = [
            'Principle 6', 'PRINCIPLE 6', 'Environment',
            'ENVIRONMENT', 'Leadership Indicator 6', 'Essential Indicator 6'
        ]
        
        lines = full_text.split('\n')
        p6_content = []
        capturing = False
        
        for line in lines:
            if any(marker in line for marker in markers):
                capturing = True
            elif capturing and 'Principle 7' in line:
                break
            
            if capturing:
                p6_content.append(line)
        
        return '\n'.join(p6_content) if p6_content else full_text[:15000]

# =============================================================================
# FREE BRSR ANALYZER
# =============================================================================

class FreeBRSRAnalyzer:
    """BRSR Analyzer using FREE LLMs"""
    
    def __init__(self, provider="groq"):
        """
        Initialize with free LLM provider
        
        Options:
        - "groq": Fast & Free (Recommended) - Get key from console.groq.com
        - "huggingface": Free API - Get token from huggingface.co
        - "ollama": Completely offline - Download from ollama.ai
        """
        print(f"\n🤖 Initializing FREE BRSR Analyzer with {provider.upper()}...")
        self.llm = FreeLLMClient(provider=provider)
        print("✓ LLM client ready!\n")
    
    def extract_brsr_requirements(self, sebi_text: str) -> List[Dict]:
        """Extract BRSR Principle 6 requirements using free LLM"""
        
        print("🔍 Extracting BRSR requirements...")
        
        prompt = f"""Extract ALL Principle 6 (Environment) requirements from this SEBI BRSR Framework text.

For each requirement, provide:
1. ID (P6.1, P6.2, etc.)
2. Title
3. Exact requirement text
4. Citation

Return ONLY valid JSON array, no markdown:
[{{"id": "P6.1", "title": "...", "requirement": "...", "citation": "..."}}]

Text (first 10000 chars):
{sebi_text[:1000000]}

JSON array:"""

        try:
            response = self.llm.generate(prompt, max_tokens=2000)
            # Clean response
            response = response.replace('```json', '').replace('```', '').strip()
            requirements = json.loads(response)
            print(f"✓ Found {len(requirements)} requirements")
            return requirements
        except Exception as e:
            print(f"⚠️  Using default requirements (parsing failed: {e})")
            return self._get_default_requirements()
    
    def _get_default_requirements(self) -> List[Dict]:
        """Fallback requirements based on SEBI BRSR 2021"""
        return [
            {
                "id": "P6.1",
                "title": "Energy Consumption & Conservation",
                "requirement": "Total energy consumption from renewable and non-renewable sources in Joules or multiples, energy intensity per rupee of turnover",
                "citation": "SEBI BRSR Framework 2021, Essential Indicator 1"
            },
            {
                "id": "P6.2",
                "title": "Water Withdrawal & Consumption",
                "requirement": "Water withdrawal by source in kiloliters, total volume of water consumption, water intensity per rupee of turnover",
                "citation": "SEBI BRSR Framework 2021, Essential Indicator 2"
            },
            {
                "id": "P6.3",
                "title": "Waste Management",
                "requirement": "Total waste generated in metric tonnes (plastic, e-waste, hazardous, non-hazardous), waste diverted from disposal",
                "citation": "SEBI BRSR Framework 2021, Essential Indicator 3"
            },
            {
                "id": "P6.4",
                "title": "Greenhouse Gas Emissions",
                "requirement": "Total Scope 1, Scope 2, and Scope 3 emissions in metric tonnes of CO2 equivalent, GHG emission intensity",
                "citation": "SEBI BRSR Framework 2021, Essential Indicator 4"
            },
            {
                "id": "P6.5",
                "title": "Biodiversity Impact",
                "requirement": "Details of operations in biodiversity hotspots, impact assessment, mitigation measures",
                "citation": "SEBI BRSR Framework 2021, Essential Indicator 5"
            }
        ]
    
    def analyze_company_disclosure(self, requirement: Dict, company_text: str) -> Dict:
        """Analyze how well company disclosure matches requirement"""
        
        prompt = f"""Audit a company's BRSR disclosure against SEBI requirement.

SEBI REQUIREMENT:
{requirement['id']}: {requirement['title']}
Required: {requirement['requirement']}

COMPANY TEXT (search in this):
{company_text[:8000]}

Analyze and return ONLY valid JSON (no markdown):
{{
  "company_disclosure": "exact text from company or 'Not disclosed'",
  "company_citation": "Page/section reference",
  "faithfulness_score": 0-3,
  "drift_category": "Exact Match or Minor Deviation or Moderate Gap or Vague/Performative",
  "justification": "2 sentence explanation with evidence"
}}

Scoring:
0 = All data provided correctly
1 = Most data, minor issues
2 = Partial data, gaps
3 = No data, vague only

JSON:"""

        try:
            response = self.llm.generate(prompt, max_tokens=1000)
            response = response.replace('```json', '').replace('```', '').strip()
            analysis = json.loads(response)
            return analysis
        except Exception as e:
            print(f"  ⚠️  Analysis failed for {requirement['id']}: {e}")
            return {
                "company_disclosure": "Analysis failed",
                "company_citation": "Unknown",
                "faithfulness_score": 2,
                "drift_category": "Moderate Gap",
                "justification": f"Unable to parse: {str(e)[:100]}"
            }
    
    def perform_full_analysis(self, sebi_pdf_path: str, company_pdf_path: str) -> Dict:
        """Complete end-to-end analysis with FREE LLM"""
        
        print("\n" + "="*60)
        print("🚀 BRSR FAITHFULNESS ANALYZER (FREE VERSION)")
        print("="*60)
        
        # Extract PDFs
        print("\n📄 Step 1: Extracting PDFs...")
        sebi_data = PDFExtractor.extract_text_from_pdf(sebi_pdf_path)
        company_data = PDFExtractor.extract_text_from_pdf(company_pdf_path)
        print(f"  ✓ SEBI: {sebi_data['total_pages']} pages")
        print(f"  ✓ Company: {company_data['total_pages']} pages")
        
        # Extract P6 section
        company_p6 = PDFExtractor.extract_principle6_section(company_data['full_text'])
        print(f"  ✓ Found Principle 6 section ({len(company_p6)} chars)")
        
        # Extract requirements
        print("\n🔍 Step 2: Extracting BRSR requirements...")
        requirements = self.extract_brsr_requirements(sebi_data['full_text'])
        
        # Analyze each requirement
        print("\n🤖 Step 3: Analyzing compliance (this may take 2-3 minutes)...")
        results = []
        for i, req in enumerate(requirements, 1):
            print(f"  [{i}/{len(requirements)}] Analyzing {req['id']}: {req['title'][:40]}...")
            analysis = self.analyze_company_disclosure(req, company_p6)
            results.append({
                'requirement': req,
                'analysis': analysis
            })
        
        # Calculate summary
        scores = [r['analysis']['faithfulness_score'] for r in results]
        summary = {
            'total_principles': len(results),
            'exact_match': scores.count(0),
            'minor_deviation': scores.count(1),
            'moderate_gap': scores.count(2),
            'vague': scores.count(3),
            'average_score': sum(scores) / len(scores) if scores else 0
        }
        
        print("\n✅ Analysis complete!")
        print(f"   Average Score: {summary['average_score']:.2f}/3.0")
        
        return {
            'results': results,
            'summary': summary
        }

# =============================================================================
# VISUALIZATION (Same as before)
# =============================================================================

class VisualizationGenerator:
    """Generate visualizations"""
    
    @staticmethod
    def create_sankey_diagram(analysis_results: Dict) -> go.Figure:
        """Create Sankey diagram"""
        results = analysis_results['results']
        
        node_labels = []
        for result in results:
            req = result['requirement']
            node_labels.append(f"{req['id']}: {req['title']}")
        
        quality_nodes = ['Exact Match', 'Minor Deviation', 'Moderate Gap', 'Vague']
        node_labels.extend(quality_nodes)
        
        source_nodes = []
        target_nodes = []
        values = []
        colors = []
        
        score_colors = {
            0: 'rgba(34, 197, 94, 0.7)',
            1: 'rgba(234, 179, 8, 0.7)',
            2: 'rgba(249, 115, 22, 0.7)',
            3: 'rgba(239, 68, 68, 0.7)'
        }
        
        for i, result in enumerate(results):
            score = result['analysis']['faithfulness_score']
            source_nodes.append(i)
            target_nodes.append(len(results) + score)
            values.append(10)
            colors.append(score_colors[score])
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                label=node_labels,
                color=['#3b82f6'] * len(results) + ['#10b981', '#f59e0b', '#f97316', '#ef4444']
            ),
            link=dict(
                source=source_nodes,
                target=target_nodes,
                value=values,
                color=colors
            )
        )])
        
        fig.update_layout(
            title="BRSR Principle 6: Faithfulness Flow (FREE LLM Analysis)",
            font=dict(size=12),
            height=600
        )
        
        return fig
    
    @staticmethod
    def create_dashboard(analysis_results: Dict) -> go.Figure:
        """Create dashboard"""
        results = analysis_results['results']
        
        principle_ids = [r['requirement']['id'] for r in results]
        scores = [r['analysis']['faithfulness_score'] for r in results]
        
        colors_map = {0: '#10b981', 1: '#f59e0b', 2: '#f97316', 3: '#ef4444'}
        bar_colors = [colors_map[score] for score in scores]
        
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.6, 0.4],
            subplot_titles=("Faithfulness Score", "Distribution"),
            specs=[[{"type": "xy"}], [{"type": "domain"}]],
            vertical_spacing=0.15
        )

        # Bar chart
        fig.add_trace(
            go.Bar(x=principle_ids, y=scores, marker=dict(color=bar_colors)),
            row=1, col=1
        )
        score_counts = [scores.count(i) for i in range(4)]
        # Pie chart
        fig.add_trace(
            go.Pie(
                labels=['Exact (0)', 'Minor (1)', 'Moderate (2)', 'Vague (3)'],
                values=score_counts,
                marker=dict(colors=['#10b981', '#f59e0b', '#f97316', '#ef4444']),
                hole=0.3
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title="BRSR Faithfulness Dashboard (FREE LLM)",
            height=800
        )
        
        return fig

# =============================================================================
# REPORT GENERATION (Same as before)
# =============================================================================

class ReportGenerator:
    """Generate reports"""
    
    @staticmethod
    def generate_summary_report(analysis_results: Dict) -> str:
        """Generate text summary"""
        summary = analysis_results['summary']
        results = analysis_results['results']
        
        report = f"""
{'='*70}
BRSR REALITY CHECK SUMMARY (FREE LLM ANALYSIS)
{'='*70}

Overall Assessment:
- Total Principles: {summary['total_principles']}
- Average Score: {summary['average_score']:.2f}/3.0

Distribution:
- Exact Match (0): {summary['exact_match']}
- Minor Deviation (1): {summary['minor_deviation']}
- Moderate Gap (2): {summary['moderate_gap']}
- Vague (3): {summary['vague']}

DETAILED FINDINGS
{'='*70}

"""
        
        for result in results:
            req = result['requirement']
            analysis = result['analysis']
            
            report += f"""
{req['id']}: {req['title']}
{'-'*70}
Score: {analysis['faithfulness_score']} - {analysis['drift_category']}

Requirement: {req['requirement']}
Citation: {req['citation']}

Company: {analysis['company_disclosure']}
Source: {analysis['company_citation']}

Justification: {analysis['justification']}

"""
        
        return report

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function"""
    
    # Choose your FREE provider
    # Options: "groq" (fastest), "huggingface", or "ollama" (offline)
    PROVIDER = "gemini"
    analyzer = FreeBRSRAnalyzer(provider=PROVIDER)
    
    # File paths
    SEBI_PDF = "data/sebi_brsr_2021.pdf"
    COMPANY_PDF = "data/infosys_brsr_2023.pdf"
    OUTPUT_DIR = "output_free"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Perform analysis
    results = analyzer.perform_full_analysis(SEBI_PDF, COMPANY_PDF)
    
    # Generate visualizations
    print("\n📊 Generating visualizations...")
    viz_gen = VisualizationGenerator()
    
    sankey_fig = viz_gen.create_sankey_diagram(results)
    sankey_fig.write_html(f"{OUTPUT_DIR}/sankey_diagram.html")
    print("  ✓ Sankey saved")
    
    dashboard_fig = viz_gen.create_dashboard(results)
    dashboard_fig.write_html(f"{OUTPUT_DIR}/dashboard.html")
    print("  ✓ Dashboard saved")
    
    # Generate report
    print("\n📝 Generating report...")
    report_gen = ReportGenerator()
    
    summary_text = report_gen.generate_summary_report(results)
    with open(f"{OUTPUT_DIR}/brsr_reality_check.txt", 'w') as f:
        f.write(summary_text)
    print("  ✓ Report saved")
    
    with open(f"{OUTPUT_DIR}/analysis_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print("  ✓ JSON saved")
    
    print(f"\n✅ Complete! Check {OUTPUT_DIR}/ for results")

if __name__ == "__main__":
    main()
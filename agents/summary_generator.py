# agents/summary_generator.py

import sys
import os
from openai import AzureOpenAI

# Ensure the script can find other modules in the project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import configuration variables from config.py
from config.config import AZURE_OPENAI_API_KEY, AZURE_ENDPOINT, AZURE_DEPLOYMENT_NAME, OPENAI_API_VERSION

class SummaryGenerator:
    def __init__(self, validations, analysis_results=None, plot_paths=None, interpretations=None, hypotheses=None):
        """
        Initialize the SummaryGenerator with analysis results and validation outcomes.

        Args:
            validations (list): List of dictionaries with hypothesis validation results (required).
            analysis_results (dict, optional): Results from DataAnalyzer with summary stats and grouped data.
            plot_paths (list, optional): List of file paths to generated plots.
            interpretations (list, optional): List of dictionaries with plot interpretations.
            hypotheses (list, optional): List of hypothesis strings.
        """
        if not validations:
            raise ValueError("No validations provided for summary generation")
        
        self.validations = validations
        self.analysis_results = analysis_results
        self.plot_paths = plot_paths
        self.interpretations = interpretations
        self.hypotheses = hypotheses
        
        # Initialize the Azure OpenAI client
        self.client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_ENDPOINT,
            api_version=OPENAI_API_VERSION
        )
        self.deployment_name = AZURE_DEPLOYMENT_NAME

    def generate(self):
        """
        Generate an enhanced executive summary using Azure OpenAI API in plain language.

        Returns:
            str: A Markdown-formatted executive summary with holistic insights.
        """
        # Format inputs for the prompt
        validation_text = self._format_validations()
        analysis_text = self._format_analysis_results() if self.analysis_results else "No analysis results provided."
        graph_text = self._format_graph_insights() if self.plot_paths and self.interpretations else "No graph insights provided."
        hypotheses_text = self._format_hypotheses() if self.hypotheses else "No hypotheses provided."

        # Enhanced prompt for a richer, synthesized summary
        prompt = f"""
        You are a healthcare analytics expert crafting an executive summary for a study on wound healing treatments for diabetic foot ulcers. The study analyzed a comprehensive dataset over an extended period (200+ weeks), comparing dressings like Aquacel Foam, Duoderm Gel, Aquacel Ag Surgical, Aquacel Ag+ Extra, and others. The analysis included statistical evaluations, graph visualizations, and hypothesis testing to assess dressing effectiveness in reducing wound area.

        Your task is to create a concise, engaging, and professional executive summary in plain English for a non-technical audience (e.g., healthcare executives). The summary should:
        1. **Introduction**: Set the stage with the study’s purpose (improving wound care outcomes) and approach (data-driven comparison of dressings), capturing attention with a clear, impactful opening.
        2. **Key Discoveries**: Synthesize standout findings from the data, graphs, and validations, focusing on what worked best, where differences emerged, and any surprises—without just repeating prior steps.
        3. **Healing Trends**: Highlight overarching patterns in wound healing (e.g., early vs. long-term effects, dressing-specific behaviors) using graph insights, avoiding technical terms like 'p-value' or 'slope'.
        4. **Actionable Insights**: Offer practical takeaways for wound care decisions, balancing what’s conclusive with areas needing more research, in a way that resonates with decision-makers.

        Use the following inputs to craft a narrative that goes beyond summarizing—tell a story of what the data reveals about wound care:

        **Data Analysis Results:**
        {analysis_text}

        **Graph Insights:**
        {graph_text}

        **Hypotheses Tested:**
        {hypotheses_text}

        **Validation Results:**
        {validation_text}

        Format the summary as a Markdown string with clear section headers (e.g., `### Introduction`). Keep it concise (200-300 words), professional, and compelling, emphasizing real-world implications over raw repetition of results.
        """

        try:
            # Call the Azure OpenAI API
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,  # Increased for a richer summary
                temperature=0.5  # Balanced for clarity and insight
            )
            
            # Extract and return the summary
            summary = response.choices[0].message.content.strip()
            return summary

        except Exception as e:
            return f"Error generating summary: {str(e)}"

    def _format_validations(self):
        """Format validation results into a text string."""
        formatted_text = ""
        for i, val in enumerate(self.validations, start=1):
            formatted_text += f"Hypothesis {i}: {val['hypothesis']}\n"
            formatted_text += f"Test: {val['test']}\n"
            formatted_text += f"Stats: {val['stats']}\n"
            formatted_text += f"Interpretation: {val['interpretation']}\n\n"
        return formatted_text.strip()

    def _format_analysis_results(self):
        """Format analysis results into a text string."""
        stats = self.analysis_results['summary_stats']
        dressings = self.analysis_results['grouped_by_dressing']
        return (
            f"Rows: {stats['num_rows']}, Weeks: {stats['weeks_covered']}, Dressings: {stats['num_dressings']}\n"
            f"Avg Wound Area: {stats['avg_wound_area']:.2f}, Std Dev: {stats['std_total_wound_area']:.2f}\n"
            f"Top Dressings by Avg Wound Area: {', '.join([f'{k}: {v:.2f}' for k, v in sorted(dressings.items(), key=lambda x: x[1])[:3]])}"
        )

    def _format_graph_insights(self):
        """Format graph insights into a text string."""
        formatted_text = ""
        for interp in self.interpretations:
            formatted_text += f"Graph: {interp['plot']}\nInsight: {interp['interpretation']}\n\n"
        return formatted_text.strip()

    def _format_hypotheses(self):
        """Format hypotheses into a text string."""
        formatted_text = ""
        for i, hyp in enumerate(self.hypotheses, start=1):
            cleaned_hyp = hyp.split('. ', 1)[-1] if '. ' in hyp and hyp[0].isdigit() else hyp
            formatted_text += f"Hypothesis {i}: {cleaned_hyp}\n"
        return formatted_text.strip()

if __name__ == "__main__":
    # Simulated data for testing
    sample_analysis = {
        'summary_stats': {'num_rows': 1655, 'weeks_covered': 248, 'num_dressings': 14, 'avg_wound_area': 1.01, 'std_total_wound_area': 131.57},
        'grouped_by_dressing': {'aquacel foam': 64.07, 'duoderm gel': 61.06, 'aquacel ag surgical': 4.34}
    }
    sample_plots = ['plot1.png', 'plot2.png']
    sample_interps = [{'plot': 'plot1.png', 'interpretation': 'Decreasing trend over time.'}]
    sample_hyps = ['1. Aquacel Foam reduces wound area faster.', '2. Duoderm Gel is less effective.']
    sample_vals = [
        {'hypothesis': 'Aquacel Foam reduces wound area faster.', 'test': 'T-test', 'stats': 't_stat=1.23, p=0.22', 'interpretation': 'No significant difference.'}
    ]

    generator = SummaryGenerator(
        validations=sample_vals,
        analysis_results=sample_analysis,
        plot_paths=sample_plots,
        interpretations=sample_interps,
        hypotheses=sample_hyps
    )
    summary = generator.generate()
    print(summary)
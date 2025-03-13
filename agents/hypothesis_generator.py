# agents/hypothesis_generator.py

import sys
import os
from openai import AzureOpenAI

# Adjust the system path to import config from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

class HypothesisGenerator:
    def __init__(self, analysis_results, interpretations):
        """
        Initialize the HypothesisGenerator with analysis results and graph interpretations.

        Args:
            analysis_results (dict): Results from DataAnalyzer containing summary stats and grouped data.
            interpretations (list): List of dictionaries with plot names and their interpretations from GraphInterpreter.
        """
        # Validate inputs
        if not analysis_results or "summary_stats" not in analysis_results:
            raise ValueError("Invalid or missing analysis results")
        if not interpretations:
            raise ValueError("No interpretations provided for hypothesis generation")

        self.analysis_results = analysis_results
        self.interpretations = interpretations
        
        # Initialize Azure OpenAI client using configuration from config.py
        self.client = AzureOpenAI(
            api_key=config.AZURE_OPENAI_API_KEY,
            azure_endpoint=config.AZURE_ENDPOINT,
            api_version=config.OPENAI_API_VERSION
            
        )
        self.deployment_name = config.AZURE_DEPLOYMENT_NAME

    def generate(self):
        """
        Generate hypotheses using Azure OpenAI based on analysis results and graph interpretations.
        Returns a list of hypotheses that can be tested statistically by downstream agents.

        Returns:
            list: List of hypothesis strings.
        """
        # Extract key data for the prompt
        summary_stats = self.analysis_results["summary_stats"]
        grouped_by_dressing = self.analysis_results["grouped_by_dressing"]
        grouped_by_week = self.analysis_results["grouped_by_week"]
        
        # Combine interpretations into a single string for the prompt
        interpretations_text = "\n".join(
            [f"Plot: {interp['plot']}\nInterpretation: {interp['interpretation']}" 
             for interp in self.interpretations]
        )

        # Define a descriptive prompt for Azure OpenAI, focusing on average wound area
        prompt = f"""
        You are a healthcare researcher analyzing wound healing data for diabetic foot ulcers over a long-term period (200+ weeks). 
        Based on the following analysis results and graph interpretations, generate 3-4 testable hypotheses about the effectiveness 
        of dressing types (e.g., Aquacel Foam, Duoderm Gel) in reducing average wound area over time or at specific time points.

        Each hypothesis should be concise, specific, and suitable for statistical validation (e.g., comparing means, trends over time).
        Consider both short-term and long-term effects, and focus on average wound area rather than total wound area.

        Analysis Results:
        - Average wound area: {summary_stats['avg_wound_area']:.2f}
        - Mean average wound area by dressing type: {grouped_by_dressing}
        - Mean average wound area by week: {grouped_by_week}

        Graph Interpretations:
        {interpretations_text}

        Provide each hypothesis as a single sentence, numbered (e.g., 1. Hypothesis text).
        """
        
        try:
            # Call Azure OpenAI API to generate hypotheses
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,  # Increased to allow for more detailed hypotheses
                temperature=0.7
            )
            
            # Extract and split the response into individual hypotheses
            hypotheses_text = response.choices[0].message.content.strip()
            hypotheses = [line.strip() for line in hypotheses_text.split("\n") if line.strip()]
            
            return hypotheses
        except Exception as e:
            return [f"Error generating hypotheses: {str(e)}"]

# Example usage for testing
if __name__ == "__main__":
    from data_analyzer import DataAnalyzer
    from graph_plotter import GraphPlotter
    from graph_interpreter import GraphInterpreter

    try:
        # Step 1: Analyze data
        analyzer = DataAnalyzer("data/wound_data.csv")
        analysis_results = analyzer.analyze()

        # Step 2: Generate plots
        plotter = GraphPlotter(analysis_results)
        plot_paths = plotter.plot()

        # Step 3: Interpret plots
        interpreter = GraphInterpreter(plot_paths)
        interpretations = interpreter.interpret()

        # Step 4: Generate hypotheses
        generator = HypothesisGenerator(analysis_results, interpretations)
        hypotheses = generator.generate()

        # Print hypotheses
        print("Generated Hypotheses:")
        for hyp in hypotheses:
            print(hyp)
    except Exception as e:
        print(f"Error: {str(e)}")
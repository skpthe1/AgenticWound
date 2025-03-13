import os
from dotenv import load_dotenv
from agents.data_analyzer import DataAnalyzer
from agents.graph_plotter import GraphPlotter
from agents.graph_interpreter import GraphInterpreter
from agents.hypothesis_generator import HypothesisGenerator
from agents.hypothesis_validator import HypothesisValidator
from agents.summary_generator import SummaryGenerator

# Load environment variables
load_dotenv()

def main():
    # Step 1: Load and analyze the data
    print("Analyzing data...")
    analyzer = DataAnalyzer("data/wound_data.csv")
    analysis_results = analyzer.analyze()
    print("Data analysis complete.")

    # Step 2: Plot graphs based on analysis results
    print("Plotting graphs...")
    plotter = GraphPlotter(analysis_results)
    plot_paths = plotter.plot()
    print("Graphs plotted successfully.")

    # Step 3: Interpret the graphs
    print("Interpreting graphs...")
    interpreter = GraphInterpreter(plot_paths)
    interpretations = interpreter.interpret()
    print("Graph interpretation complete.")

    # Step 4: Generate hypotheses based on analysis and interpretations
    print("Generating hypotheses...")
    generator = HypothesisGenerator(analysis_results, interpretations)
    hypotheses = generator.generate()
    print("Hypotheses generated.")

    # Step 5: Validate the hypotheses
    print("Validating hypotheses...")
    validator = HypothesisValidator(hypotheses, analysis_results["data_subset"])
    validations = validator.validate()
    print("Hypothesis validation complete.")

    # Step 6: Generate the executive summary
    print("Generating executive summary...")
    summary_generator = SummaryGenerator(validations)
    summary = summary_generator.generate()
    print("Executive summary generated.")

    # Step 7: Print the summary (for testing)
    print("\n=== Executive Summary ===")
    print(summary)
    print("=========================")

if __name__ == "__main__":
    main()
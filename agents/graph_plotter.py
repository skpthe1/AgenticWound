# agents/graph_plotter.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

class GraphPlotter:
    def __init__(self, analysis_results, output_dir="plots"):
        """
        Initialize the GraphPlotter with analysis results and an output directory for saving plots.

        Args:
            analysis_results (dict): Results from DataAnalyzer containing summary stats and data subset.
            output_dir (str): Directory where plots will be saved (default: "plots").
        """
        self.analysis_results = analysis_results
        self.data_subset = self.analysis_results["data_subset"]
        self.output_dir = output_dir

        # Create the output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def plot_avg_wound_area_over_time(self):
        """
        Plot a line graph showing average wound area over time (weeks) for each dressing type.
        This helps visualize trends in wound healing across different treatments.

        Returns:
            str: Path to the saved plot file.
        """
        try:
            plt.figure(figsize=(20, 10))
            sns.lineplot(x="WEEK_NUM", y="AVG_WOUND_AREA", hue="NAME", data=self.data_subset)
            plt.title("Average Wound Area Over Time by Dressing Type")
            plt.xlabel("Week")
            plt.ylabel("Average Wound Area")
            max_week = self.data_subset["WEEK_NUM"].max()
            plt.xticks(ticks=range(0, max_week + 1, 10), 
                      labels=[f"Week {i}" for i in range(0, max_week + 1, 10)], 
                      rotation=45, fontsize=8)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()

            # Save the plot
            plot_path = os.path.join(self.output_dir, "avg_wound_area_over_time.png")
            plt.savefig(plot_path)
            plt.close()
            return plot_path
        except Exception as e:
            raise Exception(f"Error plotting average wound area over time: {str(e)}")

    def plot_avg_wound_area_by_dressing(self):
        """
        Plot a bar chart showing the overall average wound area for each dressing type.
        This helps compare the effectiveness of different dressings.

        Returns:
            str: Path to the saved plot file.
        """
        try:
            plt.figure(figsize=(12, 6))
            sns.barplot(x="NAME", y="AVG_WOUND_AREA", data=self.data_subset)
            plt.title("Overall Average Wound Area by Dressing Type")
            plt.xlabel("Dressing Type")
            plt.ylabel("Average Wound Area")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            # Save the plot
            plot_path = os.path.join(self.output_dir, "avg_wound_area_by_dressing.png")
            plt.savefig(plot_path)
            plt.close()
            return plot_path
        except Exception as e:
            raise Exception(f"Error plotting overall average wound area by dressing: {str(e)}")

    def plot(self):
        """
        Generate all plots and return their file paths.
        Currently generates two plots: average wound area over time and overall average wound area by dressing.

        Returns:
            list: List of file paths to the generated plots.
        """
        plot_paths = []
        
        # Generate plot for average wound area over time
        plot_paths.append(self.plot_avg_wound_area_over_time())

        # Generate plot for overall average wound area by dressing
        plot_paths.append(self.plot_avg_wound_area_by_dressing())

        return plot_paths

# Example usage for testing
if __name__ == "__main__":
    from data_analyzer import DataAnalyzer

    try:
        # Initialize DataAnalyzer and get analysis results
        analyzer = DataAnalyzer("data/wound_data.csv")
        analysis_results = analyzer.analyze()

        # Initialize GraphPlotter and generate plots
        plotter = GraphPlotter(analysis_results)
        plot_paths = plotter.plot()
        print("Generated plots:", plot_paths)
    except Exception as e:
        print(f"Error: {str(e)}")
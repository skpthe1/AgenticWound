# agents/data_analyzer.py

# Purpose: This agent reads the wound_data.csv file, performs statistical analysis,
# and prepares data for downstream agents. It implements the Reasoning step of the ReAct pattern
# by analyzing trends and generating insights.

import pandas as pd
import os

class DataAnalyzer:
    def __init__(self, data_path):
        """
        Initialize the DataAnalyzer with the path to the CSV file.

        Args:
            data_path (str): Path to the wound_data.csv file.
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at {data_path}")
        
        self.data_path = data_path
        self.data = None
        self.data_subset = None

    def load_data(self):
        """
        Load the CSV data into a pandas DataFrame and perform initial preprocessing.
        """
        try:
            self.data = pd.read_csv(self.data_path)
            self.data["WEEK_NUM"] = self.data["WEEK"].str.extract("(\d+)").astype(int)
            self.data_subset = self.data.copy()
            
            required_columns = ["WEEK", "NAME", "TOTAL_WOUND_AREA", "WOUND_COUNT", "AVG_WOUND_AREA"]
            missing_columns = [col for col in required_columns if col not in self.data_subset.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
                
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")

    def analyze(self):
        """
        Perform statistical analysis on the data subset.
        Computes summary statistics, grouped statistics by dressing type and week,
        and returns a dictionary of results for downstream agents.

        Returns:
            dict: Contains summary stats, grouped stats, and the filtered dataset as a list of dicts.
        """
        if self.data is None or self.data_subset is None:
            self.load_data()

        summary_stats = {
            "num_rows": len(self.data_subset),
            "weeks_covered": self.data_subset["WEEK"].nunique(),
            "num_dressings": self.data_subset["NAME"].nunique(),
            "avg_total_wound_area": self.data_subset["TOTAL_WOUND_AREA"].mean(),
            "avg_wound_count": self.data_subset["WOUND_COUNT"].mean(),
            "avg_wound_area": self.data_subset["AVG_WOUND_AREA"].mean(),
            "std_total_wound_area": self.data_subset["TOTAL_WOUND_AREA"].std(),
            "missing_values": self.data_subset.isnull().sum().to_dict()
        }

        grouped_by_dressing = self.data_subset.groupby("NAME")["TOTAL_WOUND_AREA"].mean().to_dict()
        grouped_by_week = self.data_subset.groupby("WEEK")["TOTAL_WOUND_AREA"].mean().to_dict()

        # Convert DataFrame to a JSON-compatible list of dictionaries
        data_subset = self.data_subset.to_dict(orient='records')

        analysis_results = {
            "summary_stats": summary_stats,
            "grouped_by_dressing": grouped_by_dressing,
            "grouped_by_week": grouped_by_week,
            "data_subset": data_subset
        }

        return analysis_results

if __name__ == "__main__":
    try:
        analyzer = DataAnalyzer("data/wound_data.csv")
        results = analyzer.analyze()
        print("Summary Statistics:", results["summary_stats"])
        print("Grouped by Dressing:", results["grouped_by_dressing"])
        print("Grouped by Week:", results["grouped_by_week"])
    except Exception as e:
        print(f"Error: {str(e)}")
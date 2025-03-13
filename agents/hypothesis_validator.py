import pandas as pd
import re
from scipy.stats import ttest_ind, linregress, f_oneway

class HypothesisValidator:
    def __init__(self, hypotheses, data_subset):
        """Initialize with hypotheses and dataset."""
        self.hypotheses = hypotheses
        self.data_subset = data_subset.copy()
        # Convert WEEK to numeric WEEK_NUM if not already present
        if "WEEK_NUM" not in self.data_subset.columns:
            self.data_subset["WEEK_NUM"] = self.data_subset["WEEK"].str.extract(r'Week (\d+)').astype(int)
        self.data_subset["NAME"] = self.data_subset["NAME"].str.lower()
        self.available_dressings = self.data_subset["NAME"].unique()

    def validate(self):
        """Validate all hypotheses and return results."""
        validations = []
        for hypothesis in self.hypotheses:
            # Clean hypothesis text
            hyp_text = hypothesis.split(". ", 1)[-1] if ". " in hypothesis else hypothesis
            hyp_text_lower = hyp_text.lower()

            # Extract key components
            dressings, is_others = self._extract_dressings(hyp_text_lower)
            specific_week = self._extract_specific_week(hyp_text_lower)
            period = self._extract_period(hyp_text_lower)

            # Handle specific week-based hypotheses
            if specific_week is not None:
                if len(dressings) == 2:
                    validation = self._validate_at_specific_week(hyp_text, dressings, specific_week)
                elif len(dressings) == 1:
                    if "week 0" in hyp_text_lower and "after" in hyp_text_lower:
                        # Special case for Validation 4
                        validation = self._validate_consistent_decrease_after_outlier(hyp_text, dressings[0], specific_week)
                    else:
                        validation = self._validate_single_dressing_trend(hyp_text, dressings[0], specific_week)
                else:
                    validation = {
                        "hypothesis": hyp_text,
                        "test": "Not applicable",
                        "stats": "N/A",
                        "interpretation": "Invalid: Requires one or two dressings with a specific week."
                    }
            # Handle period-based or group comparison hypotheses
            elif period is not None or is_others or len(dressings) >= 1:
                # Default to full study period if none specified
                if period is None:
                    period = (self.data_subset["WEEK_NUM"].min(), self.data_subset["WEEK_NUM"].max())
                if is_others or len(dressings) > 2:
                    validation = self._validate_group_comparison(hyp_text, dressings, period, is_others)
                elif len(dressings) == 2:
                    if "reduction" in hyp_text_lower or "decrease" in hyp_text_lower:
                        validation = self._validate_rate_of_decrease(hyp_text, dressings, period)
                    else:
                        validation = self._validate_mean_over_period(hyp_text, dressings, period)
                elif len(dressings) == 1:
                    if "consistent decrease" in hyp_text_lower:
                        validation = self._validate_consistent_decrease(hyp_text, dressings[0], period)
                    else:
                        validation = self._validate_mean_over_period_single(hyp_text, dressings[0], period)
                else:
                    validation = {
                        "hypothesis": hyp_text,
                        "test": "Not applicable",
                        "stats": "N/A",
                        "interpretation": "Invalid: Requires at least one dressing with a period or group comparison."
                    }
            else:
                validation = {
                    "hypothesis": hyp_text,
                    "test": "Not applicable",
                    "stats": "N/A",
                    "interpretation": "Invalid: Unable to extract necessary information."
                }
            validations.append(validation)
        return validations

    def _extract_dressings(self, hyp_text_lower):
        """Extract dressings and detect 'other dressing types' references."""
        found_dressings = []
        is_others = "compared to other dressing types" in hyp_text_lower or "compared to other dressings" in hyp_text_lower
        for dressing in self.available_dressings:
            if dressing in hyp_text_lower and dressing not in found_dressings:
                found_dressings.append(dressing)
        return found_dressings, is_others

    def _extract_specific_week(self, hyp_text_lower):
        """Extract a specific week from the hypothesis."""
        match = re.search(r'at week (\d+)|by week (\d+)|around week (\d+)|week (\d+)', hyp_text_lower)
        if match:
            return int(match.group(1) or match.group(2) or match.group(3) or match.group(4))
        return None

    def _extract_period(self, hyp_text_lower):
        """Extract a time period from the hypothesis."""
        if "long-term period" in hyp_text_lower or "over the course of the study" in hyp_text_lower:
            return (self.data_subset["WEEK_NUM"].min(), self.data_subset["WEEK_NUM"].max())
        match = re.search(r'over a (\d+)-week period|over the first (\d+) weeks|within the first (\d+) weeks', hyp_text_lower)
        if match:
            weeks = int(match.group(1) or match.group(2) or match.group(3))
            return (0, weeks - 1)
        return None

    def _validate_at_specific_week(self, hypothesis, dressings, specific_week):
        """Validate hypotheses comparing two dressings at a specific week."""
        dressing1, dressing2 = dressings
        df_week = self.data_subset[self.data_subset["WEEK_NUM"] == specific_week]
        group1 = df_week[df_week["NAME"] == dressing1]["AVG_WOUND_AREA"]
        group2 = df_week[df_week["NAME"] == dressing2]["AVG_WOUND_AREA"]
        if group1.empty or group2.empty:
            return {
                "hypothesis": hypothesis,
                "test": "Not applicable",
                "stats": "N/A",
                "interpretation": f"No data for {dressing1} or {dressing2} at week {specific_week}."
            }
        if len(group1) < 2 or len(group2) < 2:
            avg1, avg2 = group1.mean(), group2.mean()
            interpretation = (
                f"At week {specific_week}, {dressing1}: {avg1:.4f} (n={len(group1)}), {dressing2}: {avg2:.4f} (n={len(group2)}). "
                f"{'Supports' if avg1 < avg2 else 'Does not support'} lower wound area for {dressing1} (direct comparison, insufficient data for t-test)."
            )
            return {
                "hypothesis": hypothesis,
                "test": "Direct comparison",
                "stats": f"{dressing1}: {avg1:.4f}, {dressing2}: {avg2:.4f}",
                "interpretation": interpretation
            }
        t_stat, p_val = ttest_ind(group1, group2, equal_var=False)
        interpretation = (
            f"T-test at week {specific_week}: t-stat = {t_stat:.4f}, p = {p_val:.4f}, "
            f"{dressing1} (n={len(group1)}), {dressing2} (n={len(group2)}). "
            f"{'Supports' if p_val < 0.05 and t_stat < 0 else 'Does not support'} significantly lower wound area for {dressing1}."
        )
        return {
            "hypothesis": hypothesis,
            "test": "T-test",
            "stats": f"t_stat = {t_stat:.4f}, p = {p_val:.4f}",
            "interpretation": interpretation
        }

    def _validate_single_dressing_trend(self, hypothesis, dressing, specific_week):
        """Validate trend for a single dressing at a specific week."""
        df_dressing = self.data_subset[self.data_subset["NAME"] == dressing]
        if specific_week not in df_dressing["WEEK_NUM"].values or specific_week - 1 not in df_dressing["WEEK_NUM"].values:
            return {
                "hypothesis": hypothesis,
                "test": "Not applicable",
                "stats": "N/A",
                "interpretation": f"Insufficient data for {dressing} at week {specific_week} or previous week."
            }
        value_at_week = df_dressing[df_dressing["WEEK_NUM"] == specific_week]["AVG_WOUND_AREA"].values[0]
        value_previous_week = df_dressing[df_dressing["WEEK_NUM"] == specific_week - 1]["AVG_WOUND_AREA"].values[0]
        if value_at_week > value_previous_week:
            interpretation = (
                f"For {dressing}, AVG_WOUND_AREA at week {specific_week} ({value_at_week:.4f}) > "
                f"week {specific_week - 1} ({value_previous_week:.4f}). Supports increase."
            )
        else:
            interpretation = (
                f"For {dressing}, AVG_WOUND_AREA at week {specific_week} ({value_at_week:.4f}) <= "
                f"week {specific_week - 1} ({value_previous_week:.4f}). Does not support increase."
            )
        return {
            "hypothesis": hypothesis,
            "test": "Direct comparison",
            "stats": f"week {specific_week}: {value_at_week:.4f}, week {specific_week - 1}: {value_previous_week:.4f}",
            "interpretation": interpretation
        }

    def _validate_group_comparison(self, hypothesis, dressings, period, is_others=False):
        """Validate group comparison over a period with enhanced ANOVA support."""
        start_week, end_week = period
        df_period = self.data_subset[self.data_subset["WEEK_NUM"].between(start_week, end_week)]
        
        if is_others:
            group1 = dressings
            group2 = [d for d in self.available_dressings if d not in group1]
        else:
            group1 = dressings[:1]
            group2 = [d for d in dressings[1:]] if len(dressings) > 1 else [d for d in self.available_dressings if d not in group1]

        group1_data = df_period[df_period["NAME"].isin(group1)]["AVG_WOUND_AREA"]
        group2_data = df_period[df_period["NAME"].isin(group2)]["AVG_WOUND_AREA"]
        
        if group1_data.empty or group2_data.empty:
            return {
                "hypothesis": hypothesis,
                "test": "Not applicable",
                "stats": "N/A",
                "interpretation": f"No data for group1 or group2 over weeks {start_week}-{end_week}."
            }
        
        if len(group1) == 1 and len(group2) == 1:
            t_stat, p_val = ttest_ind(group1_data, group2_data, equal_var=False)
            interpretation = (
                f"T-test for {', '.join(group1)} (n={len(group1_data)}) vs {', '.join(group2)} (n={len(group2_data)}) over weeks {start_week}-{end_week}: "
                f"t-stat = {t_stat:.4f}, p = {p_val:.4f}. "
                f"{'Supports' if p_val < 0.05 and t_stat < 0 else 'Does not support'} significantly lower mean for {', '.join(group1)}."
            )
            return {
                "hypothesis": hypothesis,
                "test": "T-test",
                "stats": f"t_stat = {t_stat:.4f}, p = {p_val:.4f}",
                "interpretation": interpretation
            }
        
        groups = [df_period[df_period["NAME"] == d]["AVG_WOUND_AREA"] for d in (group1 + group2) if not df_period[df_period["NAME"] == d].empty]
        if len(groups) >= 2:
            f_stat, p_val = f_oneway(*groups)
            interpretation = (
                f"ANOVA for {', '.join(group1 + group2)} over weeks {start_week}-{end_week}: "
                f"F-stat = {f_stat:.4f}, p = {p_val:.4f}, sample sizes = {[len(g) for g in groups]}. "
                f"{'Supports' if p_val < 0.05 else 'Does not support'} significant differences among dressings."
            )
            return {
                "hypothesis": hypothesis,
                "test": "ANOVA",
                "stats": f"F_stat = {f_stat:.4f}, p = {p_val:.4f}",
                "interpretation": interpretation
            }
        
        return {
            "hypothesis": hypothesis,
            "test": "Not applicable",
            "stats": "N/A",
            "interpretation": f"Insufficient data for ANOVA comparison over weeks {start_week}-{end_week}."
        }

    def _validate_rate_of_decrease(self, hypothesis, dressings, period):
        """Validate rate of decrease for two dressings over a period."""
        dressing1, dressing2 = dressings
        start_week, end_week = period
        df_dressing1 = self.data_subset[(self.data_subset["NAME"] == dressing1) & 
                                       (self.data_subset["WEEK_NUM"].between(start_week, end_week))]
        df_dressing2 = self.data_subset[(self.data_subset["NAME"] == dressing2) & 
                                       (self.data_subset["WEEK_NUM"].between(start_week, end_week))]
        if len(df_dressing1) < 2 or len(df_dressing2) < 2:
            return {
                "hypothesis": hypothesis,
                "test": "Not applicable",
                "stats": "N/A",
                "interpretation": f"Insufficient data for {dressing1} or {dressing2} over weeks {start_week}-{end_week}."
            }
        slope1, _, _, p_val1, _ = linregress(df_dressing1["WEEK_NUM"], df_dressing1["AVG_WOUND_AREA"])
        slope2, _, _, p_val2, _ = linregress(df_dressing2["WEEK_NUM"], df_dressing2["AVG_WOUND_AREA"])
        if slope1 < slope2 and p_val1 < 0.05:
            interpretation = (
                f"Regression for {dressing1} (n={len(df_dressing1)}): slope = {slope1:.4f}, p = {p_val1:.4f}. "
                f"For {dressing2} (n={len(df_dressing2)}): slope = {slope2:.4f}, p = {p_val2:.4f}. "
                f"Supports greater reduction for {dressing1}."
            )
        else:
            interpretation = (
                f"Regression for {dressing1} (n={len(df_dressing1)}): slope = {slope1:.4f}, p = {p_val1:.4f}. "
                f"For {dressing2} (n={len(df_dressing2)}): slope = {slope2:.4f}, p = {p_val2:.4f}. "
                f"Does not support greater reduction for {dressing1}."
            )
        return {
            "hypothesis": hypothesis,
            "test": "Linear regression",
            "stats": f"slope_{dressing1} = {slope1:.4f}, p = {p_val1:.4f}; slope_{dressing2} = {slope2:.4f}, p = {p_val2:.4f}",
            "interpretation": interpretation
        }

    def _validate_consistent_decrease(self, hypothesis, dressing, period):
        """Validate if a dressing shows a consistent decrease over a period with enhanced variability check."""
        start_week, end_week = period
        df_dressing = self.data_subset[(self.data_subset["NAME"] == dressing) & 
                                      (self.data_subset["WEEK_NUM"].between(start_week, end_week))]
        if len(df_dressing) < 3:
            return {
                "hypothesis": hypothesis,
                "test": "Not applicable",
                "stats": "N/A",
                "interpretation": f"Insufficient data for {dressing} over weeks {start_week}-{end_week}."
            }
        decreases = (df_dressing["AVG_WOUND_AREA"].diff() < 0).sum()
        total_weeks = len(df_dressing) - 1
        diffs = df_dressing["AVG_WOUND_AREA"].diff().dropna()
        variability = diffs.std() if len(diffs) > 1 else 0
        if decreases == total_weeks and variability < 1.0:
            interpretation = (
                f"For {dressing} (n={len(df_dressing)}), AVG_WOUND_AREA decreased consistently over {decreases}/{total_weeks} weeks, "
                f"variability (std dev) = {variability:.4f}. Supports consistent decrease."
            )
        else:
            interpretation = (
                f"For {dressing} (n={len(df_dressing)}), AVG_WOUND_AREA decreased in {decreases}/{total_weeks} weeks, "
                f"variability (std dev) = {variability:.4f}. Does not support consistent decrease."
            )
        return {
            "hypothesis": hypothesis,
            "test": "Trend analysis",
            "stats": f"Decreases: {decreases}/{total_weeks}, Variability: {variability:.4f}",
            "interpretation": interpretation
        }

    def _validate_consistent_decrease_after_outlier(self, hypothesis, dressing, outlier_week):
        """Validate consistent decrease after an outlier week and comparison to week 0."""
        df_dressing = self.data_subset[self.data_subset["NAME"] == dressing]
        if 0 not in df_dressing["WEEK_NUM"].values or outlier_week not in df_dressing["WEEK_NUM"].values:
            return {
                "hypothesis": hypothesis,
                "test": "Not applicable",
                "stats": "N/A",
                "interpretation": f"Insufficient data for {dressing} at week 0 or week {outlier_week}."
            }
        
        # Trend analysis after outlier week
        df_post_outlier = df_dressing[df_dressing["WEEK_NUM"] > outlier_week]
        if len(df_post_outlier) < 3:
            return {
                "hypothesis": hypothesis,
                "test": "Not applicable",
                "stats": "N/A",
                "interpretation": f"Insufficient data for {dressing} after week {outlier_week}."
            }
        decreases = (df_post_outlier["AVG_WOUND_AREA"].diff() < 0).sum()
        total_weeks = len(df_post_outlier) - 1
        diffs = df_post_outlier["AVG_WOUND_AREA"].diff().dropna()
        variability = diffs.std() if len(diffs) > 1 else 0
        
        # T-test vs week 0
        week_0_value = df_dressing[df_dressing["WEEK_NUM"] == 0]["AVG_WOUND_AREA"]
        post_outlier_values = df_post_outlier["AVG_WOUND_AREA"]
        if week_0_value.empty or post_outlier_values.empty:
            t_stat, p_val = float('nan'), float('nan')
            t_result = "No data for T-test comparison with week 0."
        else:
            t_stat, p_val = ttest_ind(week_0_value, post_outlier_values, equal_var=False)
            t_result = (
                f"T-test vs week 0: t-stat = {t_stat:.4f}, p = {p_val:.4f}, "
                f"week 0 (n={len(week_0_value)}), post-week {outlier_week} (n={len(post_outlier_values)}). "
                f"{'Supports' if p_val < 0.05 and t_stat > 0 else 'Does not support'} significantly lower mean after week {outlier_week}."
            )
        
        if decreases == total_weeks and variability < 1.0:
            trend_result = f"Decreased consistently over {decreases}/{total_weeks} weeks, variability = {variability:.4f}. Supports consistent decrease."
        else:
            trend_result = f"Decreased in {decreases}/{total_weeks} weeks, variability = {variability:.4f}. Does not support consistent decrease."
        
        interpretation = f"Trend analysis for {dressing} after week {outlier_week}: {trend_result} {t_result}"
        return {
            "hypothesis": hypothesis,
            "test": "Trend analysis and T-test",
            "stats": f"Decreases: {decreases}/{total_weeks}, Variability: {variability:.4f}, t_stat = {t_stat:.4f}, p = {p_val:.4f}",
            "interpretation": interpretation
        }

    def _validate_mean_over_period(self, hypothesis, dressings, period):
        """Validate mean wound area comparison over a period for two dressings."""
        dressing1, dressing2 = dressings
        start_week, end_week = period
        df_period = self.data_subset[self.data_subset["WEEK_NUM"].between(start_week, end_week)]
        group1 = df_period[df_period["NAME"] == dressing1]["AVG_WOUND_AREA"]
        group2 = df_period[df_period["NAME"] == dressing2]["AVG_WOUND_AREA"]
        if group1.empty or group2.empty:
            return {
                "hypothesis": hypothesis,
                "test": "Not applicable",
                "stats": "N/A",
                "interpretation": f"No data for {dressing1} or {dressing2} over weeks {start_week}-{end_week}."
            }
        t_stat, p_val = ttest_ind(group1, group2, equal_var=False)
        interpretation = (
            f"T-test for {dressing1} (n={len(group1)}) vs {dressing2} (n={len(group2)}) over weeks {start_week}-{end_week}: "
            f"t-stat = {t_stat:.4f}, p = {p_val:.4f}. "
            f"{'Supports' if p_val < 0.05 and t_stat < 0 else 'Does not support'} significantly lower mean for {dressing1}."
        )
        return {
            "hypothesis": hypothesis,
            "test": "T-test",
            "stats": f"t_stat = {t_stat:.4f}, p = {p_val:.4f}",
            "interpretation": interpretation
        }

    def _validate_mean_over_period_single(self, hypothesis, dressing, period):
        """Validate mean wound area for a single dressing over a period."""
        start_week, end_week = period
        df_period = self.data_subset[(self.data_subset["NAME"] == dressing) & 
                                    (self.data_subset["WEEK_NUM"].between(start_week, end_week))]
        if df_period.empty:
            return {
                "hypothesis": hypothesis,
                "test": "Not applicable",
                "stats": "N/A",
                "interpretation": f"No data for {dressing} over weeks {start_week}-{end_week}."
            }
        mean_value = df_period["AVG_WOUND_AREA"].mean()
        interpretation = (
            f"For {dressing} (n={len(df_period)}), mean AVG_WOUND_AREA over weeks {start_week}-{end_week} is {mean_value:.4f}. "
            f"Data available for analysis."
        )
        return {
            "hypothesis": hypothesis,
            "test": "Mean calculation",
            "stats": f"mean = {mean_value:.4f}",
            "interpretation": interpretation
        }

if __name__ == "__main__":
    from data_analyzer import DataAnalyzer
    from graph_plotter import GraphPlotter
    from graph_interpreter import GraphInterpreter
    from hypothesis_generator import HypothesisGenerator

    try:
        analyzer = DataAnalyzer("data/wound_data.csv")
        analysis_results = analyzer.analyze()
        plotter = GraphPlotter(analysis_results)
        plot_paths = plotter.plot()
        interpreter = GraphInterpreter(plot_paths)
        interpretations = interpreter.interpret()
        generator = HypothesisGenerator(analysis_results, interpretations)
        hypotheses = generator.generate()
        validator = HypothesisValidator(hypotheses, analysis_results["data_subset"])
        validations = validator.validate()

        print("\nValidated Hypotheses:")
        for val in validations:
            print(f"Hypothesis: {val['hypothesis']}")
            print(f"Test: {val['test']}")
            print(f"Stats: {val['stats']}")
            print(f"Interpretation: {val['interpretation']}\n")
    except Exception as e:
        print(f"Error: {str(e)}")
import streamlit as st
import io
import sys
import os
import json
import pandas as pd
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai.tools import BaseTool
from crewai.llm import LLM
from crewai.crews.crew_output import CrewOutput
from crewai.tasks.task_output import TaskOutput
from agents.data_analyzer import DataAnalyzer
from agents.graph_plotter import GraphPlotter
from agents.graph_interpreter import GraphInterpreter
from agents.hypothesis_generator import HypothesisGenerator
from agents.hypothesis_validator import HypothesisValidator
from agents.summary_generator import SummaryGenerator
from config.config import AZURE_OPENAI_API_KEY, AZURE_ENDPOINT, AZURE_DEPLOYMENT_NAME, OPENAI_API_VERSION

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(page_title="Wound Analysis Dashboard", layout="wide", initial_sidebar_state="expanded")

st.title("🩺 Wound Analysis Dashboard")
st.sidebar.header("Control Panel")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload Wound Data (CSV)", type="csv", help="Upload a CSV file containing wound data for analysis.")

# Initialize session state
if 'file_path' not in st.session_state: st.session_state.file_path = None
if 'analysis_results' not in st.session_state: st.session_state.analysis_results = None
if 'data_subset' not in st.session_state: st.session_state.data_subset = None
if 'analysis_summary' not in st.session_state: st.session_state.analysis_summary = None
if 'plot_paths' not in st.session_state: st.session_state.plot_paths = None
if 'interpretations' not in st.session_state: st.session_state.interpretations = None
if 'hypotheses' not in st.session_state: st.session_state.hypotheses = None
if 'validations' not in st.session_state: st.session_state.validations = None
if 'summary' not in st.session_state: st.session_state.summary = None

# Handle file upload and validation
if uploaded_file is not None:
    file_path = "temp_wound_data.csv"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.session_state.file_path = file_path
    df = pd.read_csv(file_path)
    required_columns = ['TOTAL_WOUND_AREA', 'WOUND_COUNT', 'WEEK', 'NAME']
    if not all(col in df.columns for col in required_columns):
        st.error(f"The CSV file must contain the following columns: {', '.join(required_columns)}")
        st.stop()
    with st.expander("📊 Data Summary", expanded=True):
        st.write(f"**Number of Rows:** {df.shape[0]}")
        st.write(f"**Number of Columns:** {df.shape[1]}")
        st.write("**Preview (First 5 Rows):**")
        st.dataframe(df.head())
else:
    st.info("Please upload a CSV file to begin analysis.")
    st.stop()

# Define Azure OpenAI LLM configuration
azure_llm = LLM(
    model=f"azure/{AZURE_DEPLOYMENT_NAME}",
    base_url=AZURE_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=OPENAI_API_VERSION
)

# Define Crew AI agents
data_analyzer_agent = Agent(
    role="Data Analyst",
    goal="Analyze wound data and provide statistical insights",
    backstory="Expert in statistical analysis of medical data",
    verbose=True,
    llm=azure_llm
)

graph_plotter_agent = Agent(
    role="Graph Plotter",
    goal="Generate visual representations of wound data",
    backstory="Specialist in data visualization",
    verbose=True,
    llm=azure_llm
)

graph_interpreter_agent = Agent(
    role="Graph Interpreter",
    goal="Interpret generated graphs to provide insights",
    backstory="Experienced in translating visual data into medical insights",
    verbose=True,
    llm=azure_llm
)

hypothesis_generator_agent = Agent(
    role="Hypothesis Generator",
    goal="Generate hypotheses based on data analysis and graph interpretations",
    backstory="Skilled in forming testable hypotheses from complex datasets",
    verbose=True,
    llm=azure_llm
)

hypothesis_validator_agent = Agent(
    role="Hypothesis Validator",
    goal="Validate hypotheses using statistical methods",
    backstory="Expert in statistical validation and hypothesis testing",
    verbose=True,
    llm=azure_llm
)

analysis_summary_agent = Agent(
    role="Analysis Summary Generator",
    goal="Generate an initial summary of data analysis results",
    backstory="Specialist in summarizing statistical data clearly",
    verbose=True,
    llm=azure_llm
)

summary_generator_agent = Agent(
    role="Executive Summary Generator",
    goal="Create a holistic executive summary of the entire analysis process",
    backstory="Proficient in crafting comprehensive summaries for executives",
    verbose=True,
    llm=azure_llm
)

# Define custom tools
class DataAnalyzerTool(BaseTool):
    name: str = "Data Analyzer"
    description: str = "Analyzes wound data from a CSV file and returns summary statistics"
    analyzer: DataAnalyzer

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, file_path):
        super().__init__(analyzer=DataAnalyzer(file_path))

    def _run(self, *args, **kwargs):
        full_results = self.analyzer.analyze()
        if not isinstance(full_results['data_subset'], pd.DataFrame):
            full_results['data_subset'] = pd.DataFrame(full_results['data_subset'])
        st.session_state.data_subset = full_results['data_subset']
        summary_dict = {
            'summary_stats': full_results['summary_stats'],
            'grouped_by_dressing': full_results['grouped_by_dressing'],
            'grouped_by_week': full_results['grouped_by_week']
        }
        return json.dumps(summary_dict)

class GraphPlotterTool(BaseTool):
    name: str = "Graph Plotter"
    description: str = "Plots graphs based on analyzed wound data"
    plotter: GraphPlotter

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, analysis_results):
        full_analysis_results = analysis_results.copy()
        full_analysis_results['data_subset'] = st.session_state.data_subset
        super().__init__(plotter=GraphPlotter(full_analysis_results))
    def _run(self, *args, **kwargs):
        plot_paths = self.plotter.plot()
        return json.dumps(plot_paths)


    
class GraphInterpreterTool(BaseTool):
    name: str = "Graph Interpreter"
    description: str = "Interprets plotted graphs to provide insights"
    interpreter: GraphInterpreter

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, plot_paths):
        super().__init__(interpreter=GraphInterpreter(plot_paths))

    def _run(self, *args, **kwargs):
        interpretations = self.interpreter.interpret()
        return json.dumps(interpretations)

class HypothesisGeneratorTool(BaseTool):
    name: str = "Hypothesis Generator"
    description: str = "Generates hypotheses from analysis results and interpretations"
    generator: HypothesisGenerator

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, analysis_results, interpretations):
        super().__init__(generator=HypothesisGenerator(analysis_results, interpretations))

    def _run(self, *args, **kwargs):
        hypotheses = self.generator.generate()
        if not isinstance(hypotheses, list):
            hypotheses = [hypotheses]
        return json.dumps(hypotheses)

class HypothesisValidatorTool(BaseTool):
    name: str = "Hypothesis Validator"
    description: str = "Validates hypotheses using statistical methods based on the wound data subset"
    validator: HypothesisValidator

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, hypotheses, data_subset):
        super().__init__(validator=HypothesisValidator(hypotheses, data_subset))

    def _run(self, *args, **kwargs):
        validations = self.validator.validate()
        return json.dumps(validations)

class SummaryGeneratorTool(BaseTool):
    name: str = "Summary Generator"
    description: str = "Generates summaries based on provided data"
    summary_gen: SummaryGenerator

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, summary_gen):
        super().__init__(summary_gen=summary_gen)

    def _run(self, *args, **kwargs):
        return self.summary_gen.generate()  # Use LLM for both initial and executive summaries

# Parsing function with extra care
def parse_task_output(output, expected_type):
    
    

    if isinstance(output, TaskOutput):
        raw_output = output.raw if hasattr(output, 'raw') else str(output)
    elif isinstance(output, CrewOutput):
        if hasattr(output, 'tasks_output') and output.tasks_output:
            raw_output = output.tasks_output[0].raw
        else:
            raw_output = str(output)
    else:
        raw_output = output

    if expected_type == dict:
        try:
            parsed_output = json.loads(raw_output)
            if not isinstance(parsed_output, dict):
                raise ValueError("Parsed output is not a dictionary")
        except (json.JSONDecodeError, ValueError) as e:
            st.error(f"Failed to parse output as dict: {e}")
            
            try:
                if isinstance(raw_output, str) and raw_output.strip().startswith('{'):
                    import ast
                    parsed_output = ast.literal_eval(raw_output)
                    if isinstance(parsed_output, dict):
                        return parsed_output
                raise ValueError("Output is not a valid dictionary format")
            except (ValueError, SyntaxError) as e2:
                st.error(f"Fallback parsing failed: {e2}")
                return None
    elif expected_type == str:
        parsed_output = raw_output if isinstance(raw_output, str) else str(raw_output)
    elif expected_type == list:
        try:
            if isinstance(raw_output, str) and not raw_output.startswith('['):
                parsed_output = [line.strip() for line in raw_output.split('\n') if line.strip()]
            else:
                parsed_output = json.loads(raw_output)
            if not isinstance(parsed_output, list):
                raise ValueError("Parsed output is not a list")
        except (json.JSONDecodeError, ValueError) as e:
            st.error(f"Failed to parse output as list: {e}")
            
            return None
    else:
        parsed_output = raw_output

    return parsed_output

# Sidebar buttons
if st.sidebar.button("🔍 Analyze Data"):
    if st.session_state.file_path:
        # Capture console output
        console_output = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = console_output

        try:
            # Analyze data step
            analyzer_tool = DataAnalyzerTool(st.session_state.file_path)
            analyze_task = Task(
                description="Analyze the wound data from the uploaded CSV file and provide summary statistics in a dictionary format with keys 'summary_stats', 'grouped_by_dressing', and 'grouped_by_week'.",
                agent=data_analyzer_agent,
                expected_output="Dictionary with summary statistics and grouped data",
                tools=[analyzer_tool]
            )
            analysis_crew = Crew(agents=[data_analyzer_agent], tasks=[analyze_task])
            analysis_results_raw = analysis_crew.kickoff()
            analysis_results = parse_task_output(analysis_results_raw, dict)
            if analysis_results is None:
                st.sidebar.error("Failed to parse analysis results.")
            else:
                st.session_state.analysis_results = analysis_results
                # Generate initial summary with LLM using bold Markdown
                summary_gen = SummaryGenerator(validations=[{"hypothesis": "Initial analysis", "test": "N/A", "stats": "N/A", "interpretation": "N/A"}],
                                               analysis_results=st.session_state.analysis_results)
                summary_tool = SummaryGeneratorTool(summary_gen)
                summary_task = Task(
                    description=f"""
                    Craft a professional yet engaging initial summary of the wound data analysis based on the provided analysis_results dictionary:
                    - **1. Data Overview**: Highlight the scale of the study with {analysis_results['summary_stats']['num_rows']} rows, {analysis_results['summary_stats']['weeks_covered']} weeks, and {analysis_results['summary_stats']['num_dressings']} dressings, setting the context in a concise paragraph.
                    - **2. Key Statistics**: Present the average wound area of {analysis_results['summary_stats']['avg_wound_area']:.2f} cm², standard deviation of {analysis_results['summary_stats']['std_total_wound_area']:.2f}, and a decreasing wound count trend (average {analysis_results['summary_stats']['avg_wound_count']:.2f} wounds), noting dressing effectiveness variations in a narrative style.
                    - **3. Wound Area Variability**: Describe the range from {min(analysis_results['grouped_by_week'].values()):.2f} cm² (Week {min(analysis_results['grouped_by_week'], key=analysis_results['grouped_by_week'].get)}) to {max(analysis_results['grouped_by_week'].values()):.2f} cm² (Week {max(analysis_results['grouped_by_week'], key=analysis_results['grouped_by_week'].get)}), with examples in a flowing sentence.
                    - **4. Dressing Performance**: Discuss average wound areas for key dressings (e.g., {', '.join([f'{k}: {v:.2f} cm²' for k, v in list(analysis_results['grouped_by_dressing'].items())[:3]])}) and top performers ({', '.join([f'{k}: {v:.2f} cm²' for k, v in sorted(analysis_results['grouped_by_dressing'].items(), key=lambda x: x[1])[:3]])}) in a professional tone.
                    - **5. Healing Trends**: Narrate the decreasing trend from {analysis_results['grouped_by_week']['Week 0']:.2f} cm² (Week 0) to {analysis_results['grouped_by_week'][f'Week {analysis_results['summary_stats']['weeks_covered']}']:.2f} cm² (Week {analysis_results['summary_stats']['weeks_covered']}), with examples at Week 1 ({analysis_results['grouped_by_week']['Week 1']:.2f} cm²) and Week 10 ({analysis_results['grouped_by_week']['Week 10']:.2f} cm²).
                    - **6. Specific Example**: Highlight a notable decrease between Week 0 ({analysis_results['grouped_by_week']['Week 0']:.2f} cm²) and Week 1 ({analysis_results['grouped_by_week']['Week 1']:.2f} cm²), a drop of {(analysis_results['grouped_by_week']['Week 0'] - analysis_results['grouped_by_week']['Week 1']):.2f} cm², in a compelling sentence.
                    Format as plain text with bolded section titles using Markdown syntax (e.g., `**1. Data Overview**`) followed by a paragraph, blending professionalism with an engaging, non-monotonous tone, keeping all specified numbers intact.
                    """,
                    agent=analysis_summary_agent,
                    expected_output="String with a detailed summary",
                    tools=[summary_tool]
                )
                summary_crew = Crew(agents=[analysis_summary_agent], tasks=[summary_task])
                summary_results_raw = summary_crew.kickoff()
                summary = parse_task_output(summary_results_raw, str)
                if summary is None or not isinstance(summary, str):
                    st.sidebar.error("Failed to generate analysis summary.")
                else:
                    st.session_state.analysis_summary = summary
                    st.sidebar.success("Data analyzed and summarized successfully!")
        finally:
            # Restore stdout and store console output in session state
            sys.stdout = original_stdout
            console_logs = console_output.getvalue()
            console_output.close()
            st.session_state.console_logs = console_logs
    else:
        st.sidebar.error("Please upload a CSV file first.")

if st.session_state.analysis_results:
    if st.sidebar.button("📈 Plot Graphs"):
        if st.session_state.data_subset is None:
            st.sidebar.error("Data subset not available. Please analyze data first.")
        else:
            # Capture console output
            console_output = io.StringIO()
            original_stdout = sys.stdout
            sys.stdout = console_output

            try:
                plotter_tool = GraphPlotterTool(st.session_state.analysis_results)
                plot_task = Task(
                    description="Generate plots for wound area trends over time and by dressing type based on the analysis results.",
                    agent=graph_plotter_agent,
                    expected_output="List of file paths to the generated plots",
                    tools=[plotter_tool]
                )
                plot_crew = Crew(agents=[graph_plotter_agent], tasks=[plot_task])
                plot_paths_raw = plot_crew.kickoff()
                plot_paths = parse_task_output(plot_paths_raw, list)
                if plot_paths is None:
                    st.sidebar.error("Failed to parse plot paths.")
                else:
                    st.session_state.plot_paths = plot_paths
                    st.sidebar.success("Graphs plotted successfully!")
            finally:
                # Restore stdout and store console output in session state
                sys.stdout = original_stdout
                console_logs = console_output.getvalue()
                console_output.close()
                st.session_state.console_logs = console_logs
else:
    st.sidebar.write("Please analyze data before plotting graphs.")

if st.session_state.plot_paths:
    if st.sidebar.button("🔎 Interpret Graphs"):
        # Capture console output
        console_output = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = console_output

        try:
            interpreter_tool = GraphInterpreterTool(st.session_state.plot_paths)
            interpret_task = Task(
                description="Interpret the generated graphs to provide insights on wound healing trends and dressing effectiveness.",
                agent=graph_interpreter_agent,
                expected_output="List of dictionaries with plot names and their interpretations",
                tools=[interpreter_tool]
            )
            interpret_crew = Crew(agents=[graph_interpreter_agent], tasks=[interpret_task])
            interpretations_raw = interpret_crew.kickoff()
            interpretations = parse_task_output(interpretations_raw, list)
            if interpretations is None:
                st.sidebar.error("Failed to parse graph interpretations.")
            else:
                st.session_state.interpretations = interpretations
                st.sidebar.success("Graphs interpreted successfully!")
        finally:
            # Restore stdout and store console output in session state
            sys.stdout = original_stdout
            console_logs = console_output.getvalue()
            console_output.close()
            st.session_state.console_logs = console_logs
else:
    st.sidebar.write("Please plot graphs before interpreting them.")

if st.session_state.analysis_results and st.session_state.interpretations:
    if st.sidebar.button("💡 Generate Hypotheses"):
        # Capture console output
        console_output = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = console_output

        try:
            hypothesis_tool = HypothesisGeneratorTool(st.session_state.analysis_results, st.session_state.interpretations)
            hypothesis_task = Task(
                description="Generate plausible hypotheses about wound healing based on the data analysis and graph interpretations.",
                agent=hypothesis_generator_agent,
                expected_output="List of hypothesis strings",
                tools=[hypothesis_tool]
            )
            hypothesis_crew = Crew(agents=[hypothesis_generator_agent], tasks=[hypothesis_task])
            hypotheses_raw = hypothesis_crew.kickoff()
            hypotheses = parse_task_output(hypotheses_raw, list)
            if hypotheses is None:
                st.sidebar.error("Failed to parse hypotheses.")
            else:
                st.session_state.hypotheses = hypotheses
                st.sidebar.success("Hypotheses generated successfully!")
        finally:
            # Restore stdout and store console output in session state
            sys.stdout = original_stdout
            console_logs = console_output.getvalue()
            console_output.close()
            st.session_state.console_logs = console_logs
else:
    st.sidebar.write("Please analyze data and interpret graphs before generating hypotheses.")

if st.session_state.hypotheses and st.session_state.data_subset is not None:
    if st.sidebar.button("✅ Validate Hypotheses"):
        # Capture console output
        console_output = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = console_output

        try:
            validator_tool = HypothesisValidatorTool(st.session_state.hypotheses, st.session_state.data_subset)
            validate_task = Task(
                description="Validate the generated hypotheses using statistical methods based on the wound data subset, returning a list of dictionaries with keys 'hypothesis', 'test', 'stats', and 'interpretation'.",
                agent=hypothesis_validator_agent,
                expected_output="List of dictionaries with validation results",
                tools=[validator_tool]
            )
            validate_crew = Crew(agents=[hypothesis_validator_agent], tasks=[validate_task])
            validations_raw = validate_crew.kickoff()
            validations = parse_task_output(validations_raw, list)
            if validations is None:
                st.sidebar.error("Failed to parse hypothesis validations.")
            else:
                st.session_state.validations = validations
                st.sidebar.success("Hypotheses validated successfully!")
        finally:
            # Restore stdout and store console output in session state
            sys.stdout = original_stdout
            console_logs = console_output.getvalue()
            console_output.close()
            st.session_state.console_logs = console_logs
else:
    st.sidebar.write("Please generate hypotheses and ensure data subset is available before validation.")

if st.session_state.validations:
    if st.sidebar.button("📝 Generate Summary"):
        # Capture console output
        console_output = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = console_output

        try:
            summary_gen = SummaryGenerator(
                analysis_results=st.session_state.analysis_results,
                plot_paths=st.session_state.plot_paths,
                interpretations=st.session_state.interpretations,
                hypotheses=st.session_state.hypotheses,
                validations=st.session_state.validations
            )
            summary_tool = SummaryGeneratorTool(summary_gen)
            summary_task = Task(
                description="""
                Craft an engaging and professional executive summary for a study on diabetic foot ulcer treatments, using all available analysis steps to provide a detailed, impactful narrative:
                - Start with a compelling introduction that underscores the study's critical purpose—enhancing wound care for diabetic patients—and its rigorous approach, analyzing 1655 records over 248 weeks across 14 dressings to uncover treatment efficacy patterns.
                - Synthesize key discoveries from the data, graphs, and validations, diving into standout outcomes like top-performing dressings (e.g., Aquacel Foam Lite at 3.93 cm²), unexpected trends such as early healing surges versus long-term stabilization, and performance gaps (e.g., Aquacel Ag+ Extra at 69.94 cm²), weaving a story of what worked and why.
                - Highlight healing patterns revealed by graphs, detailing the dramatic early drop from 781.76 cm² in Week 0 to 473.18 cm² in Week 1, sustained progress to 6.00 cm² by Week 248, and variability (e.g., 192.87 cm² at Week 10), using simple language to paint a vivid picture of recovery over time.
                - Summarize the hypotheses tested—such as Aquacel Foam’s long-term edge or Duoderm Gel’s early gains—and their validated outcomes, explaining how these insights refine our understanding of dressing impacts on wound healing trajectories.
                - Conclude with actionable insights for wound care decisions, balancing proven early benefits (e.g., a 308.58 cm² drop in Week 1) with the need for tailored long-term strategies, urging further research into why some dressings falter over time and how patient-specific factors might optimize outcomes, all tailored for healthcare executives to drive practical improvements.
                Use the full dataset and analysis outputs to tell a clear, impactful story about wound healing trends and treatment implications. Format as a detailed Markdown string (400-500 words) with section headers (e.g., `### Introduction`), blending professionalism with an engaging, number-rich narrative.
                """,
                agent=summary_generator_agent,
                expected_output="String with a holistic executive summary",
                tools=[summary_tool]
            )
            summary_crew = Crew(agents=[summary_generator_agent], tasks=[summary_task])
            summary_raw = summary_crew.kickoff()
            summary = parse_task_output(summary_raw, str)
            if summary is None:
                st.sidebar.error("Failed to parse executive summary.")
            else:
                st.session_state.summary = summary
                st.sidebar.success("Executive summary generated successfully!")
        finally:
            # Restore stdout and store console output in session state
            sys.stdout = original_stdout
            console_logs = console_output.getvalue()
            console_output.close()
            st.session_state.console_logs = console_logs
else:
    st.sidebar.write("Please validate hypotheses before generating the executive summary.")

# Main output area
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Analysis Summary", "Graphs", "Interpretations", "Hypotheses", "Hypothesis Validations", "Executive Summary"])

with tab1:
    if st.session_state.analysis_summary:
        st.subheader("Analysis Summary")
        st.markdown(st.session_state.analysis_summary)
    else:
        st.write("Analysis summary will appear here after data analysis.")

with tab2:
    if st.session_state.plot_paths:
        st.subheader("Generated Graphs")
        for plot_path in st.session_state.plot_paths:
            if os.path.exists(plot_path):
                st.image(plot_path, caption=os.path.basename(plot_path), use_column_width=True)
            else:
                st.error(f"Plot file not found: {plot_path}")
    else:
        st.write("Graphs will appear here after plotting.")

with tab3:
    if st.session_state.interpretations:
        st.subheader("Graph Interpretations")
        for interp in st.session_state.interpretations:
            st.write(f"**Plot:** {interp['plot']}")
            st.write(f"**Interpretation:** {interp['interpretation']}")
            st.divider()
    else:
        st.write("Interpretations will appear here after graph interpretation.")

with tab4:
    if st.session_state.hypotheses:
        st.subheader("Generated Hypotheses")
        for idx, hypothesis in enumerate(st.session_state.hypotheses, 1):
            cleaned_hypothesis = hypothesis.split('. ', 1)[-1] if '. ' in hypothesis and hypothesis[0].isdigit() else hypothesis
            st.write(f"**Hypothesis {idx}:** {cleaned_hypothesis}")
    else:
        st.write("Hypotheses will appear here after generation.")
with tab5:
    if st.session_state.validations:
        st.subheader("Hypothesis Validations")
        for idx, validation in enumerate(st.session_state.validations, 1):
            st.write(f"**Validation {idx}:**")
            st.write(f"- **Hypothesis:** {validation['hypothesis']}")
            st.write(f"- **Test:** {validation['test']}")
            st.write(f"- **Stats:** {validation['stats']}")
            st.write(f"- **Interpretation:** {validation['interpretation']}")
            st.divider()
    else:
        st.write("Validations will appear here after hypothesis validation.")

with tab6:
    if st.session_state.summary:
        st.markdown(st.session_state.summary)
    else:
        st.write("Executive summary will appear here after generation.")
if 'console_logs' in st.session_state and st.session_state.console_logs:
    with st.container():
        st.subheader("Agent Workflow Console Output")
        st.text_area("", st.session_state.console_logs, height=300, key="console_output")

# main.py
from crewai import Crew, Agent, Task, Process
from tools.forecasting_tools import ForecastingTools
from textwrap import dedent
import os
import pandas as pd
import json
from crewai import Crew, Agent, Task
from openai import OpenAI

os.environ["OPENAI_API_KEY"] = ""
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) 



FILE_PATH = "airpassengers.csv"

# Agents
forecasting_agent = Agent(
    role="Forecaster",
    goal="Generate accurate forecasts using Prophet.",
    backstory="An expert in time series forecasting for transportation data.",
    allow_delegation=True
)

evaluation_agent = Agent(
    role="Evaluator",
    goal="Evaluate the forecast results critically.",
    backstory="Ensures all forecasts are reliable before use.",
    allow_delegation=False
)

orchestration_agent = Agent(
    role="Orchestrator",
    goal="Oversee the forecasting pipeline.",
    backstory="Supervises agents and ensures high-quality output.",
    allow_delegation=True
)

# Prepare data and forecast
df = ForecastingTools.think_and_prepare_data(FILE_PATH)
forecast_df = ForecastingTools.forecast_with_prophet(df)

# Print forecast
print("\n📈 Forecast Output:\n", forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12))

# Define tasks
forecast_task = Task(
    description=dedent("""
        Use the prepared dataset and apply Prophet model to forecast passenger count.
        Generate a forecast for the next 12 months and return the dataframe.
    """),
    expected_output="Forecasted dataframe with 'ds' and 'yhat' values.",
    agent=forecasting_agent
)

evaluation_task = Task(
    description=dedent("""
        Analyze the forecasted data.
        If it contains null values in 'yhat' or no variation, request retraining.
        Otherwise, approve the forecast as valid.
    """),
    expected_output="Evaluation result with approval or rejection reason.",
    agent=evaluation_agent
)

# Crew setup
crew = Crew(
    agents=[forecasting_agent, evaluation_agent, orchestration_agent],
    tasks=[forecast_task, evaluation_task],
    process=Process.sequential
)

# Run crew
results = crew.kickoff()
print("\n🤖 CrewAI Results:\n", results)

# Evaluate again
valid, message = ForecastingTools.evaluate_forecast(forecast_df)
print("\n✅ Evaluation Summary:", message)

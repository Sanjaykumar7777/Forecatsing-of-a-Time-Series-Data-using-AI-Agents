# Multi Agentic Forecasting
Title:
# Self-Improving Forecasting System using Prophet and CrewAI with Agentic Battle Loop
# 1. Introduction
Forecasting is the process of predicting future outcomes based on historical data patterns. In real-world scenarios, such as stock markets, weather forecasting, and sales predictions, accurate forecasting is crucial.
Facebook Prophet is a powerful forecasting model designed to handle time-series data efficiently. It is particularly effective for datasets with:
- Strong seasonal effects
- Missing data
- Outliers
However, no single forecast is perfect on the first attempt. Hence, building a self-improving forecasting system using autonomous agents is a revolutionary step.
In this project:
- Agents fight, argue, and retry to reach the best forecast without human intervention.
- The system self-corrects based on evaluation feedback dynamically.
# 2. Problem Statement
Goal:
Design a multi-agent system where:
- A Forecasting Agent generates a forecast using Prophet.
- An Evaluation Agent critiques the forecast using evaluation metrics (MAE, RMSE).
- If evaluation is poor, the Forecasting Agent re-forecasts (changing model parameters and retrying).
- A supervising Orchestrator Agent ensures the loop continues until success or maximum attempts are reached.
This ensures that the forecasting output is of high quality without manual tuning.
# 3. System Architecture
Dataset (CSV) --> Forecasting Agent --> Evaluation Agent --> (Feedback Loop) --> Forecasting Agent (retry if needed) --> Final Output
Components:
- Forecasting Agent: Trains Prophet model, forecasts future data.
- Evaluation Agent: Calculates MAE, RMSE, criticizes if performance is poor.
- Orchestration Agent: Monitors the battle between agents, controls retry logic.
# 4. Tools & Technologies Used
Python 3: Programming Language
CrewAI: Build and orchestrate autonomous agents
Prophet: Time-series forecasting
scikit-learn: Evaluation metrics (MAE, RMSE)
Pandas: Data loading and preprocessing
Matplotlib (optional): Visualize forecasts
# 5. Step-by-Step Explanation
5.1 Setting up Tools (forecasting_tools.py): Includes loading, preprocessing, training Prophet, 
Summary:
•	Load dataset
•	Preprocess to Prophet-compatible format
•	Train model with tunable parameters
•	Generate forecast
•	Evaluate performance
5.2 Building Agents (main.py): Defines ForecastingAgent, EvaluationAgent and OrchestrationAgent using CrewAI.
5.3 Designing Agent Tasks: Tasks for forecasting and evaluation with dynamic parameter tuning.
        description=dedent(f"""
        Load the dataset 'Forecast.csv'.
        Preprocess it by renaming 'Date' -> 'ds' and 'Value' -> 'y'.
        Train a Prophet model with changepoint_prior_scale={changepoint_prior_scale} and seasonality_mode='{seasonality_mode}'.
        Generate a 30-day forecast.
        Return trained model and forecasted values.
        """),
        agent=forecast_agent
    )

# Evaluation Task
def create_evaluation_task():
    return Task(
        description=dedent("""
        Evaluate the forecast using MAE and RMSE metrics.
        If RMSE > 50 or MAE > 30, criticize the forecast harshly and request re-forecasting.
        Else, accept and praise the forecast.
        """),
        agent=evaluation_agent
    )

5.4 Battle Loop: Implements self-improving feedback loop by retrying failed forecasts.

# 6. Folder Structure
forecasting_project/
├── tools/
│   └── forecasting_tools.py
├── main.py
└── Forecast.csv
# 7. Output Demonstration (Sample Run)
Sample output includes logs where agents fight and retry:
🛡️ Starting Attempt 1
❌ Evaluation Agent rejected the forecast. Retrying...
✅ Evaluation Agent accepted the forecast!
Notice:
•	The agents literally "fight" (Evaluation criticizes).
•	Forecasting agent "thinks" and retrains automatically.
•	If Evaluation accepts, the battle ends!
# 8. Observations
- Behavior: Agents learn to improve predictions by adjusting hyperparameters.
- Fighting spirit: Evaluation Agent is strict.
- Agentic thinking: No manual intervention.

# 9. Conclusion
In this project, we built a multi-agent system capable of forecasting, self-evaluation, self-correction, and high-quality output. Using CrewAI, Prophet, and feedback loops, the system behaves like real AI teammates.
# 10. References
- GeeksForGeeks Prophet Tutorial
- CrewAI Official Documentation
- Prophet Official GitHub
- Scikit-learn Metrics Documentation

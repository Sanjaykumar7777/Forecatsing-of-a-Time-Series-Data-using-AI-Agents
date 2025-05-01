# tools/forecasting_tools.py
import pandas as pd
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
import matplotlib.pyplot as plt

class ForecastingTools:

    @staticmethod
    def think_and_prepare_data(file_path):
        df = pd.read_csv(file_path)
        df.columns = [col.strip().replace('#', '').replace('"', '') for col in df.columns]

        if 'Month' in df.columns and 'Passengers' in df.columns:
            df.rename(columns={'Month': 'ds', 'Passengers': 'y'}, inplace=True)
        else:
            raise ValueError("Dataset must have 'Month' and 'Passengers' columns.")

        df['ds'] = pd.to_datetime(df['ds'])
        return df

    @staticmethod
    def forecast_with_prophet(df):
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=12, freq='M')
        forecast = model.predict(future)

        # Plotting forecast and changepoints
        fig1 = model.plot(forecast, include_legend=True)
        add_changepoints_to_plot(fig1.gca(), model, forecast)
        fig2 = model.plot_components(forecast)

        plt.show()

        return forecast

    @staticmethod
    def evaluate_forecast(forecast_df):
        if forecast_df['yhat'].isnull().sum() > 0:
            return False, "Forecast has missing predictions."
        if forecast_df['yhat'].std() == 0:
            return False, "Forecast is constant. Bad forecast."
        return True, "Forecast looks good."

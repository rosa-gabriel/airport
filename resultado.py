import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from datetime import datetime
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
warnings.filterwarnings('ignore')


def load_and_prepare_data(file_path: str) -> pd.DataFrame:
    """
    Load the processed flight data and perform initial data cleaning.
    
    Args:
        file_path: Path to the processed flights CSV file
        
    Returns:
        pandas.DataFrame: Cleaned flight data
    """
    print("Loading flight data...")
    
    import re
    
    data_rows = []
    column_names = ['ICAO Empresa Aérea', 'Número Voo', 'Código Autorização (DI)', 
                   'Código Tipo Linha', 'ICAO Aeródromo Origem', 'ICAO Aeródromo Destino',
                   'Partida Prevista', 'Partida Real', 'Chegada Prevista', 
                   'Chegada Real', 'Situação Voo', 'Código Justificativa']
    
    print(f"Using column names: {column_names}")
    
    with open(file_path, 'r', encoding='utf-8') as f:

        f.readline()
        f.readline()
        
        for line in f:
            line = line.strip()
            if not line:
                continue
                
    
    
            parts = line.split(';"')
            if len(parts) >= 12:
                row = []
        
                row.append(parts[0])
        
                for i in range(1, 12):
                    if i < len(parts):
                        value = parts[i].strip('"')
                        if value == '':
                            value = None
                        row.append(value)
                    else:
                        row.append(None)
                
                data_rows.append(row)
                
        
                if len(data_rows) >= 500000:
                    break
    
    df = pd.DataFrame(data_rows, columns=column_names)
    
    print(f"Loaded {len(df):,} rows successfully")
    
    column_mapping = {
        'ICAO Empresa Aérea': 'Airline',
        'Número Voo': 'FlightNumber',
        'Código Autorização (DI)': 'AuthCode',
        'Código Tipo Linha': 'LineType',
        'ICAO Aeródromo Origem': 'OriginAirport',
        'ICAO Aeródromo Destino': 'DestinationAirport',
        'Partida Prevista': 'ScheduledDeparture',
        'Partida Real': 'ActualDeparture',
        'Chegada Prevista': 'ScheduledArrival',
        'Chegada Real': 'ActualArrival',
        'Situação Voo': 'FlightStatus',
        'Código Justificativa': 'ReasonCode'
    }
    
    df = df.rename(columns=column_mapping)
    
    df['Airline'] = df['Airline'].str.strip().str.replace('"', '')
    
    print(f"Loaded {len(df):,} flight records")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    for col in ['ScheduledDeparture', 'ActualDeparture', 'ScheduledArrival', 'ActualArrival']:
        df[col] = pd.to_datetime(df[col], format='%d/%m/%Y %H:%M', errors='coerce')
    
    valid_dates = df['ScheduledDeparture'].notna()
    df.loc[valid_dates, 'Date'] = df.loc[valid_dates, 'ScheduledDeparture'].dt.date
    df.loc[valid_dates, 'Year'] = df.loc[valid_dates, 'ScheduledDeparture'].dt.year
    df.loc[valid_dates, 'Month'] = df.loc[valid_dates, 'ScheduledDeparture'].dt.month
    df.loc[valid_dates, 'DayOfWeek'] = df.loc[valid_dates, 'ScheduledDeparture'].dt.dayofweek
    df.loc[valid_dates, 'DepartureTime'] = df.loc[valid_dates, 'ScheduledDeparture'].dt.strftime('%H:%M')
    
    df['DelayMinutes'] = (df['ActualDeparture'] - df['ScheduledDeparture']).dt.total_seconds() / 60
    df['DelayMinutes'] = df['DelayMinutes'].fillna(0)
    
    df = df[(df['FlightStatus'] != 'CANCELADO') & (df['Date'].notna())].copy()
    
    print(f"After filtering cancelled flights and invalid dates: {len(df):,} records")
    if len(df) > 0:
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"Years covered: {sorted(df['Year'].unique())}")
    else:
        print("No valid flight records found!")
    
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features for delay analysis.
    
    Args:
        df: Original flight dataframe
        
    Returns:
        pandas.DataFrame: Dataframe with engineered features
    """
    print("\nEngineering features...")
    
    df['IsDelayed'] = (df['DelayMinutes'] > 15).astype(int)
    
    def get_period_of_day(time_str):
        try:
            if pd.isna(time_str) or time_str == 'nan':
                return 'Unknown'
    
            if ':' in str(time_str):
                hour = int(str(time_str).split(':')[0])
            else:
                hour = 12
                
            if 0 <= hour <= 5:
                return 'Night'
            elif 6 <= hour <= 11:
                return 'Morning'
            elif 12 <= hour <= 17:
                return 'Afternoon'
            else:
                return 'Evening'
        except (ValueError, AttributeError):
            return 'Unknown'
    
    df['PeriodOfDay'] = df['DepartureTime'].apply(get_period_of_day)
    
    print(f"Created IsDelayed column: {df['IsDelayed'].sum():,} delayed flights ({df['IsDelayed'].mean():.1%})")
    print(f"Period of day distribution:")
    print(df['PeriodOfDay'].value_counts().sort_index())
    
    return df


def analyze_airports_most_delays(df: pd.DataFrame) -> pd.DataFrame:
    """
    Question 1: Which airport has the most delays overall?
    """
    print("\n" + "="*60)
    print("QUESTION 1: Which airport has the most delays overall?")
    print("="*60)
    
    airport_delays = df.groupby('OriginAirport').agg({
        'IsDelayed': ['sum', 'count', 'mean']
    }).round(3)
    
    airport_delays.columns = ['TotalDelays', 'TotalFlights', 'DelayRate']
    airport_delays = airport_delays.sort_values('TotalDelays', ascending=False)
    
    print("Top 10 airports with most delays:")
    print(airport_delays.head(10))
    
    return airport_delays


def analyze_airport_delay_changes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Question 2: Which airports increased/decreased delays over time?
    """
    print("\n" + "="*60)
    print("QUESTION 2: Which airports increased/decreased delays over time?")
    print("="*60)
    
    yearly_airport_delays = df.groupby(['Year', 'OriginAirport'])['IsDelayed'].sum().reset_index()
    
    years = sorted(df['Year'].unique())
    print(years)
    first_year, last_year = years[0], years[-1]
    
    first_year_data = yearly_airport_delays[yearly_airport_delays['Year'] == first_year].set_index('OriginAirport')['IsDelayed']

    last_year_data = yearly_airport_delays[yearly_airport_delays['Year'] == last_year].set_index('OriginAirport')['IsDelayed']
    
    change_df = pd.DataFrame({
        f'Delays_{first_year}': first_year_data,
        f'Delays_{last_year}': last_year_data
    }).fillna(0)
    
    change_df['Change'] = change_df[f'Delays_{last_year}'] - change_df[f'Delays_{first_year}']
    change_df['PercentChange'] = ((change_df[f'Delays_{last_year}'] - change_df[f'Delays_{first_year}']) / 
                                  change_df[f'Delays_{first_year}'].replace(0, 1) * 100).round(1)
    
    change_df = change_df.sort_values('Change', ascending=False)
    
    print(f"Airport delay changes from {first_year} to {last_year}:")
    print("Top 10 airports with biggest increases:")
    print(change_df.head(10))
    print("\nTop 10 airports with biggest decreases:")
    print(change_df.tail(10))
    
    return change_df


def analyze_yearly_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Question 3: Did delays increase or decrease over the period?
    """
    print("\n" + "="*60)
    print("QUESTION 3: Did delays increase or decrease over the period?")
    print("="*60)
    
    yearly_delays = df.groupby('Year').agg({
        'IsDelayed': ['sum', 'count', 'mean']
    }).round(3)
    
    yearly_delays.columns = ['TotalDelays', 'TotalFlights', 'DelayRate']
    
    print("Yearly delay trends:")
    print(yearly_delays)
    
    yearly_delays['DelayChange'] = yearly_delays['TotalDelays'].diff()
    yearly_delays['RateChange'] = yearly_delays['DelayRate'].diff()
    
    print("\nYear-over-year changes:")
    print(yearly_delays[['DelayChange', 'RateChange']].dropna())
    
    return yearly_delays


def analyze_delays_by_weekday(df: pd.DataFrame) -> pd.DataFrame:
    """
    Question 4: Which days of the week have the most delays (per year)?
    """
    print("\n" + "="*60)
    print("QUESTION 4: Which days of the week have the most delays (per year)?")
    print("="*60)
    
    weekday_delays = df.groupby(['Year', 'DayOfWeek'])['IsDelayed'].sum().reset_index()
    weekday_delays_pivot = weekday_delays.pivot(index='DayOfWeek', columns='Year', values='IsDelayed').fillna(0)
    
    day_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 
                 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    weekday_delays_pivot.index = weekday_delays_pivot.index.map(day_names)
    
    print("Delays by day of week (per year):")
    print(weekday_delays_pivot)
    
    return weekday_delays_pivot


def analyze_delays_by_period(df: pd.DataFrame) -> pd.DataFrame:
    """
    Question 5: Which period of the day has the most delays (per year)?
    """
    print("\n" + "="*60)
    print("QUESTION 5: Which period of the day has the most delays (per year)?")
    print("="*60)
    
    period_delays = df.groupby(['Year', 'PeriodOfDay'])['IsDelayed'].sum().reset_index()
    period_delays_pivot = period_delays.pivot(index='PeriodOfDay', columns='Year', values='IsDelayed').fillna(0)
    
    period_order = ['Morning', 'Afternoon', 'Evening', 'Night']
    period_delays_pivot = period_delays_pivot.reindex([p for p in period_order if p in period_delays_pivot.index])
    
    print("Delays by period of day (per year):")
    print(period_delays_pivot)
    
    return period_delays_pivot


def analyze_delays_by_airline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Question 6: Which airline has the most delays (per year)?
    """
    print("\n" + "="*60)
    print("QUESTION 6: Which airline has the most delays (per year)?")
    print("="*60)
    
    airline_delays = df.groupby(['Year', 'Airline'])['IsDelayed'].sum().reset_index()
    airline_delays_pivot = airline_delays.pivot(index='Airline', columns='Year', values='IsDelayed').fillna(0)
    airline_delays_pivot = airline_delays_pivot.sort_values(airline_delays_pivot.columns[-1], ascending=False)
    
    print("Top 10 airlines with most delays (per year):")
    print(airline_delays_pivot.head(10))
    
    return airline_delays_pivot


def create_visualizations(df: pd.DataFrame, airport_delays: pd.DataFrame, 
                         yearly_delays: pd.DataFrame, weekday_delays: pd.DataFrame,
                         period_delays: pd.DataFrame, airline_delays: pd.DataFrame,
                         output_dir: Path):
    """
    Create and save visualizations for all analyses.
    """
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    top_airports = airport_delays.head(15)
    plt.barh(range(len(top_airports)), top_airports['TotalDelays'])
    plt.yticks(range(len(top_airports)), top_airports.index)
    plt.xlabel('Total Delays')
    plt.title('Top 15 Airports with Most Delays')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(figures_dir / "top_airports_delays.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(yearly_delays.index, yearly_delays['TotalDelays'], marker='o', linewidth=2, markersize=8)
    plt.xlabel('Year')
    plt.ylabel('Total Delays')
    plt.title('Yearly Delay Trend')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "yearly_delay_trend.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(12, 8))
    weekday_delays.plot(kind='bar', width=0.8)
    plt.xlabel('Day of Week')
    plt.ylabel('Total Delays')
    plt.title('Delays by Day of Week (Per Year)')
    plt.xticks(rotation=45)
    plt.legend(title='Year')
    plt.tight_layout()
    plt.savefig(figures_dir / "delays_by_weekday.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    period_delays.plot(kind='bar', width=0.8)
    plt.xlabel('Period of Day')
    plt.ylabel('Total Delays')
    plt.title('Delays by Period of Day (Per Year)')
    plt.xticks(rotation=45)
    plt.legend(title='Year')
    plt.tight_layout()
    plt.savefig(figures_dir / "delays_by_period.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(12, 8))
    top_airlines = airline_delays.head(10)
    top_airlines.plot(kind='barh', width=0.8)
    plt.xlabel('Total Delays')
    plt.ylabel('Airline')
    plt.title('Top 10 Airlines with Most Delays (Per Year)')
    plt.legend(title='Year')
    plt.tight_layout()
    plt.savefig(figures_dir / "top_airlines_delays.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(yearly_delays.index, yearly_delays['DelayRate'] * 100, marker='s', linewidth=2, markersize=8, color='red')
    plt.xlabel('Year')
    plt.ylabel('Delay Rate (%)')
    plt.title('Flight Delay Rate Trend')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "delay_rate_trend.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"All visualizations saved to {figures_dir}")


def save_reports(airport_delays: pd.DataFrame, airport_changes: pd.DataFrame,
                yearly_delays: pd.DataFrame, weekday_delays: pd.DataFrame,
                period_delays: pd.DataFrame, airline_delays: pd.DataFrame,
                output_dir: Path):
    """
    Save summary tables as CSV files.
    """
    print("\nSaving summary reports...")
    
    airport_delays.to_csv(output_dir / "airport_delays_summary.csv")
    airport_changes.to_csv(output_dir / "airport_delay_changes.csv")
    yearly_delays.to_csv(output_dir / "yearly_delays_summary.csv")
    weekday_delays.to_csv(output_dir / "weekday_delays_summary.csv")
    period_delays.to_csv(output_dir / "period_delays_summary.csv")
    airline_delays.to_csv(output_dir / "airline_delays_summary.csv")
    
    print(f"All summary tables saved to {output_dir}")


def generate_insights(df: pd.DataFrame, airport_delays: pd.DataFrame,
                     airport_changes: pd.DataFrame, yearly_delays: pd.DataFrame) -> str:
    """
    Generate a summary of key insights from the analysis.
    """
    insights = []
    
    total_flights = len(df)
    total_delays = df['IsDelayed'].sum()
    delay_rate = df['IsDelayed'].mean()
    
    insights.append(f"Dataset Overview: {total_flights:,} total flights with {total_delays:,} delays ({delay_rate:.1%} delay rate)")
    
    top_airport = airport_delays.index[0]
    top_airport_delays = airport_delays.iloc[0]['TotalDelays']
    insights.append(f"Airport with most delays: {top_airport} ({top_airport_delays:,} delays)")
    
    years = sorted(yearly_delays.index)
    first_year_delays = yearly_delays.loc[years[0], 'TotalDelays']
    last_year_delays = yearly_delays.loc[years[-1], 'TotalDelays']
    
    if last_year_delays > first_year_delays:
        trend = "increased"
        change = last_year_delays - first_year_delays
    else:
        trend = "decreased"
        change = first_year_delays - last_year_delays
    
    insights.append(f"Delay trend: Delays {trend} by {change:,} from {years[0]} to {years[-1]}")
    
    biggest_increase_airport = airport_changes.index[0]
    biggest_increase = airport_changes.iloc[0]['Change']
    if biggest_increase > 0:
        insights.append(f"Biggest delay increase: {biggest_increase_airport} (+{biggest_increase:,} delays)")
    
    return " | ".join(insights)


def main():
    """
    Main function to execute the flight delay analysis.
    """
    print("Brazilian Flight Delay Analysis")
    print("=" * 50)
    print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    data_file = "data/processed/flights_all.csv"
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)
    
    df = load_and_prepare_data(data_file)
    df = engineer_features(df)
    
    airport_delays = analyze_airports_most_delays(df)
    airport_changes = analyze_airport_delay_changes(df)
    yearly_delays = analyze_yearly_trend(df)
    weekday_delays = analyze_delays_by_weekday(df)
    period_delays = analyze_delays_by_period(df)
    airline_delays = analyze_delays_by_airline(df)
    
    create_visualizations(df, airport_delays, yearly_delays, weekday_delays,
                         period_delays, airline_delays, output_dir)
    
    save_reports(airport_delays, airport_changes, yearly_delays, weekday_delays,
                period_delays, airline_delays, output_dir)
    
    print("\n" + "="*60)
    print("KEY INSIGHTS SUMMARY")
    print("="*60)
    insights = generate_insights(df, airport_delays, airport_changes, yearly_delays)
    print(insights)
    
    print(f"\nAnalysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("All results saved to reports/ directory")


if __name__ == "__main__":
    main()

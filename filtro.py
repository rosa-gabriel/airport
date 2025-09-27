import pandas as pd
from pathlib import Path
import glob
from typing import List, Optional


def process_year(year_path: Path, interim_path: Path) -> Optional[pd.DataFrame]:
    """
    Process all monthly CSV files for a given year.
    
    Args:
        year_path: Path to the year directory containing monthly CSV files
        interim_path: Path to save the processed yearly CSV file
        
    Returns:
        pandas.DataFrame: Concatenated dataframe for the year, or None if no files found
    """
    year = year_path.name
    print(f"Processing {year}...", end=" ")
    
    csv_files = sorted(glob.glob(str(year_path / "*.csv")))
    
    if not csv_files:
        print(f"No CSV files found for {year}")
        return None
    
    monthly_dataframes = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, encoding='utf-8', sep=',')
            monthly_dataframes.append(df)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            continue
    
    if not monthly_dataframes:
        print(f"No valid CSV files found for {year}")
        return None
    
    year_df = pd.concat(monthly_dataframes, ignore_index=True)
    
    date_columns = [col for col in year_df.columns if 'data' in col.lower() or 'date' in col.lower()]
    if date_columns:
        try:
    
            year_df[date_columns[0]] = pd.to_datetime(year_df[date_columns[0]], errors='coerce')
            year_df = year_df.sort_values(by=date_columns[0])
        except Exception as e:
            print(f"Warning: Could not sort by date for {year}: {e}")
    
    output_file = interim_path / f"{year}_clean.csv"
    year_df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"done ({len(year_df):,} rows)")
    return year_df


def merge_years(interim_path: Path, processed_path: Path) -> pd.DataFrame:
    """
    Merge all yearly CSV files from interim directory into final dataset.
    
    Args:
        interim_path: Path to interim directory containing yearly CSV files
        processed_path: Path to save the final processed CSV file
        
    Returns:
        pandas.DataFrame: Final merged dataframe
    """
    print("\nMerging all years...")
    
    yearly_files = sorted(glob.glob(str(interim_path / "*_clean.csv")))
    
    if not yearly_files:
        raise FileNotFoundError("No yearly CSV files found in interim directory")
    
    yearly_dataframes = []
    for csv_file in yearly_files:
        try:
            df = pd.read_csv(csv_file, encoding='utf-8', sep=',')
            yearly_dataframes.append(df)
            year = Path(csv_file).stem.replace('_clean', '')
            print(f"  - Loaded {year}: {len(df):,} rows")
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            continue
    
    final_df = pd.concat(yearly_dataframes, ignore_index=True)
    
    date_columns = [col for col in final_df.columns if 'data' in col.lower() or 'date' in col.lower()]
    if date_columns:
        try:
            final_df[date_columns[0]] = pd.to_datetime(final_df[date_columns[0]], errors='coerce')
            final_df = final_df.sort_values(by=date_columns[0])
            print(f"  - Sorted by {date_columns[0]}")
        except Exception as e:
            print(f"Warning: Could not sort final dataset by date: {e}")
    
    output_file = processed_path / "flights_all.csv"
    final_df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"Final dataset saved to {output_file}")
    return final_df


def main():
    """
    Main function to orchestrate the flight data processing pipeline.
    """
    print("Brazilian Flight Data Processing Pipeline")
    print("=" * 50)
    
    base_path = Path("data")
    raw_path = base_path / "raw"
    interim_path = base_path / "interim"
    processed_path = base_path / "processed"
    
    interim_path.mkdir(parents=True, exist_ok=True)
    processed_path.mkdir(parents=True, exist_ok=True)
    
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_path}")
    
    year_dirs = [d for d in raw_path.iterdir() if d.is_dir() and d.name.isdigit()]
    year_dirs.sort()
    
    if not year_dirs:
        raise FileNotFoundError("No year directories found in raw data path")
    
    print(f"Found {len(year_dirs)} year(s) to process: {[d.name for d in year_dirs]}")
    print()
    
    processed_years = []
    for year_dir in year_dirs:
        year_df = process_year(year_dir, interim_path)
        if year_df is not None:
            processed_years.append(year_dir.name)
    
    if not processed_years:
        print("No years were successfully processed!")
        return
    
    print(f"\nSuccessfully processed years: {processed_years}")
    
    final_df = merge_years(interim_path, processed_path)
    
    print("\n" + "=" * 50)
    print("PROCESSING COMPLETE")
    print("=" * 50)
    print(f"Final dataset shape: {final_df.shape[0]:,} rows × {final_df.shape[1]} columns")
    print(f"Years processed: {len(processed_years)}")
    print(f"Output location: {processed_path / 'flights_all.csv'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
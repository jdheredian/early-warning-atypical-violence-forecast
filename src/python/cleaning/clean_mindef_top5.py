"""
Clean crime data from Colombian Ministry of Defense (Top 5 crimes).

Author: Juan Diego Heredia Niño
Email: jd.heredian@uniandes.edu.co
Date: Oct 2025
"""
print("Started")

import pandas as pd
import yaml
from pathlib import Path
import os


def clean_crime_data(file_path: Path, output_path: Path, crime_code: str, value_column: str):
    """Clean and standardize crime data from Excel file and save to Parquet."""
    # Read Excel file
    df = pd.read_excel(file_path)
    
    # Convert date to monthly period (first day of month)
    df['FECHA_HECHO'] = df['FECHA_HECHO'].dt.to_period('M').dt.to_timestamp().dt.date
    
    # Standardize municipality code to 5-digit string
    df['COD_MUNI'] = df['COD_MUNI'].astype(str).str.zfill(5)
    
    # Group by date and municipality, summing quantities
    df = df.groupby(['FECHA_HECHO', 'COD_MUNI'])[[value_column]].sum().reset_index()
    
    # Rename columns to English standard
    df.rename(
        columns={
            'FECHA_HECHO': 'date',
            'COD_MUNI': 'mun_code',
            value_column: 'qty'
        },
        inplace=True
    )
    
    # Add crime code identifier
    df['crime_code'] = crime_code
    
    # Save to Parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

current_dir = Path(__file__).resolve().parent # Make script use its own directory as base and find paths.yml
os.chdir(current_dir) # Change working dir to script location

with open('paths.yml', 'r') as file:
    paths = yaml.safe_load(file)

raw = Path(paths['data']['raw'])
temp = Path(paths['data']['temp'])

# Crime configurations: input_file, output_file, crime_code, value_column
crimes = {
    'HOMICIDIO INTENCIONAL.xlsx': ('homicides.parquet', '01', 'VICTIMAS'),
    'SECUESTRO.xlsx': ('kidnappings.parquet', '02', 'CANTIDAD'),
    'TERRORISMO.xlsx': ('terrorism.parquet', '03', 'CANTIDAD'),
    'EXTORSIÓN.xlsx': ('extortion.parquet', '04', 'CANTIDAD'),
    'MASACRES.xlsx': ('massacres.parquet', '05', 'VICTIMAS')
}

# Process each crime type
for input_file, (output_file, crime_code, value_column) in crimes.items():
    clean_crime_data(
        raw / 'mindef' / 'top5' / input_file,
        temp / 'mindef' / 'top5' / output_file,
        crime_code=crime_code,
        value_column=value_column
    )

print("Finished")
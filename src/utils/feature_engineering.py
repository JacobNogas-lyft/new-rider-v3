import pandas as pd
import numpy as np
import re

def extract_airline_from_destination(df, destination_column='destination_place_name'):
    """
    Extract airline information from destination place name using pattern matching.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        destination_column (str): Name of the column containing destination place names
        
    Returns:
        pandas.Series: Series containing extracted airline names
    """
    
    def extract_airline(place_name):
        if pd.isna(place_name):
            return 'airline_not_in_dropoff_name'
        
        place_name_lower = str(place_name).lower()
        
        # Define airline patterns
        airline_patterns = {
            'Delta': ['delta'],
            'American': ['american'],
            'United': ['united'],
            'Alaska': ['alaska'],
            'Jet Blue': ['jetblue', 'jet blue'],
            'Air Canada': ['air canada'],
            'Allegiant Air': ['allegiant air'],
            'Hawaiian': ['hawaiian'],
            'British': ['british'],
            'Southwest': ['southwest'],
            'Spirit': ['spirit']
        }
        
        # Check each airline pattern
        for airline, patterns in airline_patterns.items():
            for pattern in patterns:
                if pattern in place_name_lower:
                    return airline
        
        return 'airline_not_in_dropoff_name'
    
    # Apply the extraction function
    airline_series = df[destination_column].apply(extract_airline)
    
    return airline_series

def add_airline_feature(df, destination_column='destination_place_name', new_column_name='airline'):
    """
    Add airline feature to the dataframe based on destination place name.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        destination_column (str): Name of the column containing destination place names
        new_column_name (str): Name for the new airline column
        
    Returns:
        pandas.DataFrame: Dataframe with the new airline column added
    """
    
    # Check if the destination column exists
    if destination_column not in df.columns:
        print(f"Warning: Column '{destination_column}' not found in dataframe.")
        print(f"Available columns containing 'destination': {[col for col in df.columns if 'destination' in col.lower()]}")
        print(f"Available columns containing 'place': {[col for col in df.columns if 'place' in col.lower()]}")
        return df
    
    # Extract airline information
    df[new_column_name] = extract_airline_from_destination(df, destination_column)
    
    # Print summary of extracted airlines
    airline_counts = df[new_column_name].value_counts()
    print(f"Airline extraction summary:")
    print(f"Total rows: {len(df)}")
    print(f"Airline distribution:")
    for airline, count in airline_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {airline}: {count:,} ({percentage:.1f}%)")
    
    return df

def add_airline_feature_from_available_columns(df):
    """
    Try to add airline feature using available destination-related columns.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        
    Returns:
        pandas.DataFrame: Dataframe with airline column added if possible
    """
    
    # List of possible destination columns to check
    possible_columns = [
        'destination_place_name',
        'destination_venue_category', 
        'place_category_destination',
        'destination_airport_code',
        'destination_category'
    ]
    
    # Find which columns exist in the dataframe
    available_columns = [col for col in possible_columns if col in df.columns]
    
    if not available_columns:
        print("No destination-related columns found. Available columns:")
        destination_cols = [col for col in df.columns if 'destination' in col.lower()]
        place_cols = [col for col in df.columns if 'place' in col.lower()]
        print(f"  Destination columns: {destination_cols}")
        print(f"  Place columns: {place_cols}")
        return df
    
    print(f"Found destination-related columns: {available_columns}")
    
    # Try to use the first available column
    best_column = available_columns[0]
    print(f"Using '{best_column}' for airline extraction")
    
    # Add the airline feature
    df = add_airline_feature(df, destination_column=best_column, new_column_name='airline')
    
    return df

def create_airline_dummy_features(df, airline_column='airline'):
    """
    Create dummy variables for airline categories.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        airline_column (str): Name of the airline column
        
    Returns:
        pandas.DataFrame: Dataframe with airline dummy columns added
    """
    
    if airline_column not in df.columns:
        print(f"Warning: Airline column '{airline_column}' not found.")
        return df
    
    # Create dummy variables
    airline_dummies = pd.get_dummies(df[airline_column], prefix='airline')
    
    # Add to dataframe
    df = pd.concat([df, airline_dummies], axis=1)
    
    print(f"Created {len(airline_dummies.columns)} airline dummy columns:")
    for col in airline_dummies.columns:
        count = airline_dummies[col].sum()
        print(f"  {col}: {count:,} occurrences")
    
    return df

# Example usage functions
def process_airline_features(df):
    """
    Complete pipeline to process airline features.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        
    Returns:
        pandas.DataFrame: Dataframe with airline features added
    """
    
    print("Processing airline features...")
    
    # Add airline feature
    df = add_airline_feature_from_available_columns(df)
    
    # Create dummy variables if airline column was successfully added
    if 'airline' in df.columns:
        df = create_airline_dummy_features(df)
    
    return df 
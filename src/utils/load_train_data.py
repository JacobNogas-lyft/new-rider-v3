import pandas as pd
import logging
import os
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_train_data():
    """
    Load the train_df.csv file from the data folder.
    
    Returns:
        pandas.DataFrame: The loaded training data
    """
    # Get the project root directory (parent of src folder)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Define the path to the train_df.csv file relative to project root
    data_path = os.path.join(project_root, 'data', 'train_df.csv')
    
    # Check if the file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data file not found at: {data_path}")
    
    try:
        # Load the CSV file
        logger.info(f"Loading training data from: {data_path}")
        df = pd.read_csv(data_path)
        
        logger.info(f"Successfully loaded training data with shape: {df.shape}")
        logger.info(f"Number of rows: {len(df)}")
        logger.info(f"Number of columns: {len(df.columns)}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading training data: {str(e)}")
        raise

def main():
    """Main function to load and display basic information about the training data."""
    try:
        # Load the training data
        df = load_train_data()
        
        # Display basic information
        print("\n" + "="*50)
        print("TRAINING DATA SUMMARY")
        print("="*50)
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("\nFirst few rows:")
        print(df.head())
        print("\nData types:")
        print(df.dtypes)
        print("\nMissing values:")
        print(df.isnull().sum())
        
        # Create categorical summary PDF
        print("\n" + "="*50)
        print("CREATING CATEGORICAL SUMMARY PDF")
        print("="*50)
        pdf_path = create_categorical_summary_pdf(df)
        print(f"PDF summary created: {pdf_path}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise

def identify_categorical_columns(df, max_unique_values=50):
    """
    Identify categorical columns in the dataframe.
    
    Args:
        df (pandas.DataFrame): The dataframe to analyze
        max_unique_values (int): Maximum number of unique values to consider a column categorical
        
    Returns:
        list: List of categorical column names
    """
    categorical_columns = []
    
    for column in df.columns:
        # Check if column is object (string) type
        if df[column].dtype == 'object':
            categorical_columns.append(column)
        # Check if column has few unique values (likely categorical)
        elif df[column].nunique() <= max_unique_values:
            categorical_columns.append(column)
    
    return categorical_columns

def create_categorical_summary_pdf(df, output_path='categorical_summary.pdf'):
    """
    Create a PDF summary of categorical columns with their unique values.
    
    Args:
        df (pandas.DataFrame): The dataframe to analyze
        output_path (str): Path where to save the PDF file
    """
    logger.info("Creating categorical summary PDF...")
    
    # Get project root for output path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(project_root, output_path)
    
    # Create the PDF document
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    story = []
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    heading_style = styles['Heading1']
    normal_style = styles['Normal']
    
    # Add title
    story.append(Paragraph("Categorical Columns Summary", title_style))
    story.append(Spacer(1, 12))
    
    # Add dataset info
    story.append(Paragraph(f"Dataset: train_df.csv", heading_style))
    story.append(Paragraph(f"Total rows: {len(df):,}", normal_style))
    story.append(Paragraph(f"Total columns: {len(df.columns)}", normal_style))
    story.append(Spacer(1, 12))
    
    # Identify categorical columns
    categorical_columns = identify_categorical_columns(df)
    
    if not categorical_columns:
        story.append(Paragraph("No categorical columns found in the dataset.", normal_style))
    else:
        story.append(Paragraph(f"Found {len(categorical_columns)} categorical columns:", heading_style))
        story.append(Spacer(1, 12))
        
        # Process each categorical column
        for i, column in enumerate(categorical_columns, 1):
            # Column header
            story.append(Paragraph(f"{i}. {column}", heading_style))
            
            # Column info
            unique_values = df[column].nunique()
            missing_values = df[column].isnull().sum()
            missing_percentage = (missing_values / len(df)) * 100
            
            info_text = f"Data type: {df[column].dtype}<br/>"
            info_text += f"Unique values: {unique_values:,}<br/>"
            info_text += f"Missing values: {missing_values:,} ({missing_percentage:.2f}%)"
            
            story.append(Paragraph(info_text, normal_style))
            story.append(Spacer(1, 6))
            
            # Show unique values
            if unique_values <= 20:
                # Show all unique values for columns with 20 or fewer unique values
                unique_vals = df[column].value_counts().sort_values(ascending=False)
                
                # Create table data
                table_data = [['Value', 'Count', 'Percentage']]
                for val, count in unique_vals.items():
                    percentage = (count / len(df)) * 100
                    # Handle NaN values
                    if pd.isna(val):
                        val_str = "NaN/Missing"
                    else:
                        val_str = str(val)
                    table_data.append([val_str, f"{count:,}", f"{percentage:.2f}%"])
                
                # Create table
                table = Table(table_data, colWidths=[3*inch, 1*inch, 1*inch])
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(table)
                
            else:
                # For columns with many unique values, show top 10 and summary
                unique_vals = df[column].value_counts().sort_values(ascending=False)
                top_10 = unique_vals.head(10)
                
                story.append(Paragraph("Top 10 most common values:", normal_style))
                
                # Create table data for top 10
                table_data = [['Value', 'Count', 'Percentage']]
                for val, count in top_10.items():
                    percentage = (count / len(df)) * 100
                    # Handle NaN values
                    if pd.isna(val):
                        val_str = "NaN/Missing"
                    else:
                        val_str = str(val)
                    table_data.append([val_str, f"{count:,}", f"{percentage:.2f}%"])
                
                # Create table
                table = Table(table_data, colWidths=[3*inch, 1*inch, 1*inch])
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(table)
                
                # Add note about remaining values
                remaining_count = unique_values - 10
                story.append(Paragraph(f"<i>... and {remaining_count:,} more unique values</i>", normal_style))
            
            story.append(Spacer(1, 12))
    
    # Build the PDF
    doc.build(story)
    logger.info(f"Categorical summary PDF saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    main() 
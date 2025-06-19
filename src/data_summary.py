import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import os
from load_data import load_parquet_data
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# Create reports directory
REPORTS_DIR = Path('reports')
REPORTS_DIR.mkdir(exist_ok=True)

def get_column_summary(df):
    """Generate summary statistics for each column"""
    summary = []
    
    for col in df.columns:
        col_info = {
            'Column Name': col,
            'Data Type': str(df[col].dtype),
            'Non-Null Count': df[col].count(),
            'Null Count': df[col].isnull().sum(),
            'Null Percentage': f"{(df[col].isnull().sum() / len(df) * 100):.2f}%",
        }
        
        # Add statistics based on data type
        if pd.api.types.is_numeric_dtype(df[col]):
            col_info.update({
                'Mean': f"{df[col].mean():.2f}" if not pd.isna(df[col].mean()) else 'N/A',
                'Std Dev': f"{df[col].std():.2f}" if not pd.isna(df[col].std()) else 'N/A',
                'Min': f"{df[col].min():.2f}" if not pd.isna(df[col].min()) else 'N/A',
                'Max': f"{df[col].max():.2f}" if not pd.isna(df[col].max()) else 'N/A',
            })
        elif pd.api.types.is_datetime64_dtype(df[col]):
            col_info.update({
                'Min Date': df[col].min(),
                'Max Date': df[col].max(),
            })
        else:
            # For categorical/string columns
            unique_vals = df[col].nunique()
            col_info.update({
                'Unique Values': unique_vals,
                'Most Common': str(df[col].mode().iloc[0]) if not df[col].mode().empty else 'N/A',
            })
        
        summary.append(col_info)
    
    return pd.DataFrame(summary)

def create_summary_pdf(df, column_summary):
    """Create a PDF report with dataset summary"""
    doc = SimpleDocTemplate(
        str(REPORTS_DIR / "dataset_summary.pdf"),
        pagesize=landscape(letter),
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12
    )
    
    # Content
    story = []
    
    # Title
    story.append(Paragraph("Dataset Summary Report", title_style))
    story.append(Spacer(1, 12))
    
    # Dataset Overview
    story.append(Paragraph("Dataset Overview", heading_style))
    overview_data = [
        ["Total Rows", str(len(df))],
        ["Total Columns", str(len(df.columns))],
        ["Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"],
        ["Date Range", f"{df['ds'].min()} to {df['ds'].max()}"],
    ]
    overview_table = Table(overview_data, colWidths=[2*inch, 4*inch])
    overview_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(overview_table)
    story.append(Spacer(1, 20))
    
    # Column Summary
    story.append(Paragraph("Column Summary", heading_style))
    
    # Only keep the 5 most important columns
    important_cols = [
        'Column Name',
        'Data Type',
        'Non-Null Count',
        'Null Percentage',
        'Unique Values'
    ]
    filtered_summary = column_summary[important_cols]
    
    # Convert column summary to list of lists for the table
    column_data = [filtered_summary.columns.tolist()] + filtered_summary.values.tolist()
    
    # Set column widths for the 5 columns
    widths = [2*inch, 1.2*inch, 1.2*inch, 1.2*inch, 1.2*inch]
    
    # Create table with column summary
    col_table = Table(column_data, colWidths=widths, repeatRows=1)
    col_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('WORDWRAP', (0, 0), (-1, -1), True),
    ]))
    story.append(col_table)
    
    # Build PDF
    doc.build(story)
    print(f"Saved dataset summary to {REPORTS_DIR / 'dataset_summary.pdf'}")

def main():
    # Load the data
    print("Loading data...")
    df = load_parquet_data()
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    
    # Generate column summary
    print("\nGenerating column summary...")
    column_summary = get_column_summary(df)
    
    # Create PDF report
    print("\nCreating summary report...")
    create_summary_pdf(df, column_summary)
    
    # Print some key statistics to console
    print("\n=== Key Statistics ===")
    print(f"\nTotal Rows: {len(df)}")
    print(f"Total Columns: {len(df.columns)}")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"Date Range: {df['ds'].min()} to {df['ds'].max()}")
    
    # Print columns with null values
    null_columns = df.columns[df.isnull().any()].tolist()
    if null_columns:
        print("\nColumns with null values:")
        for col in null_columns:
            null_count = df[col].isnull().sum()
            null_pct = (null_count / len(df) * 100)
            print(f"- {col}: {null_count} nulls ({null_pct:.2f}%)")

if __name__ == "__main__":
    main() 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# Add the src directory to the path so we can import load_data
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.load_data import load_parquet_data

def analyze_features(df):
    """
    Analyze all features in the dataframe and return a summary.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        
    Returns:
        dict: Dictionary containing feature analysis
    """
    print("Analyzing features...")
    
    feature_analysis = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'features': {}
    }
    
    # Analyze each column
    for col in df.columns:
        col_info = {
            'dtype': str(df[col].dtype),
            'null_count': df[col].isnull().sum(),
            'null_percentage': (df[col].isnull().sum() / len(df)) * 100,
            'unique_count': df[col].nunique(),
            'unique_percentage': (df[col].nunique() / len(df)) * 100
        }
        
        # Add sample values for categorical columns (first 5 unique values)
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            unique_values = df[col].dropna().unique()
            if len(unique_values) <= 10:
                col_info['sample_values'] = list(unique_values)
            else:
                col_info['sample_values'] = list(unique_values[:5]) + ['...']
        else:
            # For numeric columns, add min, max, mean
            if df[col].dtype in ['int64', 'float64']:
                col_info['min'] = df[col].min()
                col_info['max'] = df[col].max()
                col_info['mean'] = df[col].mean()
                col_info['std'] = df[col].std()
        
        feature_analysis['features'][col] = col_info
    
    return feature_analysis

def create_feature_report_pdf(feature_analysis, output_path):
    """
    Create a PDF report with feature analysis.
    
    Args:
        feature_analysis (dict): Feature analysis results
        output_path (str): Path to save the PDF
    """
    print(f"Creating PDF report at {output_path}...")
    
    # Create the PDF document
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    story = []
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=30,
        alignment=TA_CENTER
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=12,
        spaceBefore=20
    )
    normal_style = styles['Normal']
    
    # Title
    story.append(Paragraph("Feature Analysis Report - V3 Data", title_style))
    story.append(Spacer(1, 20))
    
    # Summary statistics
    story.append(Paragraph("Dataset Summary", heading_style))
    summary_data = [
        ['Metric', 'Value'],
        ['Total Rows', f"{feature_analysis['total_rows']:,}"],
        ['Total Columns', f"{feature_analysis['total_columns']:,}"],
        ['Memory Usage', f"{feature_analysis['memory_usage_mb']:.2f} MB"],
        ['Report Generated', datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
    ]
    
    summary_table = Table(summary_data, colWidths=[2*inch, 3*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 20))
    
    # Feature details
    story.append(Paragraph("Feature Details", heading_style))
    
    # Prepare feature table data
    feature_data = [['Feature Name', 'Data Type', 'Null Count', 'Null %', 'Unique Count', 'Unique %', 'Additional Info']]
    
    # Sort features by unique count (descending)
    sorted_features = sorted(feature_analysis['features'].items(), 
                           key=lambda x: x[1]['unique_count'], reverse=True)
    
    for feature_name, feature_info in sorted_features:
        # Format null percentage
        null_pct = f"{feature_info['null_percentage']:.1f}%"
        
        # Format unique percentage
        unique_pct = f"{feature_info['unique_percentage']:.1f}%"
        
        # Additional info
        additional_info = ""
        if 'sample_values' in feature_info:
            sample_str = ", ".join([str(v) for v in feature_info['sample_values']])
            additional_info = f"Sample values: {sample_str}"
        elif 'min' in feature_info and 'max' in feature_info:
            additional_info = f"Range: {feature_info['min']:.2f} to {feature_info['max']:.2f}"
        
        feature_data.append([
            feature_name,
            feature_info['dtype'],
            str(feature_info['null_count']),
            null_pct,
            str(feature_info['unique_count']),
            unique_pct,
            additional_info
        ])
    
    # Create feature table
    feature_table = Table(feature_data, colWidths=[1.5*inch, 0.8*inch, 0.7*inch, 0.6*inch, 0.7*inch, 0.6*inch, 2*inch])
    feature_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.beige, colors.white])
    ]))
    
    story.append(feature_table)
    story.append(Spacer(1, 20))
    
    # Feature categories
    story.append(Paragraph("Feature Categories", heading_style))
    
    # Categorize features
    numeric_features = []
    categorical_features = []
    datetime_features = []
    
    for feature_name, feature_info in feature_analysis['features'].items():
        dtype = feature_info['dtype']
        if 'datetime' in dtype or 'time' in dtype:
            datetime_features.append(feature_name)
        elif dtype in ['object', 'category']:
            categorical_features.append(feature_name)
        else:
            numeric_features.append(feature_name)
    
    # Create category summary
    category_data = [
        ['Category', 'Count', 'Features'],
        ['Numeric', str(len(numeric_features)), ', '.join(numeric_features[:10]) + ('...' if len(numeric_features) > 10 else '')],
        ['Categorical', str(len(categorical_features)), ', '.join(categorical_features[:10]) + ('...' if len(categorical_features) > 10 else '')],
        ['Datetime', str(len(datetime_features)), ', '.join(datetime_features[:10]) + ('...' if len(datetime_features) > 10 else '')]
    ]
    
    category_table = Table(category_data, colWidths=[1.5*inch, 0.8*inch, 4.7*inch])
    category_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'TOP')
    ]))
    
    story.append(category_table)
    
    # Build the PDF
    doc.build(story)
    print(f"PDF report saved to {output_path}")

def create_feature_csv(feature_analysis, output_path):
    """
    Create a CSV file with feature analysis.
    
    Args:
        feature_analysis (dict): Feature analysis results
        output_path (str): Path to save the CSV
    """
    print(f"Creating CSV report at {output_path}...")
    
    # Prepare data for CSV
    csv_data = []
    
    # Sort features by unique count (descending)
    sorted_features = sorted(feature_analysis['features'].items(), 
                           key=lambda x: x[1]['unique_count'], reverse=True)
    
    for feature_name, feature_info in sorted_features:
        # Additional info
        additional_info = ""
        if 'sample_values' in feature_info:
            sample_str = ", ".join([str(v) for v in feature_info['sample_values']])
            additional_info = f"Sample values: {sample_str}"
        elif 'min' in feature_info and 'max' in feature_info:
            additional_info = f"Range: {feature_info['min']:.2f} to {feature_info['max']:.2f}"
        
        csv_data.append({
            'Feature_Name': feature_name,
            'Data_Type': feature_info['dtype'],
            'Null_Count': feature_info['null_count'],
            'Null_Percentage': feature_info['null_percentage'],
            'Unique_Count': feature_info['unique_count'],
            'Unique_Percentage': feature_info['unique_percentage'],
            'Additional_Info': additional_info
        })
    
    # Create DataFrame and save to CSV
    df_csv = pd.DataFrame(csv_data)
    df_csv.to_csv(output_path, index=False)
    print(f"CSV report saved to {output_path}")

def create_feature_visualizations(df, feature_analysis, output_dir):
    """
    Create visualizations for feature analysis.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        feature_analysis (dict): Feature analysis results
        output_dir (str): Directory to save plots
    """
    print(f"Creating visualizations in {output_dir}...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette('husl')
    
    # 1. Feature types distribution
    plt.figure(figsize=(12, 8))
    dtype_counts = {}
    for feature_info in feature_analysis['features'].values():
        dtype = feature_info['dtype']
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
    
    plt.subplot(2, 2, 1)
    plt.pie(dtype_counts.values(), labels=dtype_counts.keys(), autopct='%1.1f%%')
    plt.title('Feature Data Types Distribution')
    
    # 2. Null values heatmap (top 20 features with most nulls)
    plt.subplot(2, 2, 2)
    null_percentages = [(name, info['null_percentage']) for name, info in feature_analysis['features'].items()]
    null_percentages.sort(key=lambda x: x[1], reverse=True)
    top_null_features = null_percentages[:20]
    
    feature_names = [x[0] for x in top_null_features]
    null_pcts = [x[1] for x in top_null_features]
    
    plt.barh(range(len(feature_names)), null_pcts)
    plt.yticks(range(len(feature_names)), feature_names)
    plt.xlabel('Null Percentage (%)')
    plt.title('Top 20 Features by Null Percentage')
    plt.gca().invert_yaxis()
    
    # 3. Unique values distribution
    plt.subplot(2, 2, 3)
    unique_counts = [info['unique_count'] for info in feature_analysis['features'].values()]
    plt.hist(unique_counts, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Number of Unique Values')
    plt.ylabel('Number of Features')
    plt.title('Distribution of Unique Values per Feature')
    plt.yscale('log')
    
    # 4. Memory usage by feature type
    plt.subplot(2, 2, 4)
    memory_by_type = {}
    for feature_name, feature_info in feature_analysis['features'].items():
        dtype = feature_info['dtype']
        memory = df[feature_name].memory_usage(deep=True) / 1024**2  # MB
        if dtype not in memory_by_type:
            memory_by_type[dtype] = []
        memory_by_type[dtype].append(memory)
    
    memory_means = {dtype: np.mean(memory_list) for dtype, memory_list in memory_by_type.items()}
    plt.bar(memory_means.keys(), memory_means.values())
    plt.xlabel('Data Type')
    plt.ylabel('Average Memory Usage (MB)')
    plt.title('Average Memory Usage by Data Type')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path / 'feature_analysis_plots.pdf', format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Visualizations saved to {output_path / 'feature_analysis_plots.pdf'}")

def main():
    """Main function to run the feature analysis."""
    print("Starting V3 Data Feature Analysis...")
    print("="*60)
    
    # Load v3 data
    print("Loading V3 data...")
    df = load_parquet_data('v3')
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    
    # Analyze features
    feature_analysis = analyze_features(df)
    
    # Create output directory
    output_dir = Path('/home/sagemaker-user/studio/src/new-rider-v3/reports/feature_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate PDF report
    pdf_path = output_dir / 'v3_feature_analysis_report.pdf'
    create_feature_report_pdf(feature_analysis, str(pdf_path))
    
    # Generate CSV report
    csv_path = output_dir / 'v3_feature_analysis_report.csv'
    create_feature_csv(feature_analysis, str(csv_path))
    
    # Create visualizations
    plots_dir = output_dir / 'plots'
    create_feature_visualizations(df, feature_analysis, str(plots_dir))
    
    # Print summary to console
    print("\n" + "="*60)
    print("FEATURE ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total Rows: {feature_analysis['total_rows']:,}")
    print(f"Total Columns: {feature_analysis['total_columns']:,}")
    print(f"Memory Usage: {feature_analysis['memory_usage_mb']:.2f} MB")
    
    # Count feature types
    dtype_counts = {}
    for feature_info in feature_analysis['features'].values():
        dtype = feature_info['dtype']
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
    
    print(f"\nFeature Types:")
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count}")
    
    # Top 10 features with most nulls
    null_percentages = [(name, info['null_percentage']) for name, info in feature_analysis['features'].items()]
    null_percentages.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nTop 10 Features with Most Null Values:")
    for i, (name, null_pct) in enumerate(null_percentages[:10]):
        print(f"  {i+1}. {name}: {null_pct:.1f}%")
    
    # Top 10 features with most unique values
    unique_counts = [(name, info['unique_count']) for name, info in feature_analysis['features'].items()]
    unique_counts.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nTop 10 Features with Most Unique Values:")
    for i, (name, unique_count) in enumerate(unique_counts[:10]):
        print(f"  {i+1}. {name}: {unique_count:,}")
    
    print(f"\nReports saved to:")
    print(f"  - PDF Report: {pdf_path}")
    print(f"  - CSV Report: {csv_path}")
    print(f"  - Plots: {plots_dir}")
    print("="*60)

if __name__ == "__main__":
    main() 
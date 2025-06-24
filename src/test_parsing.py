def parse_classification_report(report_path):
    """Parse a classification report and extract metrics for premium class."""
    try:
        with open(report_path, 'r') as f:
            content = f.read()
        
        print(f"Content of {report_path}:")
        print("="*50)
        print(content)
        print("="*50)
        
        # Extract metrics for "premium (not preselected)" class
        lines = content.strip().split('\n')
        for i, line in enumerate(lines):
            print(f"Line {i}: '{line}'")
            # Look for the premium class line - it has "premium" in it and contains metrics
            if 'premium' in line and any(char.isdigit() for char in line):
                print(f"Found premium line: '{line}'")
                # Parse the metrics - split by whitespace and filter out empty strings
                parts = [part for part in line.strip().split() if part]
                print(f"Parts: {parts}")
                if len(parts) >= 4:
                    try:
                        precision = float(parts[1])
                        recall = float(parts[2])
                        f1_score = float(parts[3])
                        support = int(parts[4])
                        result = {
                            'precision': precision,
                            'recall': recall,
                            'f1_score': f1_score,
                            'support': support
                        }
                        print(f"Parsed result: {result}")
                        return result
                    except (ValueError, IndexError) as e:
                        print(f"Error parsing parts: {e}")
                        continue
        print("No premium line found")
        return None
    except Exception as e:
        print(f"Error parsing {report_path}: {e}")
        return None

# Test with a single file
test_file = "/home/sagemaker-user/studio/src/new-rider-v3/reports/decision_tree/all_features/segment_airport/mode_premium/max_depth_unbounded_split2_leaf1_ccp0.0/classification_report.txt"
result = parse_classification_report(test_file)
print(f"Final result: {result}") 
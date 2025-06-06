import json
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, Any


def load_json_file(file_path: str) -> Dict:
    """Load data from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading {file_path}: {e}")
        return {}


def extract_hill_climb_results(data_dict: Dict[str, Any]) -> Dict[str, Dict]:
    """
    Extract results from hill climbing runs data.
    For each hill climb run, gather:
      - overall_score (calculated from judge_scores)
      - criteria_aggregates = mean of all judge_scores per criterion across all tasks and iterations
    
    Args:
        data_dict: Dictionary containing hill climb run data
        
    Returns:
        Dict mapping run_id to {model_name, overall_score, criteria_aggregates, num_tasks, total_scores}
    """
    hill_climb_results = {}
    
    for run_id, run_data in data_dict.items():
        model_name = run_data.get("test_model")
        if not model_name:
            continue
            
        criteria_scores = defaultdict(list)
        all_scores = []
        
        # Process creative_tasks structure
        creative_tasks = run_data.get("creative_tasks", {})
        if not creative_tasks:
            continue
            
        for task_num, task_iterations in creative_tasks.items():
            for iteration_num, iteration_data in task_iterations.items():
                if iteration_data.get("status") not in ["completed", "judged"]:
                    continue
                    
                results_by_modifier = iteration_data.get("results_by_modifier", {})
                for modifier_text, modifier_results in results_by_modifier.items():
                    judge_scores = modifier_results.get("judge_scores", {})
                    
                    for criterion, score in judge_scores.items():
                        try:
                            score_value = float(score)
                            if score_value <= 20:  # Only include scores that are <= 20
                                criteria_scores[criterion].append(score_value)
                                all_scores.append(score_value)
                        except (ValueError, TypeError):
                            pass
        
        # Calculate overall score as mean of all scores
        overall_score = np.mean(all_scores) if all_scores else None
        
        # Calculate criteria aggregates
        criteria_aggregates = {
            criterion: np.mean(scores) if scores else np.nan
            for criterion, scores in criteria_scores.items()
        }
        
        hill_climb_results[run_id] = {
            'model_name': model_name,
            'overall_score': overall_score,
            'criteria_aggregates': criteria_aggregates,
            'num_tasks': len(creative_tasks),
            'total_scores': len(all_scores),
            'run_id': run_id
        }
    
    return hill_climb_results


def create_hill_climb_dataframe(hill_climb_results: Dict[str, Dict], min_occurrences: int = 1) -> pd.DataFrame:
    """
    Create a DataFrame from hill climb results.
    
    Args:
        hill_climb_results: Output from extract_hill_climb_results
        min_occurrences: Minimum number of times a criterion must appear to be included
        
    Returns:
        DataFrame with columns: run_id, model_name, overall_score, and criteria columns
    """
    if not hill_climb_results:
        return pd.DataFrame()
    
    # Standard criteria we expect
    included_criteria = [
        "Adherence to Instructions",
        "Believable Character Actions", 
        "Nuanced Characters",
        "Consistent Voice/Tone of Writing",
        "Imagery and Descriptive Quality",
        "Elegant Prose",
        "Emotionally Engaging",
        "Emotionally Complex",
        "Coherent",
        "Meandering",
        "Weak Dialogue", 
        "Tell-Don't-Show",
        "Unsurprising or Uncreative",
        "Amateurish",
        "Purple Prose",
        "Overwrought",
        "Incongruent Ending Positivity",
        "Unearned Transformations",
        "Well-earned Lightness or Darkness",
        "Sentences Flow Naturally",
        "Overall Reader Engagement",
        "Overall Impression"
    ]
    
    # Count criterion occurrences
    criteria_counts = defaultdict(int)
    for result in hill_climb_results.values():
        for criterion in result['criteria_aggregates'].keys():
            for expected in included_criteria:
                if criterion.lower() == expected.lower():
                    criteria_counts[criterion] += 1
                    break
    
    # Filter criteria by minimum occurrences
    valid_criteria = {
        criterion for criterion, count in criteria_counts.items()
        if count >= min_occurrences
    }
    
    # Build DataFrame rows
    rows = []
    for run_id, result in hill_climb_results.items():
        row = {
            'run_id': run_id,
            'model_name': result['model_name'],
            'overall_score': result['overall_score'],
            'num_tasks': result['num_tasks'],
            'total_scores': result['total_scores']
        }
        
        # Add criteria scores
        for criterion in valid_criteria:
            row[criterion] = result['criteria_aggregates'].get(criterion, np.nan)
            
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Sort by overall score if available
    if 'overall_score' in df.columns and not df['overall_score'].isna().all():
        df = df.sort_values('overall_score', ascending=False).reset_index(drop=True)
    
    return df


def analyze_hill_climb_performance(hill_climb_file: str = "hill_climb_runs.json") -> pd.DataFrame:
    """
    Load and analyze hill climb results from JSON file.
    
    Args:
        hill_climb_file: Path to the hill climb runs JSON file
        
    Returns:
        DataFrame with hill climb analysis results
    """
    print(f"Loading hill climb data from {hill_climb_file}...")
    data = load_json_file(hill_climb_file)
    
    if not data:
        print("No data found in hill climb file.")
        return pd.DataFrame()
    
    print(f"Found {len(data)} hill climb runs.")
    
    # Extract results
    results = extract_hill_climb_results(data)
    print(f"Successfully processed {len(results)} runs.")
    
    # Create DataFrame
    df = create_hill_climb_dataframe(results)
    
    if df.empty:
        print("No valid results found.")
        return df
    
    print(f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns.")
    print(f"Criteria columns: {[col for col in df.columns if col not in ['run_id', 'model_name', 'overall_score', 'num_tasks', 'total_scores']]}")
    
    return df


def summarize_hill_climb_results(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate summary statistics for hill climb results.
    
    Args:
        df: DataFrame from analyze_hill_climb_performance
        
    Returns:
        Dictionary with summary statistics
    """
    if df.empty:
        return {}
    
    summary = {
        'total_runs': len(df),
        'unique_models': df['model_name'].nunique(),
        'models_list': df['model_name'].unique().tolist(),
        'avg_overall_score': df['overall_score'].mean() if 'overall_score' in df.columns else None,
        'score_range': {
            'min': df['overall_score'].min(),
            'max': df['overall_score'].max()
        } if 'overall_score' in df.columns else None,
        'avg_tasks_per_run': df['num_tasks'].mean() if 'num_tasks' in df.columns else None,
        'avg_scores_per_run': df['total_scores'].mean() if 'total_scores' in df.columns else None
    }
    
    # Criteria statistics
    criteria_cols = [col for col in df.columns if col not in ['run_id', 'model_name', 'overall_score', 'num_tasks', 'total_scores']]
    if criteria_cols:
        summary['criteria_stats'] = {}
        for criterion in criteria_cols:
            summary['criteria_stats'][criterion] = {
                'mean': df[criterion].mean(),
                'std': df[criterion].std(),
                'min': df[criterion].min(),
                'max': df[criterion].max(),
                'count': df[criterion].count()
            }
    
    return summary


def compare_hill_climb_vs_regular_runs(hill_climb_file: str = "hill_climb_runs.json", 
                                     regular_runs_file: str = "runs.json") -> Dict[str, pd.DataFrame]:
    """
    Compare hill climb results with regular benchmark runs.
    
    Args:
        hill_climb_file: Path to hill climb runs JSON
        regular_runs_file: Path to regular runs JSON
        
    Returns:
        Dictionary with 'hill_climb' and 'regular' DataFrames for comparison
    """
    # Analyze hill climb runs
    hc_df = analyze_hill_climb_performance(hill_climb_file)
    
    # For regular runs, we'd need to adapt the existing extract_model_results function
    # This is a placeholder - you can integrate with the existing notebook functions
    
    print(f"Hill climb runs: {len(hc_df)}")
    print(f"Models in hill climb: {hc_df['model_name'].unique() if not hc_df.empty else 'None'}")
    
    return {
        'hill_climb': hc_df,
        'regular': pd.DataFrame()  # Placeholder
    }


# Jupyter notebook friendly functions
def load_hill_climb_data_for_notebook(file_path: str = "hill_climb_runs.json"):
    """
    Load hill climb data in a format that's easy to use in Jupyter notebooks.
    Returns both the raw results dictionary and the processed DataFrame.
    """
    data = load_json_file(file_path)
    results = extract_hill_climb_results(data)
    df = create_hill_climb_dataframe(results)
    
    return {
        'raw_data': data,
        'processed_results': results, 
        'dataframe': df,
        'summary': summarize_hill_climb_results(df)
    }


if __name__ == "__main__":
    # Example usage
    df = analyze_hill_climb_performance()
    
    if not df.empty:
        print("\nDataFrame head:")
        print(df.head())
        
        print("\nSummary:")
        summary = summarize_hill_climb_results(df)
        for key, value in summary.items():
            if key != 'criteria_stats':
                print(f"{key}: {value}")
        
        print(f"\nTop 5 runs by overall score:")
        if 'overall_score' in df.columns:
            top_runs = df.nlargest(5, 'overall_score')[['run_id', 'model_name', 'overall_score']]
            print(top_runs.to_string(index=False)) 
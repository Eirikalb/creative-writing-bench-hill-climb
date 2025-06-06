import json
import numpy as np
from collections import defaultdict

def generate_hill_climb_chart_data(file_path="hill_climb_runs.json", tree_file="prompt_tree.json"):
    """Generate chart data for hill climb runs in the same format as chart_data.json"""
    
    # Load hill climb data
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Load tree structure
    tree_data = {}
    try:
        with open(tree_file, 'r', encoding='utf-8') as f:
            tree_structure = json.load(f)
            tree_data = tree_structure.get('nodes', {})
    except FileNotFoundError:
        print(f"Warning: Tree file {tree_file} not found. Tree information will not be included.")
    
    # Create mapping from run key to tree node
    run_to_node = {}
    node_info = {}
    
    for node_id, node_data in tree_data.items():
        run_key = node_data.get('benchmark_run_key')
        if run_key:
            run_to_node[run_key] = node_id
            node_info[node_id] = {
                'parent_id': node_data.get('parent_id'),
                'children': node_data.get('children', []),
                'depth': node_data.get('depth', 0),
                'system_prompt': node_data.get('system_prompt', ''),
                'node_id': node_id
            }

    # First pass: Extract all data and calculate absolute values
    all_run_data = {}
    criteria_mapping = {
        "Tell-Don't-Show": "Show-Don't-Tell",
        "Sentences Flow Naturally": "Sentence Flow", 
        "Adherence to Instructions": "Instruction Following",
        "Weak Dialogue": "Strong Dialogue",
        "Unsurprising or Uncreative": "Creativity",
        "Elegant Prose": "Elegant Prose",
        "Coherent": "Coherent",
        "Imagery and Descriptive Quality": "Descriptive Imagery",
        "Consistent Voice/Tone of Writing": "Consistent Voice & Tone",
        "Amateurish": "Avoids Amateurish Prose",
        "Meandering": "Pacing",
        "Emotionally Complex": "Emotional Depth",
        "Incongruent Ending Positivity": "Avoids Positivity Bias", 
        "Purple Prose": "Avoids Purple Prose",
        "Believable Character Actions": "Believable Characters"
    }

    for run_id, run_data in data.items():
        model_name = run_data.get('test_model')
        if not model_name:
            continue
            
        criteria_scores = defaultdict(list)
        all_scores = []
        
        creative_tasks = run_data.get('creative_tasks', {})
        if not creative_tasks:
            continue
            
        for task_num, task_iterations in creative_tasks.items():
            for iteration_num, iteration_data in task_iterations.items():
                if iteration_data.get('status') not in ['completed', 'judged']:
                    continue
                    
                results_by_modifier = iteration_data.get('results_by_modifier', {})
                for modifier_text, modifier_results in results_by_modifier.items():
                    judge_scores = modifier_results.get('judge_scores', {})
                    
                    for criterion, score in judge_scores.items():
                        try:
                            score_value = float(score)
                            if score_value <= 20:
                                criteria_scores[criterion].append(score_value)
                                all_scores.append(score_value)
                        except (ValueError, TypeError):
                            pass
        
        if not all_scores:
            continue
            
        # Calculate criteria aggregates
        criteria_aggregates = {
            criterion: np.mean(scores) if scores else 0.0
            for criterion, scores in criteria_scores.items()
        }
        
        # Create absoluteRadar data
        absolute_labels = []
        absolute_values = []
        
        for criterion, chart_label in criteria_mapping.items():
            if criterion in criteria_aggregates:
                absolute_labels.append(chart_label)
                
                # For negative criteria, invert the scores (higher score = worse, so invert for chart)
                if criterion in ["Tell-Don't-Show", "Weak Dialogue", "Unsurprising or Uncreative", 
                               "Amateurish", "Meandering", "Incongruent Ending Positivity", "Purple Prose"]:
                    # Invert: 20 - score (so lower scores become higher values)
                    value = 20 - criteria_aggregates[criterion]
                else:
                    value = criteria_aggregates[criterion]
                    
                absolute_values.append(round(value, 2))
        
        all_run_data[f"hillclimb_{run_id}"] = {
            'labels': absolute_labels,
            'values': absolute_values,
            'criteria_aggregates': criteria_aggregates
        }

    # Second pass: Calculate averages across all runs for relative values
    if not all_run_data:
        return {}
    
    # Get all unique labels and calculate averages
    all_labels = set()
    for run_data in all_run_data.values():
        all_labels.update(run_data['labels'])
    all_labels = sorted(list(all_labels))
    
    # Calculate average values for each criterion across all runs
    criterion_averages = {}
    for label in all_labels:
        values_for_criterion = []
        for run_data in all_run_data.values():
            if label in run_data['labels']:
                idx = run_data['labels'].index(label)
                values_for_criterion.append(run_data['values'][idx])
        
        if values_for_criterion:
            criterion_averages[label] = np.mean(values_for_criterion)
    
    # Third pass: Generate final chart data with relative values
    hill_climb_chart_data = {}
    
    for run_name, run_data in all_run_data.items():
        absolute_labels = run_data['labels']
        absolute_values = run_data['values']
        
        # Calculate relative values as difference from average
        relative_values = []
        for i, label in enumerate(absolute_labels):
            if label in criterion_averages:
                relative_value = absolute_values[i] - criterion_averages[label]
                relative_values.append(round(relative_value, 2))
            else:
                relative_values.append(0.0)
        
        # Calculate strengths and weaknesses (top 5 and bottom 5 relative performance)
        labeled_relative_scores = [(absolute_labels[i], relative_values[i]) for i in range(len(absolute_labels))]
        labeled_relative_scores.sort(key=lambda x: x[1], reverse=True)
        
        strengths = []
        for i, (label, relative_score) in enumerate(labeled_relative_scores[:5]):
            strengths.append({
                "criterion": label,
                "relativeScore": relative_score
            })
            
        weaknesses = []
        for i, (label, relative_score) in enumerate(labeled_relative_scores[-5:]):
            weaknesses.append({
                "criterion": label, 
                "relativeScore": relative_score
            })
        
        # Create the full chart data structure
        hill_climb_chart_data[run_name] = {
            "absoluteRadar": {
                "labels": absolute_labels,
                "values": absolute_values
            },
            "relativeRadarLog": {
                "labels": absolute_labels,
                "values": relative_values
            },
            "strengths": strengths,
            "weaknesses": weaknesses
        }
        
        # Add tree information if available
        if run_name.startswith("hillclimb_"):
            # Extract the actual run key from the run name
            # Try to find matching run key in the original data
            for original_run_id in data.keys():
                if f"hillclimb_{original_run_id}" == run_name:
                    # Try to find this run in our tree mapping
                    matching_node_id = None
                    for run_key, node_id in run_to_node.items():
                        if original_run_id in run_key or run_key in original_run_id:
                            matching_node_id = node_id
                            break
                    
                    if matching_node_id and matching_node_id in node_info:
                        info = node_info[matching_node_id]
                        tree_additions = {
                            "parent_id": info['parent_id'],
                            "child_ids": info['children'],
                            "depth": info['depth'],
                            "system_prompt": info['system_prompt'],
                            "node_id": info['node_id']
                        }
                        
                        # Add slop_index if available in tree data
                        if matching_node_id in tree_data and 'performance' in tree_data[matching_node_id]:
                            performance = tree_data[matching_node_id]['performance']
                            if 'slop_index' in performance and performance['slop_index'] is not None:
                                tree_additions["slop_index"] = performance['slop_index']
                        
                        hill_climb_chart_data[run_name].update(tree_additions)
                    break

    return hill_climb_chart_data

if __name__ == "__main__":
    chart_data = generate_hill_climb_chart_data()
    
    # Save to JSON file
    with open('hill_climb_chart_data.json', 'w') as f:
        json.dump(chart_data, f, indent=2, default=str)
    
    print(f"Chart data saved to hill_climb_chart_data.json")
    print(f"Number of hill climb runs processed: {len(chart_data)}")
    if chart_data:
        first_run = list(chart_data.keys())[0]
        print(f"Example structure for {first_run}:")
        print(f"  - absoluteRadar labels: {len(chart_data[first_run]['absoluteRadar']['labels'])}")
        print(f"  - strengths: {len(chart_data[first_run]['strengths'])}")
        print(f"  - weaknesses: {len(chart_data[first_run]['weaknesses'])}") 
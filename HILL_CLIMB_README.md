# Hill Climbing System for Creative Writing Prompts

## Overview

This system implements a tree-based hill climbing approach to optimize system prompts for creative writing tasks. It uses a judge model to evaluate performance and suggest improvements, creating a directed search through the space of possible prompts.

## Core Components

### 1. `core/prompt_tree.py` - Tree Structure Management

**PerformanceMetrics**: Stores comprehensive performance data including:
- **Dynamic Rubric Scores**: Extracted from actual criteria files (22 positive + 9 negative criteria)
- **Aggregate Scores**: Overall score (0-20), EQ-Bench creative score (0-100)
- **Technical Metrics**: Slop index, complexity, repetition analysis
- **Statistical Data**: Bootstrap confidence intervals, sample counts

**TreeNode**: Represents individual prompt variants with:
- System prompt text and evolution metadata
- Performance metrics and benchmark run links
- Tree relationship data (parent/children)

**PromptTree**: Manages tree operations including:
- Node creation and performance updates
- Multiple expansion strategies (best_leaf, breadth_first)
- Tree persistence and visualization

### 2. `core/prompt_evolution.py` - Intelligent Prompt Optimization

**PromptEvolver**: Uses judge model for analysis and suggestions:
- **Explicit Criteria Analysis**: Separates positive vs negative criteria
- **Best/Worst Performance Display**: Shows top 3 positive criteria and most problematic negative criteria
- **Trend Analysis**: Tracks score changes across evolution steps
- **Contextual Suggestions**: Provides specific improvement recommendations

**analyze_benchmark_results()**: Extracts metrics from benchmark runs:
- **Dynamic Criteria Loading**: Reads criteria from `data/creative_writing_criteria.txt` and `data/negative_criteria.txt`
- **Comprehensive Score Extraction**: Handles all 31 criteria automatically
- **Technical Metric Calculation**: Slop, complexity, repetition analysis

### 3. `hill_climb_runner.py` - Main Orchestration

**HillClimbRunner**: Coordinates the optimization process:
- Tree search with configurable strategies
- Early stopping and convergence detection
- Integration with existing benchmark infrastructure
- Progress tracking and detailed reporting

## Key Features

### Explicit Best/Worst Criteria Analysis

The system now clearly separates and highlights:

**Positive Criteria** (higher = better):
- Adherence to Instructions, Creativity, Emotional Engagement
- Coherent, Elegant Prose, Nuanced Characters, etc.

**Negative Criteria** (lower = better):  
- Meandering, Weak Dialogue, Amateurish
- Purple Prose, Overwrought, etc.

**Context Display Format**:
```
Node 1 (Root):
  Prompt: "Write compelling creative fiction..."
  Overall Score: 14.2/20 (EQ-Bench: 67.3/100)
  Best Positive Criteria: Creativity: 8.1, Elegant Prose: 7.8, Emotionally Engaging: 7.5
  Weakest Positive Criteria: Adherence to Instructions: 5.2, Coherent: 6.1
  Problematic Negative Criteria: Meandering: 6.8, Amateurish: 5.9
  Technical: Slop: 245, Complexity: 12.4, Repetition: 2.1
```

### Tree-Run Navigation and Analysis

**Bidirectional Linking**: Every tree node is linked to its benchmark run:
- `TreeNode.benchmark_run_key` contains the exact run key for the benchmark results
- `tree.get_node_by_run_key(run_key)` finds tree nodes by their benchmark run
- `tree.print_run_mapping()` shows complete tree ↔ run connections

**Analysis Tools**:
- `tree.print_tree()` shows tree structure with run keys: `[run: hillclimb_a1b2c3d4_1234567890]`
- Enhanced final results show both node IDs and benchmark run keys
- Dedicated analysis utility for deep exploration

### Intelligent Judge Model Context

The judge model receives structured context showing:
- Complete prompt evolution history with explicit performance breakdowns
- Trend analysis comparing parent vs child performance  
- Specific guidance on positive vs negative criteria
- Technical metric interpretation and goals

## Usage

### Basic Hill Climbing Run
```bash
python hill_climb_runner.py \
  --test-model "meta-llama/llama-3.1-8b-instruct" \
  --judge-model "anthropic/claude-3.5-sonnet" \
  --max-iterations 10 \
  --search-strategy best_leaf
```

### Resume from Existing Tree
```bash
python hill_climb_runner.py \
  --test-model "meta-llama/llama-3.1-8b-instruct" \
  --judge-model "anthropic/claude-3.5-sonnet" \
  --tree-file existing_prompt_tree.json \
  --max-iterations 5
```

### With Early Stopping
```bash
python hill_climb_runner.py \
  --test-model "meta-llama/llama-3.1-8b-instruct" \
  --judge-model "anthropic/claude-3.5-sonnet" \
  --early-stopping \
  --convergence-patience 3 \
  --max-iterations 20
```

## Output Files

- **`prompt_tree.json`**: Complete optimization tree with all nodes and performance data
- **`hill_climb_runs.json`**: Detailed benchmark results for each tested prompt
- **Console Output**: Real-time progress with top performing prompts and optimization paths

## Chart Data Generation and Visualization

### Generate Chart Data

After running hill climbing optimization, generate visualization-ready chart data:

```bash
python generate_hill_climb_chart_data.py
```

This creates:
- **`hill_climb_chart_data.json`**: Formatted data for radar charts and performance visualization
- Compatible with the model comparison visualization system
- Includes absolute scores, relative performance, strengths/weaknesses analysis

**Chart Data Structure**:
```json
{
  "hillclimb_run_key": {
    "absoluteRadar": {
      "labels": ["Show-Don't-Tell", "Sentence Flow", "Creativity", ...],
      "values": [15.2, 16.8, 14.5, ...]
    },
    "relativeRadarLog": {
      "labels": ["Show-Don't-Tell", "Sentence Flow", "Creativity", ...], 
      "values": [0.3, -0.1, 0.8, ...]
    },
    "strengths": [
      {"criterion": "Creativity", "relativeScore": 1.0},
      {"criterion": "Descriptive Imagery", "relativeScore": 0.7}
    ],
    "weaknesses": [
      {"criterion": "Pacing", "relativeScore": -0.9},
      {"criterion": "Coherent", "relativeScore": -0.6}
    ]
  }
}
```

### Visualize Results

**Upload to Web Visualizer**:
1. Generate the chart data file: `python generate_hill_climb_chart_data.py`
2. Open the visualization website: https://preview--radar-model-insights.lovable.app/
3. Upload your `hill_climb_chart_data.json` file to the website
4. Explore interactive radar charts showing:
   - **Absolute Performance**: Raw scores across all criteria
   - **Relative Performance**: Performance compared to average across runs
   - **Strengths/Weaknesses**: Top performing and problematic areas
   - **Evolution Tracking**: How prompts improved through iterations

**Visualization Features**:
- Interactive radar charts for each hill climbing run
- Side-by-side comparison of different prompt iterations
- Detailed breakdowns of all 15 creative writing criteria
- Export capabilities for presentations and reports

### Advanced Chart Generation

**Custom Data Sources**:
```bash
# Use specific files
python generate_hill_climb_chart_data.py \
  --runs-file custom_hill_climb_runs.json \
  --tree-file custom_prompt_tree.json

# Include tree relationship data
python generate_hill_climb_chart_data.py --include-tree-info
```

**Integration with Analysis**:
```python
from generate_hill_climb_chart_data import generate_hill_climb_chart_data

# Generate chart data programmatically
chart_data = generate_hill_climb_chart_data(
    file_path="hill_climb_runs.json",
    tree_file="prompt_tree.json"
)

# Save with custom filename
import json
with open('my_experiment_charts.json', 'w') as f:
    json.dump(chart_data, f, indent=2)
```

## Analysis and Navigation

After running the optimization, use the analysis utility to explore results:

### Basic Analysis
```bash
# Show tree summary and best evolution paths
python analyze_hill_climb_results.py

# Show only tree summary
python analyze_hill_climb_results.py --summary

# Show tree-run mapping
python analyze_hill_climb_results.py --mapping

# Inspect specific benchmark run
python analyze_hill_climb_results.py --run-key hillclimb_a1b2c3d4_1234567890
```

### Interactive Mode
```bash
python analyze_hill_climb_results.py --interactive
```

Interactive commands:
- `summary` - Tree overview and statistics
- `tree` - Full tree structure with run keys
- `best` - Best evolution paths with scores
- `mapping` - Complete tree ↔ run mapping
- `run <key>` - Detailed view of specific benchmark run
- `node <id>` - Detailed view of specific tree node

### Programmatic Access
```python
from core.prompt_tree import PromptTree

# Load tree
tree = PromptTree("prompt_tree.json")

# Navigate tree → runs
for node in tree.get_best_nodes(5):
    print(f"Node {node.node_id} → Run {node.benchmark_run_key}")

# Navigate runs → tree  
run_key = "hillclimb_a1b2c3d4_1234567890"
node = tree.get_node_by_run_key(run_key)
print(f"Run {run_key} → Node {node.node_id}")

# Show connections
tree.print_run_mapping()
```

## Integration

The system integrates seamlessly with existing infrastructure:
- Uses same creative writing prompts and judging criteria as EQ-Bench Creative
- Leverages existing API clients and benchmark infrastructure  
- Compatible with all supported models and configurations

## Criteria Files

**`data/creative_writing_criteria.txt`** (22 criteria total):
Contains both positive and negative criteria mixed together

**`data/negative_criteria.txt`** (9 criteria):
Lists specifically the negative criteria (subset of the above)

The system automatically separates these into:
- **13 Positive Criteria**: Where higher scores indicate better performance
- **9 Negative Criteria**: Where lower scores indicate better performance

## Recent Improvements

1. **Dynamic Criteria Loading**: Now reads actual criteria files instead of hardcoded metrics
2. **Explicit Best/Worst Display**: Clearly shows top positive and most problematic negative criteria
3. **Comprehensive Score Extraction**: Handles all 31 criteria with fuzzy matching for robustness
4. **Enhanced Judge Context**: Provides clear guidance on criteria types and optimization goals

## Testing

Run the test suite to verify installation:

```bash
python test_hill_climb.py
```

This tests tree operations, prompt evolution, and core functionality without requiring API keys.

## Integration with Existing Benchmark

The system seamlessly integrates with the existing creative writing benchmark:
- Uses same prompts, criteria, and judging infrastructure  
- Maintains compatibility with existing runs and analysis tools
- Simply adds system prompt optimization on top of existing evaluation

## Future Enhancements

- Multi-objective optimization (balance creativity vs. coherence)
- Ensemble prompt suggestions from multiple judge models
- Meta-learning patterns about what changes typically improve scores
- Automated stopping criteria based on convergence detection
- Parallel tree exploration (multiple branches simultaneously) 
#!/usr/bin/env python3
"""
Hill Climbing Runner for Creative Writing Prompt Optimization

This runner uses a tree-based approach to iteratively improve system prompts
based on benchmark performance feedback.
"""

import argparse
import logging
import sys
import signal
import time
from datetime import datetime

from core.prompt_tree import PromptTree, PerformanceMetrics
from core.prompt_evolution import PromptEvolver, analyze_benchmark_results
from core.benchmark import run_eq_bench_creative
from utils.logging_setup import setup_logging, get_verbosity
from utils.api import APIClient
from dotenv import load_dotenv

load_dotenv()

class HillClimbRunner:
    """Orchestrates the hill climbing optimization process"""
    
    def __init__(self, args):
        self.args = args
        self.tree = PromptTree(args.tree_file)
        
        # Set up API clients
        self.api_clients = {
            "test": APIClient(model_type="test"),
            "judge": APIClient(model_type="judge"),
            "judge_model": args.judge_model  # Store the actual model name
        }
        
        self.evolver = PromptEvolver(self.api_clients)
        
        # Track state
        self.iteration_count = 0
        self.max_iterations = args.max_iterations
        self.best_score = -float('inf')
        self.no_improvement_count = 0
        
    def run(self):
        """Main hill climbing loop"""
        logging.info(f"Starting hill climbing optimization with max {self.max_iterations} iterations")
        
        # Initialize or load tree
        if not self.tree.root_id:
            root_id = self.tree.create_root(self.args.initial_system_prompt)
            logging.info(f"Created new tree with root prompt: {self.args.initial_system_prompt}")
        else:
            logging.info(f"Resuming existing tree with {len(self.tree.nodes)} nodes")
            
        # Main optimization loop
        while self.iteration_count < self.max_iterations:
            self.iteration_count += 1
            logging.info(f"\n{'='*60}")
            logging.info(f"Hill Climbing Iteration {self.iteration_count}/{self.max_iterations}")
            logging.info(f"{'='*60}")
            
            # Select node to expand
            node_to_expand = self.tree.select_node_to_expand(strategy=self.args.selection_strategy)
            
            if not node_to_expand:
                logging.info("No suitable nodes found for expansion. Stopping.")
                break
                
            logging.info(f"Selected node {node_to_expand.node_id[:8]} for expansion")
            logging.info(f"Current prompt: {node_to_expand.system_prompt[:100]}...")
            
            # If this node doesn't have performance data, benchmark it first
            if not node_to_expand.performance or node_to_expand.performance.overall_score is None:
                logging.info("Benchmarking selected node...")
                self._benchmark_node(node_to_expand)
                
            # Generate improvement suggestions
            logging.info("Generating prompt improvements...")
            suggestions = self.evolver.suggest_improvements(
                self.tree, 
                node_to_expand, 
                num_suggestions=self.args.num_suggestions
            )
            
            if not suggestions:
                logging.warning("No suggestions generated. Continuing to next iteration.")
                continue
                
            logging.info(f"Generated {len(suggestions)} improvement suggestions")
            
            # Test each suggestion
            best_child_score = -float('inf')
            best_child_node = None
            
            for i, suggestion in enumerate(suggestions):
                logging.info(f"\nTesting suggestion {i+1}/{len(suggestions)}")
                logging.info(f"New prompt: {suggestion[:100]}...")
                
                # Create child node
                child_id = self.tree.add_child(
                    node_to_expand.node_id,
                    suggestion,
                    notes=f"Generated from iteration {self.iteration_count}, suggestion {i+1}"
                )
                
                child_node = self.tree.get_node(child_id)
                
                # Benchmark the child
                self._benchmark_node(child_node)
                
                # Track best child
                if child_node.performance and child_node.performance.overall_score:
                    child_score = child_node.performance.overall_score
                    if child_score > best_child_score:
                        best_child_score = child_score
                        best_child_node = child_node
                        
            # Check for improvement
            parent_score = node_to_expand.performance.overall_score if node_to_expand.performance else 0
            
            if best_child_score > parent_score:
                improvement = best_child_score - parent_score
                logging.info(f"\n🎉 IMPROVEMENT FOUND! Score: {parent_score:.2f} → {best_child_score:.2f} (+{improvement:.2f})")
                logging.info(f"Best prompt: {best_child_node.system_prompt}")
                self.no_improvement_count = 0
                
                if best_child_score > self.best_score:
                    self.best_score = best_child_score
                    
            else:
                self.no_improvement_count += 1
                logging.info(f"No improvement found. Count: {self.no_improvement_count}")
                
            # Early stopping if no improvement for too long
            if self.no_improvement_count >= self.args.early_stopping_patience:
                logging.info(f"No improvement for {self.args.early_stopping_patience} iterations. Stopping early.")
                break
                
            # Print tree status
            self._print_status()
            
        logging.info(f"\nHill climbing completed after {self.iteration_count} iterations")
        self._print_final_results()
        
    def _benchmark_node(self, node):
        """Run benchmark for a specific node"""
        logging.info(f"Running benchmark for node {node.node_id[:8]}...")
        
        # Create a unique run ID for this node
        run_id = f"hillclimb_{node.node_id[:8]}_{int(time.time())}"
        
        try:
            # Run benchmark with the node's system prompt
            run_key = run_eq_bench_creative(
                test_model=self.args.test_model,
                judge_model=self.args.judge_model,
                runs_file=self.args.runs_file,
                num_threads=self.args.threads,
                run_id=run_id,
                creative_prompts_file=self.args.creative_prompts_file,
                creative_criteria_file=self.args.criteria_file,
                negative_criteria_file=self.args.negative_criteria_file,
                judge_prompt_file=self.args.judge_prompt_file,
                redo_judging=False,
                save_interval=self.args.save_interval,
                iterations=3,  # Single iteration for hill climbing
                run_elo=False,  # Skip ELO as requested
                system_prompt=node.system_prompt  # Pass the system prompt
            )
            
            # Analyze results
            performance = analyze_benchmark_results(self.args.runs_file, run_key)
            
            # Update tree
            self.tree.update_performance(node.node_id, performance, run_key)
            
            if performance.overall_score:
                logging.info(f"Node {node.node_id[:8]} scored: {performance.overall_score:.2f}")
            else:
                logging.warning(f"No score obtained for node {node.node_id[:8]}")
                
        except Exception as e:
            logging.error(f"Error benchmarking node {node.node_id[:8]}: {e}")
            # Create empty performance to mark as attempted
            self.tree.update_performance(node.node_id, PerformanceMetrics())
            
    def _print_status(self):
        """Print current optimization status"""
        logging.info(f"\nCurrent Tree Status:")
        logging.info(f"- Total nodes: {len(self.tree.nodes)}")
        logging.info(f"- Best score so far: {self.best_score:.2f}")
        
        # Show top performing nodes
        best_nodes = self.tree.get_best_nodes(n=3)
        if best_nodes:
            logging.info("Top performing nodes:")
            for i, node in enumerate(best_nodes, 1):
                score = node.performance.overall_score if node.performance else "N/A"
                logging.info(f"  {i}. {node.node_id[:8]}: {score} - {node.system_prompt[:60]}...")
                
    def _print_final_results(self):
        """Print final optimization results"""
        logging.info("\n" + "="*80)
        logging.info("HILL CLIMBING OPTIMIZATION COMPLETE")
        logging.info("="*80)
        
        best_nodes = self.tree.get_best_nodes(n=5)
        
        if best_nodes:
            logging.info(f"\nTop {len(best_nodes)} performing prompts:")
            for i, node in enumerate(best_nodes, 1):
                perf = node.performance
                score = perf.overall_score if perf else "N/A"
                logging.info(f"\n{i}. Score: {score}")
                logging.info(f"   Node ID: {node.node_id}")
                logging.info(f"   Benchmark Run: {node.benchmark_run_key}")
                logging.info(f"   Prompt: {node.system_prompt}")
                if perf and perf.rubric_scores:
                    # Show top rubric scores
                    top_scores = sorted(perf.rubric_scores.items(), key=lambda x: x[1] if x[1] else 0, reverse=True)[:3]
                    scores_str = ", ".join([f"{criterion}: {score:.1f}" for criterion, score in top_scores if score is not None])
                    if scores_str:
                        logging.info(f"   Top Criteria: {scores_str}")
                if perf:
                    if perf.slop_index: logging.info(f"   Slop Index: {perf.slop_index:.0f}")
                    if perf.complexity_index: logging.info(f"   Complexity: {perf.complexity_index:.1f}")
                    
        logging.info(f"\nFiles generated:")
        logging.info(f"- Tree structure: {self.tree.tree_file}")
        logging.info(f"- Benchmark runs: {self.args.runs_file}")
        logging.info(f"- Total nodes created: {len(self.tree.nodes)}")
        logging.info(f"- Nodes with benchmark runs: {len(self.tree.get_nodes_with_runs())}")
        
        logging.info("\nTo inspect the results:")
        logging.info("- Use tree.print_tree() to visualize the optimization path")
        logging.info("- Use tree.print_run_mapping() to see tree-run connections")
        logging.info("- Check individual runs in the runs file by run key")
        
    def _print_tree_run_mapping(self):
        """Print the mapping between tree nodes and benchmark runs"""
        self.tree.print_run_mapping()


def signal_handler(signum, frame):
    print(f"\n[DEBUG] Signal {signum} caught! Stopping gracefully.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Hill Climbing Creative Writing Prompt Optimization")
    
    # Model configuration
    parser.add_argument("--test-model", required=True, help="The model to optimize prompts for")
    parser.add_argument("--judge-model", required=True, help="The model to use for evaluation and improvement suggestions")
    
    # File paths
    parser.add_argument("--runs-file", default="results/hill_climb/hill_climb_runs.json", help="File to store benchmark runs")
    parser.add_argument("--tree-file", default="results/hill_climb/prompt_tree.json", help="File to store optimization tree")
    parser.add_argument("--creative-prompts-file", default="data/creative_writing_prompts_v3.json")
    parser.add_argument("--criteria-file", default="data/creative_writing_criteria.txt")
    parser.add_argument("--negative-criteria-file", default="data/negative_criteria.txt")
    parser.add_argument("--judge-prompt-file", default="data/creative_writing_judging_prompt.txt")
    
    # Optimization parameters
    parser.add_argument("--initial-system-prompt", default="You are a creative writer. Write compelling, original stories.", 
                       help="Initial system prompt to start optimization from")
    parser.add_argument("--max-iterations", type=int, default=10, help="Maximum optimization iterations")
    parser.add_argument("--num-suggestions", type=int, default=2, help="Number of improvement suggestions per iteration")
    parser.add_argument("--selection-strategy", choices=["best_leaf", "breadth_first"], default="best_leaf",
                       help="Strategy for selecting nodes to expand")
    parser.add_argument("--early-stopping-patience", type=int, default=3, 
                       help="Stop if no improvement for this many iterations")
    
    # Benchmark parameters
    parser.add_argument("--threads", type=int, default=4, help="Number of parallel threads for benchmarking")
    parser.add_argument("--save-interval", type=int, default=2, help="How often to save benchmark progress")
    
    # Logging
    parser.add_argument("--verbosity", choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'], default="INFO")
    
    args = parser.parse_args()
    setup_logging(get_verbosity(args.verbosity))
    
    # Hook signals
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run optimization
    runner = HillClimbRunner(args)
    runner.run()


if __name__ == "__main__":
    main() 
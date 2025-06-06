#!/usr/bin/env python3
"""
Utility script to analyze hill climbing optimization results.
Provides tools to navigate between tree structure and benchmark runs.
"""

import argparse
import json
import logging
from typing import Optional
from core.prompt_tree import PromptTree, TreeNode
from utils.file_io import load_json_file

def print_tree_summary(tree: PromptTree):
    """Print a summary of the optimization tree"""
    print("=" * 60)
    print("HILL CLIMBING TREE SUMMARY")
    print("=" * 60)
    
    print(f"Total nodes: {len(tree.nodes)}")
    print(f"Nodes with runs: {len(tree.get_nodes_with_runs())}")
    print(f"Root node: {tree.root_id[:8] if tree.root_id else 'None'}")
    
    # Show depth distribution
    depths = [node.depth for node in tree.nodes.values()]
    if depths:
        print(f"Tree depth: {max(depths)}")
        print(f"Average depth: {sum(depths) / len(depths):.1f}")
    
    print()

def show_best_evolution_path(tree: PromptTree, n: int = 3):
    """Show the evolution path to the best performing nodes"""
    print("BEST EVOLUTION PATHS")
    print("-" * 40)
    
    best_nodes = tree.get_best_nodes(n=n)
    
    for i, node in enumerate(best_nodes, 1):
        print(f"\n{i}. Best Node (Score: {node.performance.overall_score:.2f}):")
        print(f"   Node ID: {node.node_id}")
        print(f"   Run Key: {node.benchmark_run_key}")
        
        # Show evolution path
        prompt_stack = tree.get_prompt_stack(node.node_id)
        performance_history = tree.get_performance_history(node.node_id)
        
        print("   Evolution Path:")
        for j, (prompt, perf) in enumerate(zip(prompt_stack, performance_history)):
            indent = "     " + "  " * j
            score_str = f"({perf.overall_score:.2f})" if perf and perf.overall_score else "(no score)"
            print(f"{indent}→ {prompt[:50]}... {score_str}")
        
        print()

def find_run_by_key(runs_file: str, run_key: str):
    """Find and display a specific benchmark run"""
    try:
        runs = load_json_file(runs_file)
        if run_key not in runs:
            print(f"Run key '{run_key}' not found in {runs_file}")
            return
        
        run_data = runs[run_key]
        print(f"BENCHMARK RUN: {run_key}")
        print("-" * 60)
        
        # Basic info
        results = run_data.get("results", {}).get("benchmark_results", {})
        print(f"Overall Score: {results.get('creative_score_0_20', 'N/A')}")
        print(f"EQ-Bench Score: {results.get('eqbench_creative_score', 'N/A')}")
        
        # Show some sample outputs
        creative_tasks = run_data.get("creative_tasks", {})
        print(f"Number of creative tasks: {len(creative_tasks)}")
        
        # Show first task as example
        if creative_tasks:
            first_task = next(iter(creative_tasks.values()))
            if first_task:
                first_prompt = next(iter(first_task.values()))
                if first_prompt and "results_by_modifier" in first_prompt:
                    first_result = next(iter(first_prompt["results_by_modifier"].values()))
                    if "model_response" in first_result:
                        response = first_result["model_response"]
                        print(f"\nSample output:")
                        print(f"  {response[:200]}...")
        
        print()
        
    except Exception as e:
        print(f"Error loading run: {e}")

def interactive_mode(tree_file: str, runs_file: str):
    """Interactive mode for exploring results"""
    tree = PromptTree(tree_file)
    
    print("Interactive Hill Climbing Results Explorer")
    print("Commands:")
    print("  summary - Show tree summary")
    print("  tree - Show full tree structure")
    print("  best - Show best evolution paths")
    print("  mapping - Show tree-run mapping")
    print("  run <key> - Show specific benchmark run")
    print("  node <id> - Show specific tree node")
    print("  quit - Exit")
    print()
    
    while True:
        try:
            cmd = input("hill-climb> ").strip().split()
            if not cmd:
                continue
                
            if cmd[0] == "quit":
                break
            elif cmd[0] == "summary":
                print_tree_summary(tree)
            elif cmd[0] == "tree":
                tree.print_tree()
            elif cmd[0] == "best":
                show_best_evolution_path(tree)
            elif cmd[0] == "mapping":
                tree.print_run_mapping()
            elif cmd[0] == "run" and len(cmd) > 1:
                find_run_by_key(runs_file, cmd[1])
            elif cmd[0] == "node" and len(cmd) > 1:
                node = tree.get_node(cmd[1])
                if node:
                    print(f"Node: {node.node_id}")
                    print(f"Prompt: {node.system_prompt}")
                    print(f"Run Key: {node.benchmark_run_key}")
                    if node.performance:
                        print(f"Score: {node.performance.overall_score}")
                    print()
                else:
                    print(f"Node '{cmd[1]}' not found")
            else:
                print("Unknown command. Type 'quit' to exit.")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Analyze hill climbing optimization results")
    parser.add_argument("--tree-file", default="prompt_tree.json", help="Tree structure file")
    parser.add_argument("--runs-file", default="hill_climb_runs.json", help="Benchmark runs file") 
    parser.add_argument("--summary", action="store_true", help="Show tree summary")
    parser.add_argument("--best-paths", action="store_true", help="Show best evolution paths")
    parser.add_argument("--mapping", action="store_true", help="Show tree-run mapping")
    parser.add_argument("--run-key", help="Show specific benchmark run")
    parser.add_argument("--interactive", action="store_true", help="Interactive exploration mode")
    
    args = parser.parse_args()
    
    # Load tree
    tree = PromptTree(args.tree_file)
    
    if args.interactive:
        interactive_mode(args.tree_file, args.runs_file)
    elif args.summary:
        print_tree_summary(tree)
    elif args.best_paths:
        show_best_evolution_path(tree)
    elif args.mapping:
        tree.print_run_mapping()
    elif args.run_key:
        find_run_by_key(args.runs_file, args.run_key)
    else:
        # Default: show summary and best paths
        print_tree_summary(tree)
        show_best_evolution_path(tree)
        print("\nUse --interactive for full exploration, --help for more options")

if __name__ == "__main__":
    main() 
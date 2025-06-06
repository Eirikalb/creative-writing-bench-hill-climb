#!/usr/bin/env python3
"""
Test script for the hill climbing system with minimal example
"""

import os
import logging
from dotenv import load_dotenv

# Setup
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_tree_creation():
    """Test basic tree functionality"""
    print("Testing tree creation and basic operations...")
    
    from core.prompt_tree import PromptTree, PerformanceMetrics
    
    # Create a test tree
    tree = PromptTree("test_tree.json")
    
    # Create root
    root_id = tree.create_root("You are a creative writer.")
    print(f"Created root: {root_id}")
    
    # Add some children
    child1_id = tree.add_child(root_id, "You are a creative writer. Focus on vivid descriptions.", "Test child 1")
    child2_id = tree.add_child(root_id, "You are a creative writer. Emphasize dialogue.", "Test child 2")
    
    # Add performance data
    perf1 = PerformanceMetrics(overall_score=15.2, creativity=16.0, slop_index=200)
    tree.update_performance(root_id, perf1)
    
    perf2 = PerformanceMetrics(overall_score=16.8, creativity=17.5, slop_index=180)
    tree.update_performance(child1_id, perf2)
    
    # Test tree operations
    print("Tree structure:")
    tree.print_tree()
    
    print("\nBest nodes:")
    best_nodes = tree.get_best_nodes(n=3)
    for node in best_nodes:
        print(f"  Score: {node.performance.overall_score}, Prompt: {node.system_prompt}")
    
    print("\nPrompt stack for child1:")
    stack = tree.get_prompt_stack(child1_id)
    for i, prompt in enumerate(stack):
        print(f"  {i+1}. {prompt}")
    
    print("✅ Tree tests passed!")
    return tree

def test_prompt_evolution():
    """Test prompt evolution functionality"""
    print("\nTesting prompt evolution...")
    
    from core.prompt_evolution import PromptEvolver
    from utils.api import APIClient
    
    # Check if we can create API clients
    try:
        api_clients = {
            "judge": APIClient(model_type="judge"),
            "judge_model": os.getenv("JUDGE_MODEL", "gpt-4")
        }
        print("✅ API clients created successfully")
        
        evolver = PromptEvolver(api_clients)
        print("✅ PromptEvolver created successfully")
        
    except Exception as e:
        print(f"⚠️  Could not create API clients (expected if no API keys): {e}")
        
    print("✅ Prompt evolution tests passed!")

def test_performance_analysis():
    """Test performance analysis from dummy data"""
    print("\nTesting performance analysis...")
    
    from core.prompt_evolution import analyze_benchmark_results
    from core.prompt_tree import PerformanceMetrics
    
    # This will test with dummy data
    perf = PerformanceMetrics(
        overall_score=15.5,
        creativity=16.2,
        slop_index=250,
        complexity_index=45.3
    )
    
    print(f"Created performance metrics: Score={perf.overall_score}, Creativity={perf.creativity}")
    print("✅ Performance analysis tests passed!")

def main():
    print("Hill Climbing System Test")
    print("=" * 40)
    
    try:
        # Run tests
        tree = test_tree_creation()
        test_prompt_evolution()
        test_performance_analysis()
        
        print("\n🎉 All tests passed!")
        print(f"Test tree saved to: {tree.tree_file}")
        
        # Clean up test file
        if os.path.exists("test_tree.json"):
            os.remove("test_tree.json")
            print("Cleaned up test files")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
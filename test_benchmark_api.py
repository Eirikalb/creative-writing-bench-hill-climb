#!/usr/bin/env python3
"""
Minimal test that replicates how the Creative Writing Benchmark calls APIs
Tests both the test model (creative generation) and judge model (evaluation) workflows
"""

import sys
import os
sys.path.append('.')  # Add current directory to path for imports

from utils.api import APIClient
from core.scoring import parse_judge_scores_creative
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_creative_generation():
    """Test the creative writing generation workflow"""
    print("\n🎨 Testing Creative Writing Generation (Test Model)")
    print("=" * 60)
    
    # Create test model API client (same as benchmark does)
    test_api = APIClient(model_type="test")
    
    # Sample prompt from the actual benchmark data
    base_prompt = """Historical Fiction: Write a scene from a story set during the height of the Roman Empire, focusing on a slice of a day in the life of a gladiator. Do not write a combat scene. Use sensory details to capture the sights, sounds, and smells of ancient Rome. Explore the gladiator's thoughts and emotions. The story should also touch on the larger political and social issues of the time period. The piece should feel like a slice of a larger story."""
    
    seed_modifier = "Include references to the gladiator's favorite Roman deity or mythological figure."
    
    # Replace <SEED> placeholder as the benchmark does
    final_prompt = base_prompt.replace("<SEED>", seed_modifier) + " First person, past tense, 1000 words."
    
    print("Prompt preview:")
    print(final_prompt[:200] + "..." if len(final_prompt) > 200 else final_prompt)
    print()
    
    try:
        # Call exactly as CreativeWritingTask.generate_creative_piece() does
        response = test_api.generate(
            model="accounts/fireworks/models/deepseek-v3",
            prompt=final_prompt,
            temperature=0.7,      # Same as benchmark
            max_tokens=4000,      # Same as benchmark  
            min_p=0.1,           # Same as benchmark
            include_seed=False    # Same as benchmark
        )
        
        print("✅ Generation successful!")
        print(f"Response length: {len(response)} characters")
        print("Response preview:")
        print("-" * 40)
        print(response[:300] + "..." if len(response) > 300 else response)
        print("-" * 40)
        
        return response
        
    except Exception as e:
        print(f"❌ Generation failed: {str(e)}")
        return None

def test_creative_judging(generated_text):
    """Test the creative writing judging workflow"""
    print("\n⚖️  Testing Creative Writing Judging (Judge Model)")
    print("=" * 60)
    
    if not generated_text:
        print("❌ Skipping judging test - no generated text available")
        return None
    
    # Create judge model API client (same as benchmark does)
    judge_api = APIClient(model_type="judge")
    
    # Load the actual judge prompt template from the benchmark
    try:
        with open("data/creative_writing_judging_prompt.txt", 'r', encoding='utf-8') as f:
            judge_prompt_template = f.read()
    except FileNotFoundError:
        print("❌ Could not find judge prompt template file")
        return None
    
    # Load criteria files as the benchmark does
    try:
        with open("data/creative_writing_criteria.txt", 'r', encoding='utf-8') as f:
            creative_writing_criteria = [line.strip() for line in f if line.strip()]
        
        with open("data/negative_criteria.txt", 'r', encoding='utf-8') as f:
            negative_criteria = [line.strip() for line in f if line.strip()]
    except FileNotFoundError as e:
        print(f"❌ Could not find criteria file: {e}")
        return None
    
    # Sample base prompt (for context in judging)
    base_prompt = "Historical Fiction: Write a scene from a story set during the height of the Roman Empire..."
    
    # Format judge prompt exactly as CreativeWritingTask.judge() does
    final_judge_prompt = judge_prompt_template.format(
        writing_prompt=base_prompt,
        test_model_response=generated_text,
        creative_writing_criteria="\n".join(["- " + c for c in creative_writing_criteria]),
        lower_is_better_criteria=", ".join(negative_criteria)
    )
    
    print("Judge prompt preview:")
    print(final_judge_prompt[:300] + "..." if len(final_judge_prompt) > 300 else final_judge_prompt)
    print()
    
    try:
        # Call exactly as CreativeWritingTask.judge() does
        judge_response = judge_api.generate(
            model="accounts/fireworks/models/deepseek-v3",
            prompt=final_judge_prompt,
            temperature=0.0,      # Same as benchmark (deterministic judging)
            max_tokens=1000,      # Same as benchmark
            include_seed=True,    # Same as benchmark
            min_p=None           # Same as benchmark (no min_p for judge)
        )
        
        print("✅ Judging successful!")
        print("Judge response:")
        print("-" * 40)
        print(judge_response)
        print("-" * 40)
        
        # Parse scores exactly as the benchmark does
        try:
            scores_dict = parse_judge_scores_creative(judge_response)
            print("\n📊 Parsed Scores:")
            for metric, score in scores_dict.items():
                print(f"  {metric}: {score}")
            
            return scores_dict
        except Exception as e:
            print(f"⚠️  Score parsing failed: {str(e)}")
            return {"raw_response": judge_response}
        
    except Exception as e:
        print(f"❌ Judging failed: {str(e)}")
        return None

def main():
    """Run the minimal benchmark API test"""
    print("🧪 Minimal Creative Writing Benchmark API Test")
    print("Replicating exact API calls used in the benchmark system")
    print("=" * 70)
    
    # Test 1: Creative Generation
    generated_text = test_creative_generation()
    
    # Test 2: Creative Judging
    if generated_text:
        scores = test_creative_judging(generated_text)
        
        print("\n" + "=" * 70)
        print("📊 FINAL SUMMARY:")
        print(f"Generation: {'✅ Success' if generated_text else '❌ Failed'}")
        print(f"Judging:    {'✅ Success' if scores else '❌ Failed'}")
        
        if generated_text and scores:
            print("\n🎉 Both API workflows working perfectly!")
            print("Ready to run the full benchmark!")
            return 0
        else:
            print("\n⚠️  Some issues detected. Check the errors above.")
            return 1
    else:
        print("\n❌ Cannot proceed to judging without successful generation.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 
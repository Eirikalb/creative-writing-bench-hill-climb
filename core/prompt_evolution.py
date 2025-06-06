import logging
from typing import List, Dict, Any
from core.prompt_tree import PromptTree, TreeNode, PerformanceMetrics

class PromptEvolver:
    """Uses the judge model to analyze performance and suggest prompt improvements"""
    
    def __init__(self, api_clients: Dict[str, Any]):
        self.api_clients = api_clients
        self.judge_api = api_clients["judge"]
    
    def format_context_for_analysis(self, tree: PromptTree, node: TreeNode) -> str:
        """Format the context for the judge model to analyze and suggest improvements"""
        
        prompt_stack = tree.get_prompt_stack(node.node_id)
        performance_history = tree.get_performance_history(node.node_id)
        
        # Load criteria files for categorization
        all_criteria = []
        negative_criteria = []
        
        try:
            import os
            if os.path.exists("data/creative_writing_criteria.txt"):
                with open("data/creative_writing_criteria.txt", 'r', encoding='utf-8') as f:
                    all_criteria = [line.strip() for line in f if line.strip()]
            
            if os.path.exists("data/negative_criteria.txt"):
                with open("data/negative_criteria.txt", 'r', encoding='utf-8') as f:
                    negative_criteria = [line.strip() for line in f if line.strip()]
        except Exception as e:
            logging.warning(f"Could not load criteria files in context formatting: {e}")
        
        # Separate positive criteria (all criteria minus negative ones)
        positive_criteria = [c for c in all_criteria if c not in negative_criteria]
        
        # Build the evolution history
        evolution_text = "PROMPT EVOLUTION HISTORY:\n\n"
        
        for i, (prompt, perf) in enumerate(zip(prompt_stack, performance_history)):
            evolution_text += f"Node {i+1} {'(Root)' if i == 0 else f'(Child of Node {i})'}:\n"
            evolution_text += f"  Prompt: \"{prompt}\"\n"
            
            if perf:
                evolution_text += f"  Overall Score: {perf.overall_score:.1f}/20"
                if perf.eqbench_creative_score:
                    evolution_text += f" (EQ-Bench: {perf.eqbench_creative_score:.1f}/100)"
                evolution_text += "\n"
                
                # Show best and worst criteria explicitly
                if perf.rubric_scores:
                    # Best performing positive criteria
                    positive_scores = []
                    for criterion in positive_criteria:
                        score = perf.get_rubric_score(criterion)
                        if score is not None:
                            positive_scores.append((criterion, score))
                    
                    if positive_scores:
                        positive_scores.sort(key=lambda x: x[1], reverse=True)
                        top_positive = positive_scores[:3]  # Top 3 positive
                        best_positive_str = ", ".join([f"{criterion}: {score:.1f}" for criterion, score in top_positive])
                        evolution_text += f"  Best Positive Criteria: {best_positive_str}\n"
                        
                        # Worst performing positive criteria (lowest scores are concerning)
                        if len(positive_scores) > 3:
                            worst_positive = positive_scores[-2:]  # Bottom 2 positive 
                            worst_positive_str = ", ".join([f"{criterion}: {score:.1f}" for criterion, score in worst_positive])
                            evolution_text += f"  Weakest Positive Criteria: {worst_positive_str}\n"
                    
                    # Worst performing negative criteria (high scores are bad)
                    negative_scores = []
                    for criterion in negative_criteria:
                        score = perf.get_rubric_score(criterion)
                        if score is not None:
                            negative_scores.append((criterion, score))
                    
                    if negative_scores:
                        negative_scores.sort(key=lambda x: x[1], reverse=True)
                        worst_negative = negative_scores[:2]  # Top 2 negative (highest = worst)
                        if worst_negative:
                            worst_negative_str = ", ".join([f"{criterion}: {score:.1f}" for criterion, score in worst_negative])
                            evolution_text += f"  Problematic Negative Criteria: {worst_negative_str}\n"
                
                # Technical metrics
                tech_metrics = []
                if perf.slop_index is not None:
                    tech_metrics.append(f"Slop: {perf.slop_index:.0f}")
                if perf.complexity_index is not None:
                    tech_metrics.append(f"Complexity: {perf.complexity_index:.1f}")
                if perf.repetition_metric is not None:
                    tech_metrics.append(f"Repetition: {perf.repetition_metric:.1f}")
                
                if tech_metrics:
                    evolution_text += f"  Technical: {', '.join(tech_metrics)}\n"
                
                # Repetitive elements
                if perf.top_bigrams:
                    evolution_text += f"  Top bigrams: {', '.join(perf.top_bigrams[:5])}\n"
                if perf.top_repetitive_words:
                    evolution_text += f"  Repetitive words: {', '.join(perf.top_repetitive_words[:5])}\n"
            else:
                evolution_text += "  Performance: Not yet measured\n"
            
            evolution_text += "\n"
        
        # Add analysis section if we have multiple data points
        if len(performance_history) > 1:
            evolution_text += "ANALYSIS:\n"
            
            # Score trends
            scores = [p.overall_score for p in performance_history if p.overall_score is not None]
            if len(scores) > 1:
                change = scores[-1] - scores[0]
                evolution_text += f"- Overall score trend: {scores[0]:.1f} → {scores[-1]:.1f} ({change:+.1f})\n"
            
            # Specific metric changes for key criteria
            if len(performance_history) >= 2:
                prev_perf = performance_history[-2]
                curr_perf = performance_history[-1]
                
                # Compare key positive criteria changes
                key_positive_criteria = ["Adherence to Instructions", "Emotionally Engaging", "Coherent", "Elegant Prose", "Creativity"]
                for criterion in key_positive_criteria:
                    prev_score = prev_perf.get_rubric_score(criterion) if prev_perf else None
                    curr_score = curr_perf.get_rubric_score(criterion) if curr_perf else None
                    
                    if prev_score is not None and curr_score is not None:
                        change = curr_score - prev_score
                        evolution_text += f"- {criterion}: {prev_score:.1f} → {curr_score:.1f} ({change:+.1f})\n"
                
                # Compare key negative criteria changes (lower is better)
                key_negative_criteria = ["Meandering", "Weak Dialogue", "Amateurish"]
                for criterion in key_negative_criteria:
                    prev_score = prev_perf.get_rubric_score(criterion) if prev_perf else None
                    curr_score = curr_perf.get_rubric_score(criterion) if curr_perf else None
                    
                    if prev_score is not None and curr_score is not None:
                        change = curr_score - prev_score
                        direction = "improved" if change < 0 else "worsened" 
                        evolution_text += f"- {criterion}: {prev_score:.1f} → {curr_score:.1f} ({direction})\n"
                
                # Compare slop index
                if prev_perf.slop_index and curr_perf.slop_index:
                    change = curr_perf.slop_index - prev_perf.slop_index
                    direction = "improved" if change < 0 else "worsened"
                    evolution_text += f"- Slop index: {prev_perf.slop_index:.0f} → {curr_perf.slop_index:.0f} ({direction})\n"
            
            evolution_text += "\n"
        
        return evolution_text
    
    def suggest_improvements(self, tree: PromptTree, node: TreeNode, num_suggestions: int = 2) -> List[str]:
        """Generate improved system prompt suggestions"""
        
        context = self.format_context_for_analysis(tree, node)
        
        improvement_prompt = f"""{context}

Your job is to suggest {num_suggestions} new and improved system prompts based on the performance data above.

Understanding the Metrics:
- POSITIVE CRITERIA (higher scores are better): These measure quality aspects like creativity, coherence, emotional engagement, elegant prose, etc.
- NEGATIVE CRITERIA (lower scores are better): These measure problematic aspects like meandering, weak dialogue, clichés, etc.
- TECHNICAL METRICS: Slop index measures clichéd language (lower is better), complexity and repetition provide additional insights

Guidelines:
1. Build incrementally on what's working - don't make massive changes
2. Address specific weaknesses identified in the metrics:
   - If positive criteria scores are low, focus on enhancing those qualities
   - If negative criteria scores are high, focus on avoiding those problems
   - If slop index is high, focus on reducing clichéd language
3. Leverage strengths: build on positive criteria that are performing well
4. Keep prompts concise but specific
5. Consider the evolution trends - what changes helped or hurt?

For each suggestion, provide:
- A brief reasoning (2-3 sentences)
- The complete new system prompt

Format your response exactly as:

SUGGESTION 1:
Reasoning: [Your reasoning here]
Prompt: [Complete system prompt here]

SUGGESTION 2: 
Reasoning: [Your reasoning here]
Prompt: [Complete system prompt here]"""

        try:
            response = self.judge_api.generate(
                model=self.api_clients["judge_model"],  # Use the actual judge model name
                prompt=improvement_prompt,
                temperature=0.3,  # Slight creativity for variety
                max_tokens=1000
            )
            
            return self._parse_suggestions(response)
            
        except Exception as e:
            logging.error(f"Error generating prompt improvements: {e}")
            return []
    
    def _parse_suggestions(self, response: str) -> List[str]:
        """Parse the judge model's suggestions"""
        suggestions = []
        lines = response.split('\n')
        
        current_suggestion = None
        current_prompt = ""
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("SUGGESTION"):
                # Save previous suggestion if exists
                if current_prompt:
                    suggestions.append(current_prompt.strip())
                current_prompt = ""
                current_suggestion = line
                
            elif line.startswith("Prompt:"):
                # Extract the prompt part
                prompt_text = line[7:].strip()  # Remove "Prompt: "
                current_prompt = prompt_text
                
        # Don't forget the last suggestion
        if current_prompt:
            suggestions.append(current_prompt.strip())
            
        logging.info(f"Parsed {len(suggestions)} prompt suggestions")
        return suggestions


def analyze_benchmark_results(runs_file: str, run_key: str) -> PerformanceMetrics:
    """Extract performance metrics from benchmark results"""
    from utils.file_io import load_json_file
    from core.metrics import (
        calculate_slop_index, calculate_complexity_index,
        calculate_repetition_metric, get_top_repetitive_words,
        get_multi_prompt_ngrams
    )
    import os
    
    try:
        runs = load_json_file(runs_file)
        run_data = runs.get(run_key, {})
        
        # Load criteria files to know what rubric scores to extract
        all_criteria = []
        negative_criteria = []
        
        try:
            if os.path.exists("data/creative_writing_criteria.txt"):
                with open("data/creative_writing_criteria.txt", 'r', encoding='utf-8') as f:
                    all_criteria = [line.strip() for line in f if line.strip()]
            
            if os.path.exists("data/negative_criteria.txt"):
                with open("data/negative_criteria.txt", 'r', encoding='utf-8') as f:
                    negative_criteria = [line.strip() for line in f if line.strip()]
        except Exception as e:
            logging.warning(f"Could not load criteria files: {e}")
        
        # Separate positive criteria (all criteria minus negative ones)
        positive_criteria = [c for c in all_criteria if c not in negative_criteria]
        
        # Get benchmark results
        results = run_data.get("results", {}).get("benchmark_results", {})
        
        # Core scores
        overall_score = results.get("creative_score_0_20")
        eqbench_score = results.get("eqbench_creative_score")
        
        # Bootstrap confidence interval
        bootstrap = results.get("bootstrap_analysis", {})
        ci_lower = bootstrap.get("ci_lower")
        ci_upper = bootstrap.get("ci_upper")
        
        # Get creative tasks for detailed analysis
        creative_tasks = run_data.get("creative_tasks", {})
        
        # Collect all generated texts and judge scores
        all_texts = []
        all_judge_scores = []
        
        for iteration_str, prompts_dict in creative_tasks.items():
            for prompt_id, task_data in prompts_dict.items():
                results_by_mod = task_data.get("results_by_modifier", {})
                for seed_mod, result_block in results_by_mod.items():
                    # Collect texts
                    model_response = result_block.get("model_response", "")
                    if model_response and not result_block.get("generation_failed", False):
                        all_texts.append(model_response)
                    
                    # Collect judge scores
                    judge_scores = result_block.get("judge_scores", {})
                    if judge_scores:
                        all_judge_scores.append(judge_scores)
        
        # Calculate technical metrics
        slop_index = None
        complexity_index = None
        repetition_metric = None
        top_bigrams = []
        top_repetitive_words = []
        
        if all_texts:
            # Slop and complexity (average across all texts)
            slop_scores = [calculate_slop_index(text) for text in all_texts]
            complexity_scores = [calculate_complexity_index(text) for text in all_texts]
            
            slop_index = sum(slop_scores) / len(slop_scores) if slop_scores else None
            complexity_index = sum(complexity_scores) / len(complexity_scores) if complexity_scores else None
            
            # Repetition analysis
            texts_with_ids = [(text, f"text_{i}") for i, text in enumerate(all_texts)]
            try:
                repetition_metric = calculate_repetition_metric(texts_with_ids)
                top_repetitive_words = [word for word, score in get_top_repetitive_words(texts_with_ids, top_n=10)]
                
                # Get bigrams
                prompts_data = {"all": all_texts}  # Simplified structure
                bigrams = get_multi_prompt_ngrams(prompts_data, n=2, top_k=10)
                top_bigrams = [" ".join(bigram) for bigram, count in bigrams]
                
            except Exception as e:
                logging.warning(f"Error calculating repetition metrics: {e}")
        
        # Extract all rubric scores dynamically
        rubric_scores = {}
        
        if all_judge_scores and positive_criteria:
            def safe_average(scores, criterion_name):
                values = []
                for score_dict in scores:
                    # Try exact match first
                    if criterion_name in score_dict and isinstance(score_dict[criterion_name], (int, float)):
                        values.append(float(score_dict[criterion_name]))
                    else:
                        # Try case-insensitive and partial matches
                        for key, value in score_dict.items():
                            if isinstance(value, (int, float)):
                                # Case insensitive match
                                if key.lower() == criterion_name.lower():
                                    values.append(float(value))
                                    break
                                # Partial match (for variations like "Elegant Prose" vs "Elegant")
                                elif criterion_name.lower() in key.lower() or key.lower() in criterion_name.lower():
                                    values.append(float(value))
                                    break
                
                return sum(values) / len(values) if values else None
            
            # Extract scores for all criteria
            for criterion in positive_criteria:
                score = safe_average(all_judge_scores, criterion)
                if score is not None:
                    rubric_scores[criterion] = score
            
            # Also extract scores for negative criteria
            for criterion in negative_criteria:
                score = safe_average(all_judge_scores, criterion)
                if score is not None:
                    rubric_scores[criterion] = score
        
        return PerformanceMetrics(
            overall_score=overall_score,
            eqbench_creative_score=eqbench_score,
            slop_index=slop_index,
            complexity_index=complexity_index,
            repetition_metric=repetition_metric,
            num_samples=len(all_texts),
            bootstrap_ci_lower=ci_lower,
            bootstrap_ci_upper=ci_upper,
            top_bigrams=top_bigrams,
            top_repetitive_words=top_repetitive_words,
            rubric_scores=rubric_scores
        )
        
    except Exception as e:
        logging.error(f"Error analyzing benchmark results: {e}")
        return PerformanceMetrics() 
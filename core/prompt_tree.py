import json
import uuid
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class PerformanceMetrics:
    """Stores performance metrics for a tree node"""
    
    # Aggregate scores
    overall_score: Optional[float] = None  # 0-20 scale
    eqbench_creative_score: Optional[float] = None  # 0-100 scale
    
    # Technical metrics
    slop_index: Optional[float] = None
    complexity_index: Optional[float] = None
    repetition_metric: Optional[float] = None
    
    # Statistical info
    num_samples: int = 0
    bootstrap_ci_lower: Optional[float] = None
    bootstrap_ci_upper: Optional[float] = None
    
    # Top repeated elements
    top_bigrams: List[str] = None
    top_repetitive_words: List[str] = None
    
    # Dynamic rubric scores - will be populated from actual criteria files
    rubric_scores: Dict[str, float] = None
    
    def __post_init__(self):
        if self.top_bigrams is None:
            self.top_bigrams = []
        if self.top_repetitive_words is None:
            self.top_repetitive_words = []
        if self.rubric_scores is None:
            self.rubric_scores = {}
    
    def get_rubric_score(self, criterion: str) -> Optional[float]:
        """Get score for a specific rubric criterion"""
        return self.rubric_scores.get(criterion)
    
    def set_rubric_score(self, criterion: str, score: float):
        """Set score for a specific rubric criterion"""
        self.rubric_scores[criterion] = score
    
    def get_positive_criteria_average(self, positive_criteria: List[str]) -> Optional[float]:
        """Get average of positive criteria scores"""
        scores = [self.rubric_scores.get(criterion) for criterion in positive_criteria 
                 if criterion in self.rubric_scores and self.rubric_scores[criterion] is not None]
        return sum(scores) / len(scores) if scores else None
    
    def get_negative_criteria_average(self, negative_criteria: List[str]) -> Optional[float]:
        """Get average of negative criteria scores (these should be low)"""
        scores = [self.rubric_scores.get(criterion) for criterion in negative_criteria 
                 if criterion in self.rubric_scores and self.rubric_scores[criterion] is not None]
        return sum(scores) / len(scores) if scores else None

@dataclass
class TreeNode:
    """Represents a single node in the prompt optimization tree"""
    node_id: str
    system_prompt: str
    parent_id: Optional[str] = None
    depth: int = 0
    created_at: str = None
    
    # Performance data
    performance: Optional[PerformanceMetrics] = None
    
    # Tree structure
    children: List[str] = None  # List of child node IDs
    
    # Metadata
    notes: str = ""
    benchmark_run_key: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.children is None:
            self.children = []
        if self.performance is None:
            self.performance = PerformanceMetrics()

class PromptTree:
    """Manages the tree structure for prompt optimization"""
    
    def __init__(self, tree_file: str = "prompt_tree.json"):
        self.tree_file = tree_file
        self.nodes: Dict[str, TreeNode] = {}
        self.root_id: Optional[str] = None
        self.load_tree()
    
    def create_root(self, initial_system_prompt: str) -> str:
        """Create the root node of the tree"""
        if self.root_id:
            logging.warning("Root node already exists, creating new root will orphan existing tree")
        
        root_id = str(uuid.uuid4())
        root_node = TreeNode(
            node_id=root_id,
            system_prompt=initial_system_prompt,
            parent_id=None,
            depth=0,
            notes="Initial system prompt"
        )
        
        self.nodes[root_id] = root_node
        self.root_id = root_id
        self.save_tree()
        
        logging.info(f"Created root node {root_id} with prompt: {initial_system_prompt[:100]}...")
        return root_id
    
    def add_child(self, parent_id: str, system_prompt: str, notes: str = "") -> str:
        """Add a child node to the specified parent"""
        if parent_id not in self.nodes:
            raise ValueError(f"Parent node {parent_id} not found")
        
        parent_node = self.nodes[parent_id]
        child_id = str(uuid.uuid4())
        
        child_node = TreeNode(
            node_id=child_id,
            system_prompt=system_prompt,
            parent_id=parent_id,
            depth=parent_node.depth + 1,
            notes=notes
        )
        
        self.nodes[child_id] = child_node
        parent_node.children.append(child_id)
        
        self.save_tree()
        
        logging.info(f"Added child node {child_id} to parent {parent_id}")
        return child_id
    
    def update_performance(self, node_id: str, performance: PerformanceMetrics, benchmark_run_key: str = None):
        """Update performance metrics for a node"""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")
        
        self.nodes[node_id].performance = performance
        if benchmark_run_key:
            self.nodes[node_id].benchmark_run_key = benchmark_run_key
        
        self.save_tree()
        
        logging.info(f"Updated performance for node {node_id}: overall_score={performance.overall_score}")
    
    def get_node(self, node_id: str) -> Optional[TreeNode]:
        """Get a node by ID"""
        return self.nodes.get(node_id)
    
    def get_prompt_stack(self, node_id: str) -> List[str]:
        """Get the full prompt evolution path from root to this node"""
        if node_id not in self.nodes:
            return []
        
        stack = []
        current_id = node_id
        
        while current_id:
            node = self.nodes[current_id]
            stack.append(node.system_prompt)
            current_id = node.parent_id
        
        stack.reverse()  # Root first
        return stack
    
    def get_performance_history(self, node_id: str) -> List[PerformanceMetrics]:
        """Get performance metrics from root to this node"""
        if node_id not in self.nodes:
            return []
        
        history = []
        current_id = node_id
        
        while current_id:
            node = self.nodes[current_id]
            if node.performance and node.performance.overall_score is not None:
                history.append(node.performance)
            current_id = node.parent_id
        
        history.reverse()  # Root first
        return history
    
    def get_best_nodes(self, n: int = 5, metric: str = "overall_score") -> List[TreeNode]:
        """Get the top N nodes by performance metric"""
        nodes_with_scores = []
        
        for node in self.nodes.values():
            if node.performance and hasattr(node.performance, metric):
                score = getattr(node.performance, metric)
                if score is not None:
                    nodes_with_scores.append((score, node))
        
        # Sort by score descending
        nodes_with_scores.sort(key=lambda x: x[0], reverse=True)
        
        return [node for score, node in nodes_with_scores[:n]]
    
    def get_leaf_nodes(self) -> List[TreeNode]:
        """Get all leaf nodes (nodes with no children)"""
        return [node for node in self.nodes.values() if not node.children]
    
    def select_node_to_expand(self, strategy: str = "best_leaf") -> Optional[TreeNode]:
        """Select a node for expansion based on strategy"""
        if strategy == "best_leaf":
            # Select the best-performing leaf node
            leaf_nodes = self.get_leaf_nodes()
            if not leaf_nodes:
                return None
            
            best_node = None
            best_score = -float('inf')
            
            for node in leaf_nodes:
                if node.performance and node.performance.overall_score is not None:
                    if node.performance.overall_score > best_score:
                        best_score = node.performance.overall_score
                        best_node = node
            
            # If no scored leaf nodes found, return an unscored leaf (including root)
            if best_node is None:
                unscored_leaves = [node for node in leaf_nodes 
                                 if not node.performance or node.performance.overall_score is None]
                if unscored_leaves:
                    # Prefer root node if it's unscored, otherwise pick shallowest
                    root_node = self.get_node(self.root_id)
                    if root_node and root_node in unscored_leaves:
                        return root_node
                    else:
                        return min(unscored_leaves, key=lambda x: x.depth)
            
            return best_node
        
        elif strategy == "breadth_first":
            # Select shallowest unscored node, prioritizing root if unscored
            unscored_nodes = [node for node in self.nodes.values() 
                            if not node.performance or node.performance.overall_score is None]
            if not unscored_nodes:
                return None
            
            # If root is unscored, always return it first
            root_node = self.get_node(self.root_id)
            if root_node and root_node in unscored_nodes:
                return root_node
            
            return min(unscored_nodes, key=lambda x: x.depth)
        
        return None
    
    def save_tree(self):
        """Save tree to file"""
        tree_data = {
            "root_id": self.root_id,
            "nodes": {node_id: asdict(node) for node_id, node in self.nodes.items()}
        }
        
        with open(self.tree_file, 'w') as f:
            json.dump(tree_data, f, indent=2)
    
    def load_tree(self):
        """Load tree from file"""
        try:
            with open(self.tree_file, 'r') as f:
                tree_data = json.load(f)
            
            self.root_id = tree_data.get("root_id")
            
            # Reconstruct nodes
            for node_id, node_dict in tree_data.get("nodes", {}).items():
                # Convert performance dict back to PerformanceMetrics
                if node_dict.get("performance"):
                    node_dict["performance"] = PerformanceMetrics(**node_dict["performance"])
                
                self.nodes[node_id] = TreeNode(**node_dict)
                
            logging.info(f"Loaded tree with {len(self.nodes)} nodes")
            
        except FileNotFoundError:
            logging.info("No existing tree file found, starting fresh")
        except Exception as e:
            logging.error(f"Error loading tree: {e}")
    
    def print_tree(self, node_id: str = None, indent: int = 0):
        """Print tree structure for debugging"""
        if node_id is None:
            node_id = self.root_id
        
        if not node_id or node_id not in self.nodes:
            return
        
        node = self.nodes[node_id]
        prefix = "  " * indent
        
        score_str = ""
        if node.performance and node.performance.overall_score is not None:
            score_str = f" (score: {node.performance.overall_score:.2f})"
        
        run_str = ""
        if node.benchmark_run_key:
            run_str = f" [run: {node.benchmark_run_key}]"
        
        print(f"{prefix}{node.node_id[:8]}: {node.system_prompt[:60]}...{score_str}{run_str}")
        
        for child_id in node.children:
            self.print_tree(child_id, indent + 1)
    
    def get_node_by_run_key(self, run_key: str) -> Optional[TreeNode]:
        """Find a tree node by its benchmark run key"""
        for node in self.nodes.values():
            if node.benchmark_run_key == run_key:
                return node
        return None
    
    def get_nodes_with_runs(self) -> List[TreeNode]:
        """Get all nodes that have associated benchmark runs"""
        return [node for node in self.nodes.values() if node.benchmark_run_key is not None]
    
    def get_run_keys(self) -> List[str]:
        """Get all benchmark run keys from the tree"""
        return [node.benchmark_run_key for node in self.nodes.values() if node.benchmark_run_key is not None]
    
    def print_run_mapping(self):
        """Print mapping between tree nodes and benchmark runs"""
        print("\nTree Node → Benchmark Run Mapping:")
        print("-" * 50)
        
        for node in self.nodes.values():
            if node.benchmark_run_key:
                score_str = f" (score: {node.performance.overall_score:.2f})" if node.performance and node.performance.overall_score else ""
                print(f"Node {node.node_id[:8]}: {node.benchmark_run_key}{score_str}")
                print(f"  Prompt: {node.system_prompt[:80]}...")
                print()
        
        nodes_without_runs = [node for node in self.nodes.values() if not node.benchmark_run_key]
        if nodes_without_runs:
            print("Nodes without benchmark runs:")
            for node in nodes_without_runs:
                print(f"  {node.node_id[:8]}: {node.system_prompt[:60]}...")
            print() 
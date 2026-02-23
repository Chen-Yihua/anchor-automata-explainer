"""
Base class for automata learners.
All specific learners (DFA, Register Automata, etc.) should inherit from this class.
"""
import copy
import numpy as np
from abc import ABC, abstractmethod


class BaseAutomataLearner(ABC):
    """
    The abstract base class for automata learners.
    All specific learners (DFA, Register Automata, etc.) should inherit from this class.
    """
    
    def __init__(self):
        self.model = None
        self.training_data = []
        self.testing_data = []
        self.pos_samples = []
        self.neg_samples = []
    
    def reset(self):
        """Reset the learner state"""
        self.model = None
        self.training_data = []
        self.testing_data = []
        self.pos_samples = []
        self.neg_samples = []
    
    # ========== Abstract Methods (must be implemented by subclasses) ==========
    
    @abstractmethod
    def create_init_automata(self, data_type, positive_samples, negative_samples):
        """Create initial automata"""
        pass
    
    @abstractmethod
    def propose_automata(self, automata_list, state, sample_fcn, iteration, previous_best: list, data_type: str = 'Tabular'):
        """Propose new automata candidates"""
        pass
    
    @abstractmethod
    def check_path_accepted(self, automaton, path) -> bool:
        """Check if a path is accepted by the automaton"""
        pass
    
    @abstractmethod
    def check_path_exist(self, automaton, path) -> bool:
        """Check if a path exists in the automaton"""
        pass
    
    @abstractmethod
    def clone_automaton(self, automaton):
        """Clone an automaton"""
        pass
    
    @abstractmethod
    def serialize_automaton(self, automaton) -> int:
        """Serialize automaton to a hashable signature for deduplication"""
        pass
    
    @abstractmethod
    def automaton_to_graphviz(self, automaton) -> str:
        """Convert automaton to graphviz format for visualization"""
        pass
    
    # ========== Shared Methods ==========
    
    def convert_to_rpni_format(self, pos_samples=None, neg_samples=None):
        """General format conversion method"""
        for i in pos_samples:
            self.training_data.append([tuple(i), True])
        if neg_samples is None:
            return self.training_data
        for i in neg_samples:
            self.training_data.append([tuple(i), False])
        return self.training_data
    
    def update_state_metrics(self, state, old_automaton, new_automaton, data, labels, op_name):
        """
        Update state metrics for a proposed automaton.
        Uses the subclass's check_path_accepted and check_path_exist methods.
        """
        key_new = id(new_automaton)
        key_old = id(old_automaton)
        
        # Use subclass's path checking methods
        # t_idx = set(i for i, p in enumerate(data) if self.check_path_exist(new_automaton, p))
        accepts = np.array([self.check_path_accepted(new_automaton, p) for p in data])
        true_accept = np.sum((labels == 1) & (accepts == True))
        false_reject = np.sum((labels == 0) & (accepts == False))

        # state["t_idx"][key_new] = t_idx
        state['t_nsamples'][key_new] = float(len(data))
        state['t_accepted'][key_new] = float(np.sum(accepts))
        state["t_positives"][key_new] = float(true_accept)
        state["t_negatives"][key_new] = float(false_reject)
        state["t_order"][key_new] = copy.deepcopy(state["t_order"].get(key_old, []))
        state["t_order"][key_new].append((op_name, "updated"))


        # print(f"Proposed Automaton ID: {key_new} (from {key_old} via {op_name})")
        # print(self.automaton_to_graphviz(new_automaton))
        print("--------------------------------------------")

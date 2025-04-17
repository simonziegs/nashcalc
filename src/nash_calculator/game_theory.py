"""Module for calculating Nash equilibria and related game theory strategies for fighting game situations."""
import numpy as np
from scipy.optimize import linprog

class GameTheory:
    def __init__(self, attacker_moves, defender_moves, payoff_matrix):
        """Initialize with moves and payoff matrix.
        
        Args:
            attacker_moves (list): List of attacker move names.
            defender_moves (list): List of defender move names.
            payoff_matrix (np.ndarray): Payoff matrix (n_attacker x n_defender).
        """
        self.attacker_moves = attacker_moves
        self.defender_moves = defender_moves
        self.payoff_matrix = payoff_matrix

    def calculate_mixed_nash(self):
        """Compute the mixed-strategy Nash equilibrium using linear programming.
        
        Returns:
            tuple: (attacker_moves, defender_moves, attacker_probs, defender_probs, game_value)
        
        Raises:
            ValueError: If payoff matrix dimensions are invalid or solver fails.
        """
        n_attacker = len(self.attacker_moves)
        n_defender = len(self.defender_moves)
        if self.payoff_matrix.shape != (n_attacker, n_defender):
            raise ValueError("Payoff matrix dimensions don't match number of moves")
        
        # Attacker LP
        c_attacker = [-1] + [0] * n_attacker
        A_ub_attacker = np.hstack((np.ones((n_defender, 1)), -self.payoff_matrix.T))
        b_ub_attacker = np.zeros(n_defender)
        A_eq_attacker = np.array([[0] + [1] * n_attacker])
        b_eq_attacker = [1]
        bounds_attacker = [(None, None)] + [(0, 1)] * n_attacker
        
        try:
            res_attacker = linprog(
                c_attacker, A_ub=A_ub_attacker, b_ub=b_ub_attacker,
                A_eq=A_eq_attacker, b_eq=b_eq_attacker, bounds=bounds_attacker
            )
            if not res_attacker.success:
                raise ValueError(f"Attacker LP solver failed: {res_attacker.message}")
        except ValueError as e:
            raise ValueError(f"Attacker LP solver error: {str(e)}")
        
        # Defender LP
        c_defender = [1] + [0] * n_defender
        A_ub_defender = np.hstack((-np.ones((n_attacker, 1)), self.payoff_matrix))
        b_ub_defender = np.zeros(n_attacker)
        A_eq_defender = np.array([[0] + [1] * n_defender])
        b_eq_defender = [1]
        bounds_defender = [(None, None)] + [(0, 1)] * n_defender
        
        try:
            res_defender = linprog(
                c_defender, A_ub=A_ub_defender, b_ub=b_ub_defender,
                A_eq=A_eq_defender, b_eq=b_eq_defender, bounds=bounds_defender
            )
            if not res_defender.success:
                raise ValueError(f"Defender LP solver failed: {res_defender.message}")
        except ValueError as e:
            raise ValueError(f"Defender LP solver error: {str(e)}")
        
        attacker_probs_optimal = res_attacker.x[1:]
        defender_probs_optimal = res_defender.x[1:]
        game_value_optimal = -res_attacker.fun
        
        return (self.attacker_moves, self.defender_moves, attacker_probs_optimal,
                defender_probs_optimal, game_value_optimal)

    def compute_nash_for_subset(self, attacker_subset=None, defender_subset=None):
        """Compute Nash equilibrium for a subset of moves.
        
        Args:
            attacker_subset (list, optional): Indices of attacker moves. Defaults to all.
            defender_subset (list, optional): Indices of defender moves. Defaults to all.
        
        Returns:
            tuple: (attacker_probs, defender_probs, expected_value) or (None, None, None) if failed.
        
        Raises:
            ValueError: If subsets are invalid or calculation fails.
        """
        n_attacker, n_defender = self.payoff_matrix.shape
        if attacker_subset is None:
            attacker_subset = list(range(n_attacker))
        if defender_subset is None:
            defender_subset = list(range(n_defender))
        
        if not all(0 <= i < n_attacker for i in attacker_subset) or not all(0 <= j < n_defender for j in defender_subset):
            raise ValueError("Invalid move subset indices")
        
        sub_matrix = self.payoff_matrix[np.ix_(attacker_subset, defender_subset)]
        sub_attacker_moves = [self.attacker_moves[i] for i in attacker_subset]
        sub_defender_moves = [self.defender_moves[j] for j in defender_subset]
        
        try:
            result = GameTheory(sub_attacker_moves, sub_defender_moves, sub_matrix).calculate_mixed_nash()
            return result[2], result[3], result[4]
        except ValueError:
            return None, None, None  # Return None for failed subsets (handled by caller)
    
    def simplify_attacker_strategy(self, threshold_percent):
        """Simplify attacker's strategy while keeping EV >= (100% - threshold%) of original.
    
        Args:
            threshold_percent (float): Percentage threshold for EV reduction (0–100).
        
        Returns:
            tuple: (simplified_attacker_moves, defender_moves, attacker_probs, defender_probs, simplified_ev)
        
        Raises:
            ValueError: If threshold is invalid or calculation fails.
        """
        if not 0 <= threshold_percent <= 100:
            raise ValueError("Threshold percentage must be between 0 and 100")
        
        lower_threshold_factor = 1 - (threshold_percent / 100)
        nash_result = self.calculate_mixed_nash()
        original_ev = nash_result[4]
        lower_threshold = lower_threshold_factor * original_ev
        
        current_subset = list(range(len(self.attacker_moves)))
        
        while True:
            best_ev = -float('inf')
            best_subset = None
            for i in current_subset:
                test_subset = [x for x in current_subset if x != i]
                if len(test_subset) == 0:
                    continue
                a_probs, d_probs, ev = self.compute_nash_for_subset(test_subset, None)
                if ev is not None and ev > best_ev:
                    best_ev = ev
                    best_subset = test_subset
            if best_subset is None or best_ev < lower_threshold:
                break
            current_subset = best_subset
        
        a_probs, d_probs, simplified_ev = self.compute_nash_for_subset(current_subset, None)
        if a_probs is None:
            raise ValueError("Failed to compute simplified strategy")
        
        simplified_moves = [self.attacker_moves[i] for i in current_subset]
        return (simplified_moves, self.defender_moves, a_probs, d_probs, simplified_ev)

    def simplify_defender_strategy(self, threshold_percent):
        """Simplify defender's strategy while keeping EV <= (100% + threshold%) of original.
    
        Args:
            threshold_percent (float): Percentage threshold for EV increase (0–100).
        
        Returns:
            tuple: (attacker_moves, simplified_defender_moves, attacker_probs, defender_probs, simplified_ev)
        
        Raises:
            ValueError: If threshold is invalid or calculation fails.
        """
        if not 0 <= threshold_percent <= 100:
            raise ValueError("Threshold percentage must be between 0 and 100")
        
        upper_threshold_factor = 1 + (threshold_percent / 100)
        nash_result = self.calculate_mixed_nash()
        original_ev = nash_result[4]
        upper_threshold = upper_threshold_factor * original_ev
        
        current_subset = list(range(len(self.defender_moves)))
        
        while True:
            best_ev = float('inf')
            best_subset = None
            for j in current_subset:
                test_subset = [x for x in current_subset if x != j]
                if len(test_subset) == 0:
                    continue
                a_probs, d_probs, ev = self.compute_nash_for_subset(None, test_subset)
                if ev is not None and ev < best_ev:
                    best_ev = ev
                    best_subset = test_subset
            if best_subset is None or best_ev > upper_threshold:
                break
            current_subset = best_subset
        
        a_probs, d_probs, simplified_ev = self.compute_nash_for_subset(None, current_subset)
        if d_probs is None:
            raise ValueError("Failed to compute simplified strategy")
        
        simplified_moves = [self.defender_moves[j] for j in current_subset]
        return (self.attacker_moves, simplified_moves, a_probs, d_probs, simplified_ev)
            
    def calculate_subtle_exploit(self, attacker_probs, defender_probs, exploit_weight):
        """Compute a subtle exploit strategy for one player against fixed opponent probabilities.
        
        Args:
            attacker_probs (list): Attacker probabilities (None for unlocked).
            defender_probs (list): Defender probabilities (None for unlocked).
            exploit_weight (float): Weight for exploiting opponent's strategy (0–1).
        
        Returns:
            tuple: (attacker_moves, defender_moves, attacker_probs, defender_probs, game_value)
        
        Raises:
            ValueError: If inputs are invalid or calculation fails.
        """
        if not 0 <= exploit_weight <= 1:
            raise ValueError("Exploit weight must be between 0 and 1")
        
        n_attacker, n_defender = len(self.attacker_moves), len(self.defender_moves)
        if len(attacker_probs) != n_attacker or len(defender_probs) != n_defender:
            raise ValueError("Probability lists don't match move counts")
        
        attacker_locked = [p is not None for p in attacker_probs]
        defender_locked = [p is not None for p in defender_probs]
        
        attacker_locked_sum = sum(p for p in attacker_probs if p is not None)
        defender_locked_sum = sum(p for p in defender_probs if p is not None)
        if attacker_locked_sum > 1 or defender_locked_sum > 1:
            raise ValueError("Locked probabilities must not exceed 100%")
        
        nash_result = self.calculate_mixed_nash()
        nash_attacker_probs_opt, nash_defender_probs_opt = nash_result[2], nash_result[3]
        nash_game_value = nash_result[4]
        
        if all(defender_locked) and not any(attacker_locked):
            if not 0.99 <= defender_locked_sum <= 1.01:
                raise ValueError(f"Defender probabilities must sum to 100% (got {defender_locked_sum*100:.2f}%)")
            pure_result = self.calculate_attacker_best_response(defender_probs)
            pure_attacker_probs = pure_result[2]
            dominant_idx = np.argmax(pure_attacker_probs)
            subtle_attacker_probs = np.array(nash_attacker_probs_opt) * (1 - exploit_weight)
            subtle_attacker_probs[dominant_idx] += exploit_weight
            subtle_game_value = np.dot(subtle_attacker_probs, np.dot(self.payoff_matrix, defender_probs))
            return (self.attacker_moves, self.defender_moves, subtle_attacker_probs,
                    np.array(defender_probs), subtle_game_value)
        
        elif all(attacker_locked) and not any(defender_locked):
            if not 0.99 <= attacker_locked_sum <= 1.01:
                raise ValueError(f"Attacker probabilities must sum to 100% (got {attacker_locked_sum*100:.2f}%)")
            pure_result = self.calculate_defender_best_response(attacker_probs)
            pure_defender_probs = pure_result[3]
            dominant_idx = np.argmax(pure_defender_probs)
            subtle_defender_probs = np.array(nash_defender_probs_opt) * (1 - exploit_weight)
            subtle_defender_probs[dominant_idx] += exploit_weight
            subtle_game_value = np.dot(attacker_probs, np.dot(self.payoff_matrix, subtle_defender_probs))
            return (self.attacker_moves, self.defender_moves, np.array(attacker_probs),
                    subtle_defender_probs, subtle_game_value)
        
        else:
            raise ValueError("Lock all moves for exactly one side for subtle exploit")
            

    
    def calculate_attacker_best_response(self, defender_probs):
        """Compute attacker's best response to fixed defender probabilities.
    
        Args:
            defender_probs (list): List of defender probabilities (None for unlocked).
        
        Returns:
            tuple: (attacker_moves, defender_moves, attacker_probs, defender_probs, game_value)
        
        Raises:
            ValueError: If probabilities are invalid or solver fails.
        """
        n_attacker = len(self.attacker_moves)
        n_defender = len(self.defender_moves)
        if len(defender_probs) != n_defender:
            raise ValueError("Defender probabilities length doesn't match moves")
        
        fixed_defender_probs = np.array([p if p is not None else 0 for p in defender_probs])
        remaining_prob = 1 - sum(p for p in defender_probs if p is not None)
        unlocked_count = sum(1 for p in defender_probs if p is None)
        
        if unlocked_count > 0:
            fixed_defender_probs += (remaining_prob / unlocked_count) * np.array([1 if p is None else 0 for p in defender_probs])
        else:
            if abs(fixed_defender_probs.sum() - 1) > 1e-6:
                raise ValueError("Locked defender probabilities must sum to 100%")
            fixed_defender_probs /= fixed_defender_probs.sum()
            
        c_attacker = self.payoff_matrix.T @ fixed_defender_probs
        A_eq_attacker = np.ones((1, n_attacker))
        b_eq_attacker = [1]
        bounds_attacker = [(0, 1)] * n_attacker
        
        res_attacker = linprog(c_attacker, A_eq=A_eq_attacker, b_eq=b_eq_attacker, bounds=bounds_attacker, method='highs')
        
        if not res_attacker.success:
            raise ValueError(f"Failed to find attacker best response: {res_attacker.message}")
            
        attacker_probs = res_attacker.x
        game_value = np.dot(attacker_probs, np.dot(self.payoff_matrix, fixed_defender_probs))
        
        return self.attacker_moves, self.defender_moves, attacker_probs, fixed_defender_probs, game_value
    
    def calculate_defender_best_response(self, attacker_probs):
        """Compute defender's best response to fixed attacker probabilities.
    
        Args:
            attacker_probs (list): List of attacker probabilities (None for unlocked).
        
        Returns:
            tuple: (attacker_moves, defender_moves, attacker_probs, defender_probs, game_value)
        
        Raises:
            ValueError: If probabilities are invalid or solver fails.
        """
        n_attacker = len(self.attacker_moves)
        n_defender = len(self.defender_moves)
        if len(attacker_probs) != n_attacker:
            raise ValueError("Attacker probabilities length doesn't match moves")
        
        fixed_attacker_probs = np.array([p if p is not None else 0 for p in attacker_probs])
        remaining_prob = 1 - sum(p for p in attacker_probs if p is not None)
        unlocked_count = sum(1 for p in attacker_probs if p is None)
        
        if unlocked_count > 0:
            fixed_attacker_probs += (remaining_prob / unlocked_count) * np.array([1 if p is None else 0 for p in attacker_probs])
        else:
            if abs(fixed_attacker_probs.sum() - 1) > 1e-6:
                raise ValueError("Locked attacker probabilities must sum to 100%")
            fixed_attacker_probs /= fixed_attacker_probs.sum()
            
        c_defender = self.payoff_matrix.T @ fixed_attacker_probs
        A_eq_defender = np.ones((1, n_defender))
        b_eq_defender = [1]
        bounds_defender = [(0, 1)] * n_defender
        
        res_defender = linprog(c_defender, A_eq=A_eq_defender, b_eq=b_eq_defender, bounds=bounds_defender, method='highs')
        
        if not res_defender.success:
            raise ValueError(f"Failed to find defender best response: {res_defender.message}")
            
        defender_probs = res_defender.x
        game_value = np.dot(fixed_attacker_probs, np.dot(self.payoff_matrix, defender_probs))
        
        return self.attacker_moves, self.defender_moves, fixed_attacker_probs, defender_probs, game_value
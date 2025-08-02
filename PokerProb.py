import random
import math
from collections import Counter
from itertools import combinations

# Constants
RANKS = '23456789TJQKA'
SUITS = 'cdhs'
DECK = [r + s for r in RANKS for s in SUITS]
C = math.sqrt(2)

def flatten(state):
    return [card for group in state for card in (group if isinstance(group, list) else [group])]

# Hand Evaluation
def hand_rank(hand):
    values = sorted(['--23456789TJQKA'.index(c[0]) for c in hand], reverse=True)
    suits = [c[1] for c in hand]
    counts = Counter(values)
    is_flush = len(set(suits)) == 1
    is_straight = sorted(values) == list(range(values[0], values[0] - 5, -1))
    count_vals = sorted(counts.items(), key=lambda x: (-x[1], -x[0]))
    kind_vals = [v for v, _ in count_vals]

    if is_straight and is_flush:
        return (8, values[0])
    elif count_vals[0][1] == 4:
        return (7, *kind_vals)
    elif count_vals[0][1] == 3 and count_vals[1][1] == 2:
        return (6, *kind_vals)
    elif is_flush:
        return (5, values)
    elif is_straight:
        return (4, values[0])
    elif count_vals[0][1] == 3:
        return (3, *kind_vals)
    elif count_vals[0][1] == 2 and count_vals[1][1] == 2:
        return (2, *kind_vals)
    elif count_vals[0][1] == 2:
        return (1, *kind_vals)
    else:
        return (0, values)

def compare_hands(hand1, hand2):
    return hand_rank(hand1) > hand_rank(hand2)

# MCTS Node
class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0
        self.untried = None

    def ucb1(self, child):
        if child.visits == 0:
            return float('inf')
        return (child.wins / child.visits) + C * math.sqrt(math.log(self.visits) / child.visits)

    def best_child(self):
        return max(self.children, key=lambda c: self.ucb1(c))

    def expand(self, possible_children):
        if self.untried is None:
            self.untried = random.sample(possible_children, min(1000, len(possible_children)))
        if not self.untried:
            return None
        next_state = self.state + [self.untried.pop()]
        child = Node(next_state, parent=self)
        self.children.append(child)
        return child

    def update(self, result):
        self.visits += 1
        self.wins += result

# MCTS Logic
def mcts(your_cards, simulations=1000):
    root = Node([your_cards])

    for _ in range(simulations):
        node = root
        state = [your_cards]

        # Selection
        while node.children and node.untried == []:
            node = node.best_child()
            state = node.state

        # Expansion
        used = set(flatten(state))
        remaining = [c for c in DECK if c not in used]
        level = len(state) - 1
        options = []

        if level == 0:
            combos = list(combinations(remaining, 2))
            if combos:
                options = [list(combo) for combo in random.sample(combos, min(1000, len(combos)))]
        elif level == 1:
            combos = list(combinations(remaining, 3))
            if combos:
                options = [list(combo) for combo in random.sample(combos, min(1000, len(combos)))]
        elif level in [2, 3]:
            if remaining:
                options = [[c] for c in random.sample(remaining, min(1000, len(remaining)))]

        if not options:
            continue

        child = node.expand(options)
        if child is None:
            continue
        node = child
        state = node.state

        # Simulation
        used = set(flatten(state))
        remaining = [c for c in DECK if c not in used]
        while len(flatten(state)) < 9:
            if not remaining:
                break
            card = random.choice(remaining)
            state.append([card])
            remaining.remove(card)

        if len(flatten(state)) < 9:
            continue

        your_cards = state[0]
        opp_cards = state[1]
        board = flatten(state[2:])

        if any(card in your_cards for card in opp_cards):
            continue

        your_full = your_cards + board
        opp_full = opp_cards + board
        result = compare_hands(your_full, opp_full)

        while node is not None:
            node.update(int(result))
            node = node.parent

    return root.wins / root.visits if root.visits > 0 else 0.0

#  Preflop Winrate Table
def generate_preflop_table():
    table_values = [
        [0.51, 0.35, 0.36, 0.37, 0.37, 0.38, 0.40, 0.42, 0.44, 0.47, 0.50, 0.53, 0.57],  # 2
        [0.38, 0.55, 0.38, 0.39, 0.39, 0.40, 0.43, 0.44, 0.45, 0.48, 0.51, 0.54, 0.58],  # 3
        [0.39, 0.41, 0.58, 0.41, 0.41, 0.42, 0.45, 0.46, 0.46, 0.49, 0.52, 0.55, 0.59],  # 4
        [0.40, 0.42, 0.44, 0.61, 0.43, 0.44, 0.46, 0.47, 0.47, 0.50, 0.53, 0.56, 0.60],  # 5
        [0.40, 0.42, 0.44, 0.46, 0.64, 0.45, 0.46, 0.48, 0.49, 0.51, 0.54, 0.57, 0.61],  # 6
        [0.40, 0.42, 0.44, 0.46, 0.48, 0.67, 0.48, 0.49, 0.50, 0.52, 0.54, 0.57, 0.61],  # 7
        [0.43, 0.43, 0.45, 0.47, 0.49, 0.50, 0.70, 0.51, 0.52, 0.54, 0.56, 0.58, 0.62],  # 8
        [0.45, 0.46, 0.46, 0.48, 0.50, 0.51, 0.53, 0.73, 0.54, 0.55, 0.57, 0.59, 0.63],  # 9
        [0.47, 0.48, 0.49, 0.49, 0.51, 0.53, 0.54, 0.56, 0.76, 0.57, 0.59, 0.62, 0.65],  # T
        [0.49, 0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.57, 0.59, 0.78, 0.60, 0.62, 0.65],  # J
        [0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.61, 0.62, 0.81, 0.63, 0.66],  # Q
        [0.55, 0.56, 0.57, 0.58, 0.59, 0.60, 0.61, 0.63, 0.64, 0.65, 0.65, 0.83, 0.67],  # K
        [0.59, 0.60, 0.61, 0.62, 0.62, 0.63, 0.64, 0.64, 0.66, 0.67, 0.67, 0.68, 0.86],  # A
    ]
    ranks = list(RANKS)
    table = {}
    for i, suited_rank in enumerate(ranks):
        for j, offsuit_rank in enumerate(ranks):
            high = max(suited_rank, offsuit_rank, key=lambda x: ranks.index(x))
            low = min(suited_rank, offsuit_rank, key=lambda x: ranks.index(x))
            suited = ranks.index(suited_rank) > ranks.index(offsuit_rank)
            key = (high, low, suited)
            table[key] = table_values[i][j]
    return table


preflop_table = generate_preflop_table()

def get_ground_truth(card1, card2):
    r1, s1 = card1[0], card1[1]
    r2, s2 = card2[0], card2[1]
    suited = s1 == s2
    ranks_ordered = sorted([r1, r2], key=lambda r: RANKS.index(r), reverse=True)
    key = (ranks_ordered[0], ranks_ordered[1], suited)
    return preflop_table.get(key, None)

# Run Estimator
def run_estimator(card1, card2, sims=5000):
    est_prob = mcts([card1, card2], simulations=sims)
    gt_prob = get_ground_truth(card1, card2)
    print(f"\nEstimated win probability for [{card1} {card2}]: {est_prob:.4f}")
    if gt_prob is not None:
        print(f"Ground-truth winrate from table: {gt_prob:.4f}")
        print(f"Difference (est - truth): {est_prob - gt_prob:+.4f}")
    else:
        print("Ground-truth not available for this hand.")

# run
if __name__ == "__main__":
    run_estimator("Ah", "Kd", sims=5000) # Ace of hearts, King of Diamonds, no. of simulations

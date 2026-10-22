import argparse
import copy
import math
import random
import sys
import time
from typing import Dict, List, Optional, Tuple


# ══════════════════════════════════════════════════════════════════
# §1  SUDOKU CSP REPRESENTATION
# ══════════════════════════════════════════════════════════════════

class SudokuCSP:
    """
    Represents a Sudoku puzzle as a Constraint Satisfaction Problem (CSP).

    CSP formulation
    ───────────────
    • Variables  : every cell (r, c) with r,c ∈ {0, …, size-1}
    • Domains    : each cell holds the set of values {1, …, size} not yet
                   ruled out by constraint propagation
    • Constraints: AllDifferent on each row, column, and box
    """

    def __init__(self, size: int = 9):
        """
        Initialise an empty Sudoku CSP of the given size.
        size must be a perfect square (4, 9, 16, 25, …).
        """
        self.size = size
        self.box_size = int(math.isqrt(size))
        if self.box_size * self.box_size != size:
            raise ValueError(f"size must be a perfect square, got {size}")

        # domains[r][c] = set of values still legal for cell (r, c)
        self.domains: List[List[set]] = [
            [set(range(1, size + 1)) for _ in range(size)]
            for _ in range(size)
        ]

        # grid[r][c] = assigned value; 0 = unassigned
        self.grid: List[List[int]] = [[0] * size for _ in range(size)]

        # Search statistics
        self.nodes_visited: int = 0    # total nodes expanded
        self.backtracks:    int = 0    # total backtrack events

    # ─── loading ──────────────────────────────────────────────────

    def load_puzzle(self, puzzle: List[List[int]]) -> bool:
        """
        Initialise the CSP from a 2-D puzzle grid (0 = empty cell).
        Returns False if the puzzle is immediately inconsistent.
        """
        for r in range(self.size):
            for c in range(self.size):
                v = puzzle[r][c]
                if v != 0:
                    self.grid[r][c] = v
                    self.domains[r][c] = {v}

        # Propagate all pre-filled cells
        for r in range(self.size):
            for c in range(self.size):
                if self.grid[r][c] != 0:
                    if not self._propagate(r, c, self.grid[r][c]):
                        return False  # contradiction in the given clues
        return True

    # ─── constraint propagation (forward checking) ────────────────

    def _propagate(self, row: int, col: int, value: int) -> bool:
        """
        Forward-checking propagation: remove *value* from the domains of
        every unassigned cell that shares a constraint with (row, col).

        Returns False if any domain becomes empty (domain wipe-out).
        """
        for pr, pc in self._peers(row, col):
            if self.grid[pr][pc] == 0:          # only unassigned peers
                self.domains[pr][pc].discard(value)
                if not self.domains[pr][pc]:    # domain wipe-out detected
                    return False
        return True

    # ─── peer set ─────────────────────────────────────────────────

    def _peers(self, row: int, col: int) -> List[Tuple[int, int]]:
        """
        Return all cells that share at least one constraint with (row, col):
        same row, same column, or same box.
        """
        peers: set = set()

        # same row
        for c in range(self.size):
            if c != col:
                peers.add((row, c))

        # same column
        for r in range(self.size):
            if r != row:
                peers.add((r, col))

        # same box
        br = (row // self.box_size) * self.box_size
        bc = (col // self.box_size) * self.box_size
        for r in range(br, br + self.box_size):
            for c in range(bc, bc + self.box_size):
                if (r, c) != (row, col):
                    peers.add((r, c))

        return list(peers)

    # ─── helpers ──────────────────────────────────────────────────

    def unassigned(self) -> List[Tuple[int, int]]:
        """Return all unassigned cells."""
        return [(r, c)
                for r in range(self.size)
                for c in range(self.size)
                if self.grid[r][c] == 0]

    def is_solved(self) -> bool:
        """True when every cell has been assigned."""
        return all(self.grid[r][c] != 0
                   for r in range(self.size)
                   for c in range(self.size))

    # ─── display ──────────────────────────────────────────────────

    def print_grid(self, show_domains: bool = False):
        """Pretty-print the current grid (and optionally domain sizes)."""
        sep = ("─" * (self.size * 3 + self.box_size - 1))
        for r in range(self.size):
            if r % self.box_size == 0 and r != 0:
                print(sep)
            row_str = ""
            for c in range(self.size):
                if c % self.box_size == 0 and c != 0:
                    row_str += "│ "
                v = self.grid[r][c]
                if v != 0:
                    row_str += f"{v:2d} "
                elif show_domains:
                    row_str += f"({len(self.domains[r][c])}) "
                else:
                    row_str += " .  "
            print(row_str)
        print()


# ══════════════════════════════════════════════════════════════════
# §2  HEURISTICS
# ══════════════════════════════════════════════════════════════════

def mrv_select(csp: SudokuCSP) -> Optional[Tuple[int, int]]:
    """
    MRV (Minimum Remaining Values) — variable selection heuristic.

    Chooses the unassigned cell with the *smallest* domain.
    Rationale ("fail-first"): constraining cells are tried first, so
    failures are detected earlier and the search tree is pruned more
    aggressively.

    Tie-breaking via Degree Heuristic:
        Among cells with equal domain size, choose the one with the most
        constraints on *other unassigned* cells (highest degree).
        This maximises constraint propagation on the next assignment.
    """
    cells = csp.unassigned()
    if not cells:
        return None

    min_domain = min(len(csp.domains[r][c]) for r, c in cells)
    candidates = [(r, c) for r, c in cells
                  if len(csp.domains[r][c]) == min_domain]

    if len(candidates) == 1:
        return candidates[0]

    # Degree heuristic tie-breaker
    unassigned_set = set(cells)

    def degree(cell: Tuple[int, int]) -> int:
        r, c = cell
        return sum(1 for p in csp._peers(r, c) if p in unassigned_set)

    return max(candidates, key=degree)


def naive_select(csp: SudokuCSP) -> Optional[Tuple[int, int]]:
    """
    Baseline variable selection: first unassigned cell in reading order.
    Used to demonstrate the benefit of MRV.
    """
    for r in range(csp.size):
        for c in range(csp.size):
            if csp.grid[r][c] == 0:
                return (r, c)
    return None


def lcv_order(csp: SudokuCSP, row: int, col: int) -> List[int]:
    """
    LCV (Least Constraining Value) — value ordering heuristic.

    Orders the values in the domain of (row, col) from least to most
    constraining.  A value is "less constraining" when assigning it
    removes fewer options from neighbouring cells — keeping the future
    search space as large as possible.
    """
    def ruled_out(v: int) -> int:
        return sum(
            1 for pr, pc in csp._peers(row, col)
            if csp.grid[pr][pc] == 0 and v in csp.domains[pr][pc]
        )

    return sorted(csp.domains[row][col], key=ruled_out)


# ══════════════════════════════════════════════════════════════════
# §3  BACKTRACKING SOLVER
# ══════════════════════════════════════════════════════════════════

def _backtrack(csp: SudokuCSP,
               use_mrv: bool,
               use_lcv: bool) -> bool:
    """
    Recursive backtracking search (internal).

    At each step:
      1. Select the next variable (MRV + degree, or naive order).
      2. Iterate over its values (LCV order, or natural order).
      3. Assign the value and propagate (forward checking).
      4. Recurse; backtrack if inconsistency is detected.

    The state is saved/restored via deepcopy so backtracking is exact.
    """
    csp.nodes_visited += 1

    if csp.is_solved():
        return True

    # ── variable selection ──────────────────────────────────
    cell = mrv_select(csp) if use_mrv else naive_select(csp)
    if cell is None:
        return False   # should not happen in a well-formed puzzle

    row, col = cell

    # ── value ordering ──────────────────────────────────────
    values = lcv_order(csp, row, col) if use_lcv else list(csp.domains[row][col])

    for value in values:
        # ── save state for backtracking ──────────────────────
        saved_domains = copy.deepcopy(csp.domains)
        saved_grid    = copy.deepcopy(csp.grid)

        # ── assign ──────────────────────────────────────────
        csp.grid[row][col]    = value
        csp.domains[row][col] = {value}

        # ── forward checking ─────────────────────────────────
        consistent = csp._propagate(row, col, value)

        if consistent:
            if _backtrack(csp, use_mrv, use_lcv):
                return True                 # solution found

        # ── backtrack: restore saved state ───────────────────
        csp.backtracks  += 1
        csp.domains      = saved_domains
        csp.grid         = saved_grid

    return False   # no value worked → backtrack further


def solve(puzzle: List[List[int]],
          size:    int  = 9,
          use_mrv: bool = True,
          use_lcv: bool = True
          ) -> Tuple[Optional[SudokuCSP], float]:
    """
    Public entry point: solve a Sudoku puzzle.

    Parameters
    ----------
    puzzle  : 2-D list, 0 = empty cell
    size    : board size (must be a perfect square)
    use_mrv : enable MRV + degree heuristic
    use_lcv : enable LCV value ordering

    Returns
    -------
    (SudokuCSP, elapsed_seconds) — CSP contains the solved grid.
    Returns (None, elapsed) if the puzzle has no solution.
    """
    csp = SudokuCSP(size)
    ok  = csp.load_puzzle(puzzle)

    start   = time.perf_counter()
    solved  = ok and _backtrack(csp, use_mrv, use_lcv)
    elapsed = time.perf_counter() - start

    return (csp if solved else None, elapsed)


# ══════════════════════════════════════════════════════════════════
# §4  PUZZLE GENERATOR
# ══════════════════════════════════════════════════════════════════

def generate_puzzle(size: int = 9,
                    difficulty: float = 0.55,
                    seed: Optional[int] = None
                    ) -> List[List[int]]:
    """
    Generate a random, solvable Sudoku puzzle.

    Strategy
    --------
    1. Fill diagonal boxes (they are mutually independent — no shared
       constraints — so they can be filled independently with random
       permutations without violating any constraint).
    2. Solve the rest of the grid to produce a complete solution.
    3. Remove `difficulty × size²` cells at random.

    Parameters
    ----------
    size       : board size (4, 9, 16)
    difficulty : fraction of cells to blank out [0 = full, 1 = empty]
    seed       : random seed for reproducibility
    """
    if seed is not None:
        random.seed(seed)

    box_size = int(math.isqrt(size))
    grid     = [[0] * size for _ in range(size)]

    # ── step 1: fill diagonal boxes ─────────────────────────
    for box_idx in range(size // box_size):
        vals = list(range(1, size + 1))
        random.shuffle(vals)
        br, bc = box_idx * box_size, box_idx * box_size
        k = 0
        for r in range(br, br + box_size):
            for c in range(bc, bc + box_size):
                grid[r][c] = vals[k]
                k += 1

    # ── step 2: complete the grid ────────────────────────────
    csp = SudokuCSP(size)
    csp.load_puzzle(grid)
    _backtrack(csp, use_mrv=True, use_lcv=False)   # use fast config for generation
    full_grid = [row[:] for row in csp.grid]

    # ── step 3: remove cells ─────────────────────────────────
    cells = [(r, c) for r in range(size) for c in range(size)]
    random.shuffle(cells)
    remove_count = int(size * size * difficulty)
    for r, c in cells[:remove_count]:
        full_grid[r][c] = 0

    return full_grid


# ══════════════════════════════════════════════════════════════════
# §5  BUILT-IN BENCHMARK PUZZLES
# ══════════════════════════════════════════════════════════════════

# ── 4×4 ──────────────────────────────────────────────────────────
PUZZLE_4 = [
    [1, 0, 0, 4],
    [0, 4, 1, 0],
    [0, 1, 4, 0],
    [4, 0, 0, 1],
]

# ── 9×9 (classic "hard" puzzle) ───────────────────────────────────
PUZZLE_9 = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
]

# ── 9×9 "very hard" (AI Escargot — one of the hardest known) ──────
PUZZLE_9_HARD = [
    [1, 0, 0, 0, 0, 7, 0, 9, 0],
    [0, 3, 0, 0, 2, 0, 0, 0, 8],
    [0, 0, 9, 6, 0, 0, 5, 0, 0],
    [0, 0, 5, 3, 0, 0, 9, 0, 0],
    [0, 1, 0, 0, 8, 0, 0, 0, 2],
    [6, 0, 0, 0, 0, 4, 0, 0, 0],
    [3, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 4, 0, 0, 0, 0, 0, 0, 7],
    [0, 0, 7, 0, 0, 0, 3, 0, 0],
]

# ── 16×16 (generated; 0 = empty) ──────────────────────────────────
# Pre-generated for deterministic benchmarks.
PUZZLE_16 = [
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 11,  0,  0,  0,  0],
    [ 0,  5,  0,  0,  0,  3,  0,  0,  0,  0,  0,  0, 16,  0,  0,  0],
    [ 0,  0, 12,  0,  0,  0,  0,  1,  0,  0,  6,  0,  0,  0,  0,  0],
    [ 0,  0,  0, 16,  0,  0,  0,  0,  3,  0,  0,  0,  0,  4,  0,  0],
    [ 0,  0,  5,  0,  0,  0,  0,  0,  0, 12,  0,  0,  0,  0, 15,  0],
    [ 0,  0,  0,  0,  8,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 11],
    [ 0,  0,  0, 14,  0,  0,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [10,  0,  0,  0,  0,  0,  0, 16,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  6,  0,  0,  0,  0,  0,  0, 13],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0, 11,  0,  0, 10,  0,  0,  0],
    [ 7,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  4,  0,  0,  0,  0,  0,  0,  0,  5,  0,  0,  0,  0],
    [ 0,  0,  9,  0,  0,  0,  0,  0,  0,  0,  0,  0,  7,  0,  0,  0],
    [ 0, 16,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  3,  0,  0],
    [ 0,  0,  0,  0, 15,  0,  0,  0,  0,  0,  0,  0,  0,  0,  8,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  4,  0,  0,  0,  0,  0,  0,  0,  2],
]


# ══════════════════════════════════════════════════════════════════
# §6  BENCHMARKING
# ══════════════════════════════════════════════════════════════════

def run_benchmarks(repetitions: int = 1) -> List[Dict]:
    """
    Run benchmarks for 4×4, 9×9, and 16×16 puzzles and print a
    formatted comparison table.

    Each size is tested twice:
      • With    MRV + forward checking + LCV
      • Without heuristics (baseline backtracking)

    The 16×16 baseline is skipped by default (can be very slow).
    """
    configs = [
        (4,  PUZZLE_4,      "given"),
        (9,  PUZZLE_9,      "given"),
        (9,  PUZZLE_9_HARD, "hard" ),
        (16, PUZZLE_16,     "given"),
    ]

    all_results = []

    print("\n" + "═" * 80)
    print(f"  {'Size':>4}  {'Puzzle':>8}  {'Heuristics':>12}  "
          f"{'Time (s)':>10}  {'Nodes':>8}  {'Backtracks':>10}  {'Status':>6}")
    print("═" * 80)

    for size, puzzle, label in configs:
        for use_h in [True, False]:
            # Skip baseline for 16×16 (would take too long for a classroom demo)
            if size == 16 and not use_h:
                continue

            times, nodes_list, bt_list = [], [], []
            for _ in range(repetitions):
                csp_result, elapsed = solve(
                    puzzle, size,
                    use_mrv=use_h,
                    use_lcv=use_h,
                )
                if csp_result:
                    times.append(elapsed)
                    nodes_list.append(csp_result.nodes_visited)
                    bt_list.append(csp_result.backtracks)

            avg_time = sum(times) / len(times)       if times else -1
            avg_nodes = sum(nodes_list) / len(nodes_list) if nodes_list else -1
            avg_bt    = sum(bt_list) / len(bt_list)  if bt_list else -1
            ok = "✓" if times else "✗"
            heur_str = "MRV+FC+LCV" if use_h else "Baseline  "

            print(f"  {size:>4}  {label:>8}  {heur_str:>12}  "
                  f"{avg_time:>10.4f}  {avg_nodes:>8.0f}  "
                  f"{avg_bt:>10.0f}  {ok:>6}")

            all_results.append({
                "size":       size,
                "label":      label,
                "heuristics": use_h,
                "time":       avg_time,
                "nodes":      avg_nodes,
                "backtracks": avg_bt,
                "solved":     bool(times),
            })

    print("═" * 80)
    print("  Note: 16×16 with heuristics may take a few seconds on hard puzzles.")
    print("        Baseline (no heuristics) skipped for 16×16 to avoid timeout.\n")
    return all_results


# ══════════════════════════════════════════════════════════════════
# §7  OPTIONAL: OR-TOOLS CP-SAT COMPARISON
# ══════════════════════════════════════════════════════════════════

def solve_with_ortools(puzzle: List[List[int]],
                       size:   int = 9
                       ) -> Tuple[Optional[List[List[int]]], float]:
    """
    Solve a Sudoku using OR-Tools CP-SAT solver (if installed).

    Install: pip install ortools

    This provides an industrial-strength baseline for comparison with
    our hand-rolled backtracking implementation.
    """
    try:
        from ortools.sat.python import cp_model
    except ImportError:
        print("  OR-Tools not installed. Run: pip install ortools")
        return None, 0.0

    model  = cp_model.CpModel()
    solver = cp_model.CpSolver()
    box    = int(math.isqrt(size))

    # ── variables ────────────────────────────────────────────
    cells = [[model.NewIntVar(1, size, f"c_{r}_{c}")
              for c in range(size)]
             for r in range(size)]

    # ── clue constraints ─────────────────────────────────────
    for r in range(size):
        for c in range(size):
            if puzzle[r][c] != 0:
                model.Add(cells[r][c] == puzzle[r][c])

    # ── AllDifferent on rows, columns, boxes ─────────────────
    for r in range(size):
        model.AddAllDifferent(cells[r])
    for c in range(size):
        model.AddAllDifferent([cells[r][c] for r in range(size)])
    for br in range(0, size, box):
        for bc in range(0, size, box):
            model.AddAllDifferent([
                cells[r][c]
                for r in range(br, br + box)
                for c in range(bc, bc + box)
            ])

    start  = time.perf_counter()
    status = solver.Solve(model)
    elapsed = time.perf_counter() - start

    if status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        solution = [[solver.Value(cells[r][c]) for c in range(size)]
                    for r in range(size)]
        print(f"  OR-Tools CP-SAT: solved in {elapsed:.4f}s "
              f"(branches={solver.NumBranches()}, "
              f"conflicts={solver.NumConflicts()})")
        return solution, elapsed
    else:
        print("  OR-Tools: no solution found")
        return None, elapsed


# ══════════════════════════════════════════════════════════════════
# §8  AI ASSISTANT
# ══════════════════════════════════════════════════════════════════

class AIAssistant:
    """
    An AI assistant for Sudoku / CSP solving.

    Mode A — Heuristic (always available, sound)
    ─────────────────────────────────────────────
    Uses MRV and LCV internally to suggest the next move.
    The recommendation is provably correct (just the heuristic itself).

    Mode B — LLM (optional, EXPERIMENTAL, UNSOUND)
    ────────────────────────────────────────────────
    Calls the Anthropic Claude API to generate a natural-language
    explanation of why a cell has its current domain.

    ⚠️  DISCLAIMER: LLM outputs are generated by a language model and
    may contain factual errors or invalid constraint reasoning.
    Never use LLM output directly to guide a constraint solver —
    always verify programmatically.  This mode is labelled
    "experimental" and is provided purely for illustration.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key  = api_key
        self.llm_mode = api_key is not None

    # ─── heuristic suggestion ─────────────────────────────────────

    def suggest(self, csp: SudokuCSP) -> Dict:
        """
        Return a heuristic-based suggestion for the next move.
        """
        cell = mrv_select(csp)
        if cell is None:
            return {"message": "Puzzle is already solved!"}

        row, col = cell
        domain   = sorted(csp.domains[row][col])
        values   = lcv_order(csp, row, col)
        best_val = values[0] if values else None

        peers_assigned = sum(
            1 for pr, pc in csp._peers(row, col)
            if csp.grid[pr][pc] != 0
        )

        return {
            "cell":         (row + 1, col + 1),   # 1-indexed for display
            "domain":       domain,
            "domain_size":  len(domain),
            "suggested":    best_val,
            "peer_count":   peers_assigned,
            "reasoning": (
                f"Cell ({row+1},{col+1}) has the smallest domain "
                f"{domain} (size={len(domain)}) — chosen by MRV. "
                f"Value {best_val} is least constraining (LCV)."
            ),
            "confidence":   "heuristic — sound",
        }

    def print_suggestion(self, csp: SudokuCSP):
        """Pretty-print a move suggestion."""
        s = self.suggest(csp)
        if "message" in s:
            print("  " + s["message"])
            return

        print("\n  ┌─────────────────────────────────────────┐")
        print("  │         🤖  AI ASSISTANT SUGGESTION      │")
        print("  ├─────────────────────────────────────────┤")
        print(f"  │  Target cell    : ({s['cell'][0]}, {s['cell'][1]})            ")
        print(f"  │  Domain         : {s['domain']}")
        print(f"  │  Suggested value: {s['suggested']}")
        print(f"  │  Reasoning      : {s['reasoning'][:60]}")
        print(f"  │                   {s['reasoning'][60:]}")
        print(f"  │  Confidence     : {s['confidence']}")
        print("  └─────────────────────────────────────────┘\n")

    # ─── LLM explanation ──────────────────────────────────────────

    def explain_with_llm(self, csp: SudokuCSP,
                         row: int, col: int) -> str:
        """
        Ask the Claude API to explain in natural language why cell (row, col)
        has its current domain.

        ⚠️  EXPERIMENTAL — output may be incorrect.
        Requires: pip install anthropic
        """
        if not self.llm_mode:
            return ("LLM mode disabled.  Pass api_key= to AIAssistant() "
                    "to enable this experimental feature.")

        try:
            import anthropic
        except ImportError:
            return "Install the anthropic package first: pip install anthropic"

        domain   = sorted(csp.domains[row][col])
        box_r    = (row // csp.box_size) * csp.box_size
        box_c    = (col // csp.box_size) * csp.box_size
        row_vals = sorted({csp.grid[row][c]
                           for c in range(csp.size) if csp.grid[row][c] != 0})
        col_vals = sorted({csp.grid[r][col]
                           for r in range(csp.size) if csp.grid[r][col] != 0})
        box_vals = sorted({
            csp.grid[r][c]
            for r in range(box_r, box_r + csp.box_size)
            for c in range(box_c, box_c + csp.box_size)
            if csp.grid[r][c] != 0
        })

        prompt = (
            f"You are a Constraint Satisfaction tutor explaining Sudoku to a student.\n\n"
            f"Puzzle size  : {csp.size}×{csp.size}\n"
            f"Cell         : row {row+1}, column {col+1}  (1-indexed)\n"
            f"Current domain: {domain}\n"
            f"Row values   : {row_vals}\n"
            f"Column values: {col_vals}\n"
            f"Box values   : {box_vals}\n\n"
            f"In 2–3 sentences, explain to the student why the domain of this cell "
            f"is {domain}.  Reference the CSP constraints (row, column, box) "
            f"and forward checking."
        )

        try:
            client  = anthropic.Anthropic(api_key=self.api_key)
            message = client.messages.create(
                model      = "claude-opus-4-5",
                max_tokens = 300,
                messages   = [{"role": "user", "content": prompt}],
            )
            raw = message.content[0].text
        except Exception as exc:
            return f"API error: {exc}"

        banner = (
            "\n  ╔══════════════════════════════════════════════════╗\n"
            "  ║  ⚠️  LLM EXPLANATION — EXPERIMENTAL / UNSOUND  ║\n"
            "  ║  Do NOT use this output to guide the solver.    ║\n"
            "  ╚══════════════════════════════════════════════════╝\n"
        )
        return banner + raw


# ══════════════════════════════════════════════════════════════════
# §9  TEST SUITE
# ══════════════════════════════════════════════════════════════════

def _check_solution(csp: SudokuCSP) -> bool:
    """Verify that a solved grid satisfies all Sudoku constraints."""
    size     = csp.size
    box_size = csp.box_size
    expected = list(range(1, size + 1))

    for r in range(size):
        if sorted(csp.grid[r]) != expected:
            return False
    for c in range(size):
        if sorted(csp.grid[r][c] for r in range(size)) != expected:
            return False
    for br in range(0, size, box_size):
        for bc in range(0, size, box_size):
            box = [csp.grid[r][c]
                   for r in range(br, br + box_size)
                   for c in range(bc, bc + box_size)]
            if sorted(box) != expected:
                return False
    return True


def run_tests():
    """
    Minimal test suite — run directly or via pytest.
    Each test asserts correctness and prints timing.
    """
    print("\n" + "═" * 50)
    print("  RUNNING TEST SUITE")
    print("═" * 50)

    # ── test 1: 4×4 ─────────────────────────────────────────
    csp4, t4 = solve(PUZZLE_4, 4)
    assert csp4 is not None,    "test_4x4: should be solvable"
    assert csp4.is_solved(),    "test_4x4: should be marked solved"
    assert _check_solution(csp4), "test_4x4: solution invalid"
    print(f"  ✓ test_4x4          ({t4:.4f}s,  {csp4.nodes_visited} nodes)")

    # ── test 2: 9×9 easy ─────────────────────────────────────
    csp9, t9 = solve(PUZZLE_9, 9)
    assert csp9 is not None,    "test_9x9_easy: should be solvable"
    assert _check_solution(csp9), "test_9x9_easy: solution invalid"
    print(f"  ✓ test_9x9_easy     ({t9:.4f}s,  {csp9.nodes_visited} nodes)")

    # ── test 3: 9×9 hard ─────────────────────────────────────
    csp9h, t9h = solve(PUZZLE_9_HARD, 9)
    assert csp9h is not None,     "test_9x9_hard: should be solvable"
    assert _check_solution(csp9h),"test_9x9_hard: solution invalid"
    print(f"  ✓ test_9x9_hard     ({t9h:.4f}s,  {csp9h.nodes_visited} nodes)")

    # ── test 4: baseline vs MRV (nodes should decrease) ──────
    csp_base, _  = solve(PUZZLE_9, 9, use_mrv=False, use_lcv=False)
    csp_mrv,  _  = solve(PUZZLE_9, 9, use_mrv=True,  use_lcv=True )
    assert csp_base is not None and csp_mrv is not None
    # MRV should visit no more nodes than naive (usually far fewer)
    assert csp_mrv.nodes_visited <= csp_base.nodes_visited + 1, (
        f"MRV should not visit more nodes than baseline "
        f"({csp_mrv.nodes_visited} vs {csp_base.nodes_visited})"
    )
    print(f"  ✓ test_mrv_efficiency "
          f"(baseline={csp_base.nodes_visited}, MRV={csp_mrv.nodes_visited})")

    # ── test 5: generator produces solvable puzzles ───────────
    for sz in [4, 9]:
        p   = generate_puzzle(sz, difficulty=0.5, seed=42)
        csp_, _ = solve(p, sz)
        assert csp_ is not None, f"test_generator_{sz}: generated puzzle unsolvable"
    print(f"  ✓ test_generator    (4×4 and 9×9)")

    # ── test 6: AI assistant suggestion ──────────────────────
    partial = SudokuCSP(9)
    partial.load_puzzle(PUZZLE_9)
    assistant = AIAssistant()
    s = assistant.suggest(partial)
    assert "cell" in s and "suggested" in s, "test_assistant: bad output"
    print(f"  ✓ test_ai_assistant (suggested cell {s['cell']})")

    print("═" * 50)
    print("  All tests passed ✓\n")


# ══════════════════════════════════════════════════════════════════
# §10  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="CSP Sudoku Solver — DS1 2025/2026"
    )
    parser.add_argument("--test",  action="store_true", help="Run test suite only")
    parser.add_argument("--bench", action="store_true", help="Run benchmarks only")
    parser.add_argument("--size",  type=int, default=9,
                        help="Puzzle size for ad-hoc generation (4, 9, or 16)")
    parser.add_argument("--llm-key", default=None,
                        help="Anthropic API key for LLM assistant (experimental)")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════╗")
    print("║  CSP Sudoku Solver   DS1 2025/2026                   ║")
    print("║  Topic 3: Constraint Satisfaction                    ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    if args.test:
        run_tests()
        return

    if args.bench:
        run_benchmarks()
        return

    # ── default: demo + tests + benchmarks ──────────────────
    run_tests()

    # ── solve and display the 9×9 puzzle ─────────────────────
    print("Solving the 9×9 'easy' puzzle …\n")
    print("  Input:")
    demo = SudokuCSP(9)
    demo.load_puzzle(PUZZLE_9)
    demo.print_grid()

    csp, elapsed = solve(PUZZLE_9, 9)
    print("  Solution:")
    csp.print_grid()
    print(f"  Elapsed    : {elapsed:.4f}s")
    print(f"  Nodes      : {csp.nodes_visited}")
    print(f"  Backtracks : {csp.backtracks}\n")

    # ── AI assistant demo (heuristic mode) ───────────────────
    print("AI Assistant demo — suggesting next move on unsolved puzzle:")
    partial = SudokuCSP(9)
    partial.load_puzzle(PUZZLE_9)
    assistant = AIAssistant(api_key=args.llm_key)
    assistant.print_suggestion(partial)

    # ── LLM demo (only if API key supplied) ──────────────────
    if args.llm_key:
        print("LLM Explanation demo (cell row=1, col=3):")
        explanation = assistant.explain_with_llm(partial, row=0, col=2)
        print(explanation)

    # ── full benchmark table ─────────────────────────────────
    print("Benchmark table:\n")
    run_benchmarks()

    # ── optional OR-Tools comparison ────────────────────────
    print("OR-Tools comparison (9×9):")
    solve_with_ortools(PUZZLE_9, size=9)
    print()


if __name__ == "__main__":
    main()

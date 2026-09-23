"""AST normalizer, structural signature extractor, and similarity metric for Arm 3: AST-RL.

Literature Basis:
- TreeDiff: Structural Code Comparison (ASE 2025)
- VeriSeek: Structure-Guided Code Synthesis Verification (ICSE 2025)

AST-RL incorporates the Abstract Syntax Tree (AST) into the reward function:
    R_AST(y) = exp(-alpha * TreeDist(AST(y), AST(y*)))

This prevents lexical surface overfitting (variable names, docstrings) while
enforcing rigorous control-flow logic and structural correctness.
"""

import ast
import math
from typing import List, Set


class ASTNormalizer(ast.NodeTransformer):
    """Normalizes AST by renaming all variables and arguments to canonical placeholder symbols.

    This isolates pure control flow and syntactic structure from superficial
    naming differences (e.g. `n` vs `element` vs `item`).
    """

    def visit_Name(self, node: ast.Name) -> ast.Name:
        return ast.copy_location(ast.Name(id="_v", ctx=node.ctx), node)

    def visit_arg(self, node: ast.arg) -> ast.arg:
        return ast.copy_location(ast.arg(arg="_a", annotation=None), node)


def get_ast_signature(code: str) -> List[str]:
    """Returns depth-first sequence of normalized AST node type names.

    Gracefully catches SyntaxErrors (returns ['SyntaxError']).
    """
    try:
        parsed = ast.parse(code)
        normalized = ASTNormalizer().visit(parsed)
        return [type(node).__name__ for node in ast.walk(normalized)]
    except SyntaxError:
        return ["SyntaxError"]
    except Exception:
        return ["SyntaxError"]


def simAST(code_gen: str, code_ref: str) -> float:
    """Calculates structural AST similarity between generated code and canonical reference in [0, 1].

    Combines:
    - Jaccard similarity of normalized AST node types (60% weight)
    - Length ratio of node sequences to penalize incomplete/truncated solutions (40% weight)
    """
    sig_gen = get_ast_signature(code_gen)
    sig_ref = get_ast_signature(code_ref)

    if "SyntaxError" in sig_gen or "SyntaxError" in sig_ref:
        return 0.0

    set_gen: Set[str] = set(sig_gen)
    set_ref: Set[str] = set(sig_ref)

    intersection = len(set_gen & set_ref)
    union = max(len(set_gen | set_ref), 1)
    jaccard = intersection / union

    len_ratio = min(len(sig_gen), len(sig_ref)) / max(len(sig_gen), len(sig_ref), 1)
    score = 0.6 * jaccard + 0.4 * len_ratio
    return round(float(score), 4)


def ast_reward(code_gen: str, code_ref: str, alpha: float = 0.05) -> float:
    """Computes TreeDiff / VeriSeek exponential AST reward.

    R_AST = exp(-alpha * (1 - simAST) * 100)
    """
    sim = simAST(code_gen, code_ref)
    tree_dist = 1.0 - sim
    return round(float(math.exp(-alpha * tree_dist * 100)), 4)

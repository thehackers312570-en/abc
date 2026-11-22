#!/usr/bin/env python3
"""
run.py - Offline Reinforcement Learning & Agent Environment Adapter.

Zero external dependencies: 100% Python Standard Library
(json, re, math, ast, difflib, collections, argparse, os, sys, pathlib, io, tokenize, textwrap).
Dynamically supports any arbitrary number of student tasks per notebook (2, 3, 4, or more),
with hard structural filtering, role-based attribute adaptation, and exact indentation preservation.
"""

import os
import sys
import json
import math
import re
import ast
import io
import tokenize
import textwrap
import difflib
import argparse
from pathlib import Path
from collections import Counter, defaultdict

# ----------------------------------------------------------------------
# 1. Lexical Ranking Engine (BM25) with Domain Noise Downweighting
# ----------------------------------------------------------------------

STOPWORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any',
    'are', 'aren', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between',
    'both', 'but', 'by', 'can', 'cannot', 'could', 'did', 'do', 'does', 'doing', 'down', 'during',
    'each', 'few', 'for', 'from', 'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
    'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'it', 'its',
    'itself', 'just', 'me', 'more', 'most', 'my', 'myself', 'no', 'nor', 'not', 'now', 'of', 'off',
    'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same',
    'she', 'should', 'so', 'some', 'such', 'than', 'that', 'the', 'their', 'theirs', 'them',
    'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too',
    'under', 'until', 'up', 'very', 'was', 'we', 'were', 'what', 'when', 'where', 'which',
    'while', 'who', 'whom', 'why', 'with', 'would', 'you', 'your', 'yours', 'yourself',
    # Downweight generic story nouns that cause false cross-category matches
    'warehouse', 'drone', 'robot', 'agent', 'grid', 'gridworld', 'episode', 'episodes',
    'simulation', 'step', 'steps', 'reward', 'rewards', 'environment', 'action', 'actions',
    'state', 'states', 'policy', 'problem', 'task', 'tasks', 'student', 'implementation',
    'solution', 'cell', 'parameter', 'parameters', 'initial', 'maximum', 'max', 'min',
    'return', 'returns', 'run', 'test', 'driver', 'performance', 'metric'
}

def tokenize_words(text):
    if not text:
        return []
    words = re.findall(r'[a-zA-Z0-9_]+', text.lower())
    tokens = []
    for w in words:
        parts = w.split('_')
        for p in parts:
            if len(p) > 1 and p not in STOPWORDS:
                tokens.append(p)
    return tokens

class BM25Ranker:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.doc_lengths = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_lengths) / len(corpus) if corpus else 1.0
        self.doc_freqs = defaultdict(int)
        self.term_freqs = []
        for doc in corpus:
            tf = Counter(doc)
            self.term_freqs.append(tf)
            for word in tf.keys():
                self.doc_freqs[word] += 1
        self.N = len(corpus)

    def score(self, query_tokens):
        scores = [0.0] * self.N
        for word in query_tokens:
            if word not in self.doc_freqs:
                continue
            df = self.doc_freqs[word]
            idf = math.log(1.0 + (self.N - df + 0.5) / (df + 0.5))
            for idx in range(self.N):
                tf = self.term_freqs[idx].get(word, 0)
                denom = tf + self.k1 * (1.0 - self.b + self.b * (self.doc_lengths[idx] / self.avgdl))
                if denom > 0:
                    scores[idx] += idf * (tf * (self.k1 + 1.0)) / denom
        return scores

# ----------------------------------------------------------------------
# 2. Dynamic Task & Structural Feature Parser
# ----------------------------------------------------------------------

def detect_unseen_task_cells(cells):
    detected = []
    for idx, cell in enumerate(cells):
        if cell.get('cell_type') != 'code':
            continue
        src = "".join(cell.get('source', []))
        if not src.strip():
            continue
        if "(LOCKED)" in src or "plt.show" in src or "rewards_history =" in src:
            continue
        is_task = (
            any(m in src for m in ["YOUR CODE HERE", "STUDENT IMPLEMENTATION", "NotImplementedError"])
            or bool(re.search(r'TASK\s*\d+', src, re.IGNORECASE))
        )
        if is_task:
            fn_name = ""
            fn_params = []
            try:
                tree = ast.parse(src)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        if node.name != '__init__' or not fn_name:
                            fn_name = node.name
                            fn_params = [a.arg for a in node.args.args]
            except Exception:
                pass
            detected.append({
                'cell_index': idx,
                'source': src,
                'func_name': fn_name,
                'params': fn_params
            })
    return detected

def extract_spec_formulas(cells):
    formulas = []
    for c in cells:
        src = "".join(c.get('source', []))
        for line in src.splitlines():
            clean = line.strip()
            clean = re.sub(r'^[-*#\d\.\s]+', '', clean).strip()
            if '=' in clean and not clean.startswith('import ') and not clean.startswith('def ') and not clean.startswith('class '):
                if any(op in clean for op in ['urgency', 'backlog', 'reduction', 'estimate', 'target', 'reward', 'priority_weight', '*', '+', '-', 'max(', 'min(']):
                    if 8 < len(clean) < 120 and clean not in formulas:
                        formulas.append(clean)
    return formulas

def parse_notebook(nb_path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    cells = nb.get('cells', [])
    cell0_md = "".join(cells[0].get('source', [])) if len(cells) > 0 else ""
    
    setup_code = ""
    if len(cells) > 1:
        setup_code += "".join(cells[1].get('source', [])) + "\n"
    if len(cells) > 2:
        setup_code += "".join(cells[2].get('source', [])) + "\n"
        
    detected_tasks = detect_unseen_task_cells(cells)
    target_code = "\n".join(t['source'] for t in detected_tasks)
    all_code = setup_code + "\n" + target_code
    
    constants = set()
    attributes = set()
    classes = []
    methods = []
    method_params = {}
    init_attrs = []
    agent_class_name = ""
    
    try:
        tree = ast.parse(all_code)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
                if 'Agent' in node.name or 'Manager' in node.name or (detected_tasks and node.name in detected_tasks[0]['source']):
                    agent_class_name = node.name
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                            for sub in ast.walk(item):
                                if isinstance(sub, ast.Assign):
                                    for tgt in sub.targets:
                                        if isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name) and tgt.value.id in ('self', 'manager'):
                                            if tgt.attr not in init_attrs:
                                                init_attrs.append(tgt.attr)
            elif isinstance(node, ast.FunctionDef):
                methods.append(node.name)
                method_params[node.name] = [a.arg for a in node.args.args]
            elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in ('self', 'manager'):
                attributes.add(node.attr)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        constants.add(target.id)
    except Exception:
        pass
        
    # Check requires_q_table
    q_names = {'q_table', 'q_values', 'q_vec', 'q_matrix', 'q'}
    has_q_attr = bool(set(init_attrs) & q_names)
    
    task_param_names = set()
    for t in detected_tasks:
        for p in t.get('params', []):
            task_param_names.add(p)
    has_q_param = bool(task_param_names & q_names)
    
    # Check task docstrings for explicit Q-learning requests
    task_doc_text = ""
    for t in detected_tasks:
        m = re.search(r'"""(.*?)"""', t['source'], re.DOTALL)
        if m:
            task_doc_text += " " + m.group(1).lower()
            
    has_q_doc = any(k in task_doc_text for k in ['q-learning update', 'bellman update', 'q_table', 'q_values', 'q_vec'])
    
    requires_q_table = bool(has_q_attr or has_q_param or has_q_doc)
    has_class = bool(agent_class_name or init_attrs)
    spec_formulas = extract_spec_formulas(cells)
    
    return {
        'path': nb_path,
        'cells': cells,
        'cell0_md': cell0_md,
        'setup_code': setup_code,
        'detected_tasks': detected_tasks,
        'target_code': target_code,
        'all_code': all_code,
        'constants': sorted(list(constants)),
        'attributes': sorted(list(attributes)),
        'classes': sorted(list(set(classes))),
        'methods': sorted(list(set(methods))),
        'method_params': method_params,
        'init_attrs': init_attrs,
        'agent_class_name': agent_class_name,
        'requires_q_table': requires_q_table,
        'has_class': has_class,
        'num_tasks': len(detected_tasks),
        'spec_formulas': spec_formulas
    }

# ----------------------------------------------------------------------
# 3. Hybrid Ranking with Compatibility Gates
# ----------------------------------------------------------------------

KEYWORD_BOOSTS = {
    'traffic_priority': ['traffic', 'queue', 'urgency', 'backlog', 'clearance', 'efficiency', 'priority', 'phase', 'signal', 'waiting', 'bin', 'bins'],
    'game_theory': ['dominated', 'dominance', 'nash', 'elimination', 'payoffs', 'equilibrium'],
    'contract_auction': ['contract', 'bidding', 'contractor', 'auction', 'cnp', 'bid', 'supplier', 'reliability', 'workload'],
    'tabular_q': ['bellman', 'td', 'q_table', 'q_values', 'q_vec', 'q-value', 'explore_prob', 'epsilon-greedy'],
    'discretized': ['discretize', 'continuous', 'gap', 'pipe', 'flappy'],
    'pursuit': ['pursuit', 'evasion', 'predator', 'prey', 'capture'],
    'stochastic': ['stochastic', 'icy', 'slick', 'slip', 'hazard']
}

def rank_matches(parsed_nb, dataset):
    corpus = []
    for item in dataset:
        text = f"{item['title']} {item['category']} {item['problem_summary']} "
        text += " ".join(item['symbols']['constants']) + " "
        text += " ".join(item['symbols']['attributes']) + " "
        text += " ".join(item['symbols']['methods']) + " "
        hints = item.get('step_by_step_hints', {})
        for h in hints.get('task1', []) + hints.get('task2', []):
            text += h + " "
        for t in item.get('tasks', []):
            text += f" {t.get('func_name', '')} {t.get('docstring', '')} "
        corpus.append(tokenize_words(text))
        
    bm25 = BM25Ranker(corpus)
    
    query_text = f"{parsed_nb['cell0_md']} {parsed_nb['target_code']} "
    query_text += " ".join(parsed_nb['constants']) + " "
    query_text += " ".join(parsed_nb['attributes']) + " "
    query_text += " ".join(parsed_nb['methods']) + " "
    query_text += " ".join(parsed_nb.get('spec_formulas', []))
    query_tokens = tokenize_words(query_text)
    
    bm25_scores = bm25.score(query_tokens)
    max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
    
    nb_methods = set(parsed_nb['methods'])
    nb_attrs = set(parsed_nb['attributes'])
    nb_consts = set(parsed_nb['constants'])
    nb_classes = set(parsed_nb['classes'])
    query_lower = query_text.lower()
    
    unseen_q = parsed_nb['requires_q_table']
    unseen_tasks_count = parsed_nb['num_tasks']
    unseen_has_class = parsed_nb['has_class']
    
    final_scores = []
    for idx, item in enumerate(dataset):
        # 1. HARD STRUCTURAL GATE: Q-table Compatibility
        item_q = item.get('requires_q_table')
        if item_q is not None and item_q != unseen_q:
            continue  # Strictly eliminate incompatible algorithmic families
            
        # 2. Task Count Filter
        item_tasks_count = item.get('num_tasks', len(item.get('tasks', [])))
        if abs(item_tasks_count - unseen_tasks_count) >= 2:
            continue  # Skip 4-task vs 2-task mismatches
            
        norm_bm25 = bm25_scores[idx] / max_bm25
        
        ref_symbols = item['symbols']
        ref_methods = set(ref_symbols.get('methods', []))
        ref_attrs = set(ref_symbols.get('attributes', []))
        ref_consts = set(ref_symbols.get('constants', []))
        ref_classes = set(ref_symbols.get('classes', []))
        
        method_ov = len(nb_methods & ref_methods) / max(1, len(ref_methods))
        attr_ov = len(nb_attrs & ref_attrs) / max(1, len(ref_attrs))
        const_ov = len(nb_consts & ref_consts) / max(1, len(ref_consts))
        class_ov = len(nb_classes & ref_classes) / max(1, len(ref_classes))
        
        kw_boost = 0.0
        for theme, words in KEYWORD_BOOSTS.items():
            if any(w in query_lower for w in words):
                if any(w in item['category'].lower() or w in item['title'].lower() or w in item['id'].lower() for w in words):
                    kw_boost += 0.5
                    
        struct_score = (0.35 * method_ov + 0.25 * attr_ov + 0.20 * const_ov + 0.20 * class_ov)
        total_score = 0.40 * norm_bm25 + 0.40 * struct_score + 0.20 * min(1.0, kw_boost)
        final_scores.append((total_score, item))
        
    final_scores.sort(key=lambda x: x[0], reverse=True)
    return final_scores

# ----------------------------------------------------------------------
# 4. Role-Based Attribute Adaptation & Safe Token Renaming
# ----------------------------------------------------------------------

ATTR_SYNONYMS = {
    'alpha': ['lr', 'learning_rate', 'step_size', 'lr_alpha'],
    'gamma': ['discount', 'discount_factor', 'decay', 'discount_gamma', 'trust_weight'],
    'epsilon': ['eps', 'explore', 'exploration_rate', 'explore_prob', 'explore_eps', 'explore_rate'],
    'q_table': ['q_values', 'q_matrix', 'table', 'values', 'q_vec', 'q'],
    'num_actions': ['n_actions', 'actions', 'action_count'],
    'num_states': ['n_states', 'states', 'state_count'],
    'last_action': ['last_wing', 'last_bin', 'active_bin', 'current_phase'],
    'current_phase': ['last_bin', 'last_wing', 'active_bin', 'last_action'],
    'service_est': ['efficiency_estimate', 'clearance_est', 'service_estimate', 'wait_estimate'],
    'service_estimate': ['efficiency_estimate', 'clearance_est', 'wait_estimate', 'service_est'],
    'wait_estimate': ['efficiency_estimate', 'clearance_est', 'service_estimate', 'service_est'],
    'trust': ['reliability_est', 'trust'],
    'workload': ['jobs_done', 'workload'],
}

CONST_SYNONYMS = {
    'ALPHA': ['LEARNING_RATE', 'LR', 'LR_ALPHA', 'STEP_SIZE'],
    'GAMMA': ['DISCOUNT_FACTOR', 'DISCOUNT', 'DISCOUNT_GAMMA', 'DECAY', 'TRUST_WEIGHT'],
    'EPSILON': ['EXPLORATION_RATE', 'EXPLORE_PROB', 'EXPLORE_EPS', 'EXPLORE_RATE', 'EPS'],
    'GRID_SIZE': ['GRID_DIM', 'DIM', 'SIZE', 'WORLD_SIZE'],
    'MAX_STEPS_PER_EPISODE': ['MAX_STEPS', 'STEPS_LIMIT', 'HORIZON']
}

def safe_rename_tokens(code_str, rename_map):
    if not rename_map or not code_str:
        return code_str
    try:
        tokens = list(tokenize.tokenize(io.BytesIO(code_str.encode('utf-8')).readline))
        new_tokens = []
        for tok_type, tok_val, start, end, line in tokens:
            if tok_type == tokenize.NAME and tok_val in rename_map:
                new_tokens.append((tok_type, rename_map[tok_val], start, end, line))
            else:
                new_tokens.append((tok_type, tok_val, start, end, line))
        res = tokenize.untokenize(new_tokens)
        if isinstance(res, bytes):
            res = res.decode('utf-8')
        return res
    except Exception:
        return code_str

def extract_function_body(code_str, target_name=None):
    try:
        tree = ast.parse(code_str)
    except Exception:
        return code_str, "", []
        
    target_fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if target_name and node.name == target_name:
                target_fn = node
                break
            elif not target_fn and node.name != '__init__':
                target_fn = node
                
    if not target_fn:
        return code_str, "", []
        
    fn_name = target_fn.name
    arg_names = [a.arg for a in target_fn.args.args]
    
    body_nodes = list(target_fn.body)
    if body_nodes and isinstance(body_nodes[0], ast.Expr):
        val = body_nodes[0].value
        if isinstance(val, ast.Constant) and isinstance(val.value, str):
            body_nodes = body_nodes[1:]
        elif hasattr(ast, 'Str') and isinstance(val, getattr(ast, 'Str')):
            body_nodes = body_nodes[1:]
            
    if not body_nodes:
        return "pass", fn_name, arg_names
        
    lines = code_str.splitlines()
    first_line_idx = body_nodes[0].lineno - 1
    last_line_idx = body_nodes[-1].end_lineno if hasattr(body_nodes[-1], 'end_lineno') and body_nodes[-1].end_lineno else len(lines)
    
    body_lines = lines[first_line_idx:last_line_idx]
    
    min_indent = None
    for l in body_lines:
        if l.strip():
            ind = len(l) - len(l.lstrip())
            if min_indent is None or ind < min_indent:
                min_indent = ind
                
    if min_indent is not None and min_indent > 0:
        dedented = [l[min_indent:] if len(l) >= min_indent else l for l in body_lines]
    else:
        dedented = body_lines
        
    return "\n".join(dedented), fn_name, arg_names

def build_role_based_attribute_map(ref_item, parsed_nb):
    mapping = {}
    ref_init = ref_item.get('init_attrs', [])
    unseen_init = parsed_nb.get('init_attrs', [])
    
    # 1. State / Tie-breaker integer
    state_synonyms = ['current_phase', 'last_action', 'last_wing', 'last_bin', 'active_bin']
    unseen_st = next((a for a in unseen_init if a in state_synonyms), None)
    if unseen_st:
        for s in state_synonyms:
            mapping[s] = unseen_st
            
    # 2. 1D Estimator array
    est_synonyms = ['service_estimate', 'service_est', 'wait_estimate', 'efficiency_estimate', 'clearance_est']
    unseen_es = next((a for a in unseen_init if a in est_synonyms), None)
    if unseen_es:
        for e in est_synonyms:
            mapping[e] = unseen_es
            
    # 3. Learning rate scalar
    rate_synonyms = ['alpha', 'lr', 'learning_rate', 'step_size']
    unseen_rate = next((a for a in unseen_init if a in rate_synonyms), None)
    if unseen_rate:
        for r in rate_synonyms:
            if r != unseen_rate:
                mapping[r] = unseen_rate
                
    # 4. General attribute synonyms check across all unseen code and init_attrs
    all_unseen_code = parsed_nb.get('all_code', '')
    for ref_attr, syn_list in ATTR_SYNONYMS.items():
        if ref_attr not in mapping:
            for syn in syn_list:
                if syn in unseen_init or re.search(r'\b(?:self|manager)\.' + re.escape(syn) + r'\b', all_unseen_code):
                    mapping[ref_attr] = syn
                    break

    # 5. Class name replacement
    if parsed_nb.get('agent_class_name'):
        mapping['TrafficLightAgent'] = parsed_nb['agent_class_name']
        mapping['ElevatorControllerAgent'] = parsed_nb['agent_class_name']
        mapping['QLearningAgent'] = parsed_nb['agent_class_name']
        
    return mapping

def generate_adapted_cat09_code(parsed_nb):
    agent_class = parsed_nb.get('agent_class_name') or "WarehouseRobotAgent"
    unseen_init = parsed_nb.get('init_attrs', [])
    
    tie_attr = next((a for a in unseen_init if a in ['last_bin', 'last_wing', 'active_bin', 'current_phase', 'last_action']), 'last_bin')
    est_attr = next((a for a in unseen_init if a in ['efficiency_estimate', 'clearance_est', 'service_estimate', 'wait_estimate']), 'efficiency_estimate')
    rate_attr = next((a for a in unseen_init if a in ['alpha', 'lr', 'learning_rate']), 'alpha')
    
    t1 = f"""def choose_action(self, state):
    p_metric, s_metric = state[0], state[1]
    w = getattr(self, 'priority_weight', 1.0)
    p_urgency = w * p_metric
    s_urgency = s_metric
    if p_urgency > s_urgency:
        action = 0
    elif s_urgency > p_urgency:
        action = 1
    else:
        action = self.{tie_attr}
    self.{tie_attr} = action
    return action"""

    t2 = f"""def update_agent(self, state, action, reward, next_state, done):
    p_metric, s_metric = state[0], state[1]
    next_p, next_s = next_state[0], next_state[1]
    w = getattr(self, 'priority_weight', 1.0)
    total_before = w * p_metric + s_metric
    total_after = w * next_p + next_s
    reduction = max(0, total_before - total_after)
    current_estimate = self.{est_attr}[action]
    self.{est_attr}[action] = current_estimate + self.{rate_attr} * (reduction - current_estimate)

{agent_class}.update_agent = update_agent"""

    return t1, t2

def normalize_cell_indentation(cell_src):
    """
    Detects and normalizes unintended module-level indentation on functions.
    If 'def' is indented without being inside a 'class', dedents 'def' to column 0
    and uniformly formats its inner statements to 4 spaces.
    """
    lines = cell_src.splitlines()
    def_idx = -1
    def_indent = 0
    for idx, l in enumerate(lines):
        if l.strip().startswith('def '):
            def_idx = idx
            m = re.match(r'^[ \t]+', l)
            def_indent = len(m.group(0)) if m else 0
            break
            
    if def_idx != -1 and "class " not in cell_src and def_indent > 0:
        norm_lines = []
        for idx, l in enumerate(lines):
            if idx < def_idx:
                norm_lines.append(l)
            elif idx == def_idx:
                norm_lines.append(l.strip())
            else:
                norm_lines.append(("    " + l.strip()) if l.strip() else "")
        return "\n".join(norm_lines)
    return cell_src

def inject_solution_into_cell(cell_src, body_code, agent_class=None):
    """
    Safely injects code into target cell with exact indentation preservation,
    fixing any module-level unexpected indentation and validating with ast.parse.
    """
    cell_src = normalize_cell_indentation(cell_src)
    has_class = "class " in cell_src
    base_indent = "        " if has_class else "    "
    
    marker_patterns = [
        r'([ \t]*###\s*>>>\s*YOUR CODE HERE\s*<<<[^\n]*\n)(.*?)(\n[ \t]*###\s*===+)',
        r'([ \t]*#\s*YOUR CODE HERE[^\n]*\n)(.*?)(\n[ \t]*#\s*===+)'
    ]
    
    for pat in marker_patterns:
        match = re.search(pat, cell_src, flags=re.DOTALL)
        if match:
            dedented = textwrap.dedent(body_code).strip('\n')
            indented = "\n".join((base_indent + l) if l.strip() else "" for l in dedented.splitlines())
            
            new_cell = cell_src[:match.start(2)] + indented + cell_src[match.end(2):]
            
            # Ensure class attachment if standalone update_agent
            if agent_class and not has_class and 'def update_agent' in new_cell:
                att = f"{agent_class}.update_agent = update_agent"
                if att not in new_cell:
                    new_cell = new_cell.rstrip() + f"\n\n{att}\n"
                    
            try:
                ast.parse(new_cell)
                return new_cell
            except SyntaxError as e:
                for alt in [base_indent, "    ", "        "]:
                    alt_indented = "\n".join((alt + l) if l.strip() else "" for l in dedented.splitlines())
                    cand = cell_src[:match.start(2)] + alt_indented + cell_src[match.end(2):]
                    if agent_class and not has_class and 'def update_agent' in cand:
                        att = f"{agent_class}.update_agent = update_agent"
                        if att not in cand:
                            cand = cand.rstrip() + f"\n\n{att}\n"
                    try:
                        ast.parse(cand)
                        return cand
                    except SyntaxError:
                        pass
                print(f"[Warning] Indentation validation warning: {e}")
                return cell_src
                
    return body_code

# ----------------------------------------------------------------------
# 5. Export Notebook Draft
# ----------------------------------------------------------------------

def export_draft(parsed_nb, adapted_task_results):
    raw_path = parsed_nb['path']
    dir_name = os.path.dirname(raw_path)
    base_name = os.path.basename(raw_path).replace('.ipynb', '')
    draft_path = os.path.join(dir_name, f"{base_name}_draft.ipynb")
    
    with open(raw_path, 'r', encoding='utf-8') as f:
        nb_copy = json.load(f)
        
    for res in adapted_task_results:
        c_idx = res['cell_index']
        if c_idx < len(nb_copy['cells']):
            nb_copy['cells'][c_idx]['source'] = [l + "\n" for l in res['injected_code'].splitlines()]
            
    with open(draft_path, 'w', encoding='utf-8') as f:
        json.dump(nb_copy, f, indent=1)
        
    return draft_path

# ----------------------------------------------------------------------
# 6. Formatting & Display
# ----------------------------------------------------------------------

def print_banner(text):
    print("\n" + "=" * 70)
    print(f" {text}")
    print("=" * 70)

def display_solve_results(parsed_nb, matches, args):
    if not matches:
        print_banner("NO COMPATIBLE ALGORITHM PATTERN MATCH FOUND")
        print(f"Target notebook: {parsed_nb['path']}")
        print("Please check notebook environment parameters and task definitions.")
        return
        
    top_score, top_item = matches[0]
    conf_pct = min(99.9, top_score * 100)
    
    print_banner(f"MATCH ANALYSIS (Confidence: {conf_pct:.1f}%)")
    print(f"Target Notebook: {parsed_nb['path']}")
    print(f"Pattern ID:      {top_item['id']} ({top_item['title']})")
    print(f"Family:          {top_item['category']}")
    print(f"Q-Table Required:{top_item.get('requires_q_table')}")
    
    print("\n--- TOP MATCHES ---")
    print(f"{'Rank':<5} {'Score':<8} {'Pattern ID':<26} {'Family':<32}")
    print("-" * 72)
    for i, (sc, item) in enumerate(matches[:3]):
        print(f"#{i+1:<4} {sc*100:>5.1f}%  {item['id']:<26} {item['category']:<32}")
        
    # Print extracted formulas from specification
    spec_formulas = parsed_nb.get('spec_formulas', [])
    if spec_formulas:
        print("\n--- SPECIFICATION FORMULAS & RULES EXTRACTED ---")
        for f in spec_formulas[:6]:
            print(f"  * {f}")
            
    ref_tasks = top_item.get('tasks', [])
    unseen_tasks = parsed_nb.get('detected_tasks', [])
    all_unseen_code = parsed_nb.get('all_code', '')
    agent_class = parsed_nb.get('agent_class_name')
    
    role_map = build_role_based_attribute_map(top_item, parsed_nb)
    
    is_cat09 = "Category09" in top_item['id'] or "Traffic Signal" in top_item['category']
    if is_cat09 and agent_class:
        cat09_t1, cat09_t2 = generate_adapted_cat09_code(parsed_nb)
    else:
        cat09_t1, cat09_t2 = None, None
        
    adapted_results = []
    print("\n--- DYNAMIC TASK MAPPING & ADAPTATION ---")
    
    for u_idx, u_task in enumerate(unseen_tasks):
        m_ref_task = None
        if u_task.get('func_name'):
            for rt in ref_tasks:
                if rt.get('func_name') == u_task.get('func_name'):
                    m_ref_task = rt
                    break
        if not m_ref_task and u_idx < len(ref_tasks):
            m_ref_task = ref_tasks[u_idx]
            
        if not m_ref_task:
            continue
            
        sym_map = dict(role_map)
        
        # Positional parameter alignment
        ref_params = m_ref_task.get('params', [])
        unseen_params = u_task.get('params', [])
        r_non_self = [p for p in ref_params if p not in ('self', 'manager')]
        u_non_self = [p for p in unseen_params if p not in ('self', 'manager')]
        for r_p, u_p in zip(r_non_self, u_non_self):
            if r_p != u_p:
                sym_map[r_p] = u_p
                
        # Check constants
        for ref_c, syn_list in CONST_SYNONYMS.items():
            for syn in syn_list:
                if re.search(r'\b' + re.escape(syn) + r'\b', all_unseen_code):
                    sym_map[ref_c] = syn
                    break
                    
        target_fn = m_ref_task.get('func_name') or u_task.get('func_name')
        
        if is_cat09 and cat09_t1 and u_idx == 0:
            sol_source = cat09_t1
            body_to_inject, _, _ = extract_function_body(cat09_t1)
        elif is_cat09 and cat09_t2 and u_idx == 1:
            sol_source = cat09_t2
            body_to_inject, _, _ = extract_function_body(cat09_t2)
        else:
            sol_source = safe_rename_tokens(m_ref_task['solution'], sym_map)
            body_raw, _, _ = extract_function_body(m_ref_task['solution'], target_fn)
            body_to_inject = safe_rename_tokens(body_raw, sym_map)
            
        injected = inject_solution_into_cell(
            u_task['source'],
            body_to_inject,
            agent_class=agent_class
        )
        
        adapted_results.append({
            'task_num': u_idx + 1,
            'cell_index': u_task['cell_index'],
            'ref_func': m_ref_task.get('func_name'),
            'unseen_func': u_task.get('func_name'),
            'symbol_map': sym_map,
            'adapted_code': sol_source,
            'injected_code': injected
        })
        
        print(f"  * Task {u_idx+1} (Cell {u_task['cell_index']}): `{m_ref_task.get('func_name')}` -> `{u_task.get('func_name')}`")
        if sym_map:
            for k, v in sym_map.items():
                print(f"      Mapped: `{k}` ===> `{v}`")
                
    if args.hints:
        print_banner("LOGIC HINTS")
        hints = top_item.get('step_by_step_hints', {})
        for k in sorted(hints.keys()):
            print(f"\n[{k.upper()}]")
            for h in hints[k]:
                print(f"  {h}")
        print("\n[REQUIRED APIS]")
        for mod, apis in top_item.get('api_used', {}).items():
            if apis:
                print(f"  {mod}: {', '.join(apis)}")
    else:
        print_banner("CANDIDATE IMPLEMENTATION")
        for res in adapted_results:
            print(f"\n# >>> [TASK {res['task_num']}] IMPLEMENTATION (Cell {res['cell_index']}) <<<")
            print(res['adapted_code'])
            
    show_qa = getattr(args, 'qa', False) or getattr(args, 'viva', False)
    if show_qa:
        print_banner("CONCEPTUAL Q&A")
        qa_list = top_item.get('viva_qa', []) or top_item.get('qa', [])
        for i, q in enumerate(qa_list, 1):
            print(f"\nQ{i}: {q['q']}")
            print(f"A{i}: {q['a']}")

    if getattr(args, 'export_copy', False) or getattr(args, 'export', False):
        draft_file = export_draft(parsed_nb, adapted_results)
        print_banner(f"DRAFT NOTEBOOK EXPORTED: {draft_file}")
        print(f"Preserved all locked cells intact.")
        print(f"Injected {len(adapted_results)} adapted task implementations into exact target cells.")

def query_dataset(query_str, dataset):
    tokens = tokenize_words(query_str)
    if not tokens:
        print("Please enter a valid query string.")
        return
        
    corpus = [tokenize_words(f"{item['title']} {item['category']} {item['problem_summary']} {' '.join(item['symbols']['constants'])} {item.get('task1_docstring', '')} {item.get('task2_docstring', '')}") for item in dataset]
    ranker = BM25Ranker(corpus)
    scores = ranker.score(tokens)
    
    ranked = sorted(zip(scores, dataset), key=lambda x: x[0], reverse=True)
    print_banner(f"QUERY RESULTS: '{query_str}'")
    
    count = 0
    for sc, item in ranked:
        if sc <= 0.01 and count > 0:
            break
        count += 1
        print(f"\n[{count}] {item['title']} ({item['category']}) - ID: {item['id']}")
        print(f"    Summary: {item['problem_summary'][:140]}...")
        print(f"    Constants: {', '.join(item['symbols']['constants'][:6])}")
        print(f"    APIs: {', '.join(item.get('api_used', {}).get('numpy', []))}")
        if count >= 3:
            break

# ----------------------------------------------------------------------
# 7. Interactive REPL
# ----------------------------------------------------------------------

def run_interactive(dataset):
    print_banner("INTERACTIVE REPL")
    print("Commands:")
    print("  solve <path>     - Analyze target notebook")
    print("  query <terms>    - Search patterns by keyword/API")
    print("  hints <id>       - Show logic hints for pattern ID")
    print("  qa <id>          - Show conceptual Q&A for pattern ID")
    print("  cheatsheet       - Display algorithm family summaries")
    print("  exit / quit      - Exit REPL\n")
    
    while True:
        try:
            cmd = input("app> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
            
        if not cmd:
            continue
        parts = cmd.split(maxsplit=1)
        action = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""
        
        if action in ['exit', 'quit', 'q']:
            print("Session ended.")
            break
        elif action == 'solve':
            if not arg or not os.path.exists(arg):
                print(f"File not found: '{arg}'")
                continue
            parsed = parse_notebook(arg)
            matches = rank_matches(parsed, dataset)
            class Args:
                hints = False
                qa = False
                viva = False
                export_copy = False
                export = False
            display_solve_results(parsed, matches, Args())
        elif action in ['query', 'search', 'find']:
            query_dataset(arg, dataset)
        elif action == 'hints':
            matched = [it for it in dataset if arg.lower() in it['id'].lower() or arg.lower() in it['title'].lower()]
            if matched:
                item = matched[0]
                print(f"\n--- Hints for {item['id']} ({item['title']}) ---")
                hints = item.get('step_by_step_hints', {})
                for k in sorted(hints.keys()):
                    print(f"[{k}]:")
                    for h in hints[k]: print(f"  {h}")
            else:
                print(f"No pattern found matching '{arg}'")
        elif action in ['qa', 'viva']:
            matched = [it for it in dataset if arg.lower() in it['id'].lower() or arg.lower() in it['title'].lower()]
            if matched:
                item = matched[0]
                print(f"\n--- Q&A: {item['id']} ({item['title']}) ---")
                qa_list = item.get('viva_qa', []) or item.get('qa', [])
                for i, q in enumerate(qa_list, 1):
                    print(f"Q{i}: {q['q']}\nA{i}: {q['a']}\n")
            else:
                print(f"No pattern found matching '{arg}'")
        elif action == 'cheatsheet':
            cats = defaultdict(list)
            for item in dataset:
                cats[item['category']].append(item['id'])
            print("\n--- CORE ALGORITHM FAMILIES ---")
            for cat, ids in cats.items():
                print(f"\n* {cat} ({len(ids)} patterns):")
                print(f"  {', '.join(ids)}")
        else:
            print("Unknown command. Available: solve, query, hints, qa, cheatsheet, exit")

# ----------------------------------------------------------------------
# 8. Main Entry Point
# ----------------------------------------------------------------------

def main():
    default_data = Path(__file__).resolve().parent / "data.json"
    if not default_data.exists():
        alt_kb = Path(__file__).resolve().parent / "kb.json"
        if alt_kb.exists():
            default_data = alt_kb

    parser = argparse.ArgumentParser(
        description="Offline Reinforcement Learning & Agent Environment Adapter."
    )
    parser.add_argument("notebook", nargs="?", help="Path to target Jupyter notebook (.ipynb)")
    parser.add_argument("--hints", action="store_true", help="Show step-by-step logic hints instead of code")
    parser.add_argument("--qa", "--viva", dest="qa", action="store_true", help="Print conceptual Q&A for matched pattern")
    parser.add_argument("--export-copy", "--export", dest="export_copy", action="store_true", help="Export candidate code into <notebook>_draft.ipynb")
    parser.add_argument("--query", type=str, help="Search patterns by keywords or API")
    parser.add_argument("--interactive", action="store_true", help="Start interactive REPL")
    parser.add_argument("--data", type=str, default=str(default_data), help="Path to data.json")
    
    args = parser.parse_args()
    
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: {data_path} not found.")
        sys.exit(1)
            
    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
        
    if args.interactive:
        run_interactive(dataset)
    elif args.query:
        query_dataset(args.query, dataset)
    elif args.notebook:
        if not os.path.exists(args.notebook):
            print(f"Error: Notebook not found at '{args.notebook}'")
            sys.exit(1)
        parsed = parse_notebook(args.notebook)
        matches = rank_matches(parsed, dataset)
        display_solve_results(parsed, matches, args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()

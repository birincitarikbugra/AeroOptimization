"""
v2 Model: Air Vehicle Project Scheduling with Uncertainty
=========================================================
Modules:
  1. CPM baseline calculation
  2. PERT analysis (asymmetric distributions)
  3. Monte Carlo Simulation with:
     - 3 risk-category distributions (triangular/beta-pert/lognormal)
     - Inter-activity correlation (Iman-Conover)
     - Rework loops (Bernoulli + duration penalty)
     - Bimodal supply chain delays
  4. RCPSP (Resource-Constrained) - 3 resource scenarios
  5. Risk metrics: CI, CRI, SSI
  6. Mitigation scenario comparison + stochastic dominance
  7. Tornado / sensitivity analysis

Author: v2 model for senior project, May 2026
"""
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist, norm, lognorm, triang
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
RANDOM_STATE = 42

# ============================================================
# 1. DATA LOADING
# ============================================================
def load_data(labeled_path='outputs/labeled_tasks.pkl'):
    df = pd.read_pickle(labeled_path)
    tasks = {}
    for _, row in df.iterrows():
        tasks[int(row['ID'])] = {
            'task_id': int(row['ID']),
            'task_name': row['Name'],
            'duration': float(row['Duration']),
            'predecessors': row['predecessors_clean'],  # [(pred_id, rel_type), ...]
            'risk_category': row['risk_category'],
            'phase': row['phase'],
            'subsystem': row['subsystem'],
            'resource_type': row['resource_type'],
            'wbs_level': int(row['Outline Level']),
        }
    return tasks

# ============================================================
# 2. NETWORK UTILITIES
# ============================================================
def topological_sort(tasks):
    """Kahn's algorithm for topological ordering"""
    in_degree = {tid: 0 for tid in tasks}
    successors = {tid: [] for tid in tasks}
    for tid, t in tasks.items():
        for pred_id, _ in t['predecessors']:
            if pred_id in tasks:
                in_degree[tid] += 1
                successors[pred_id].append(tid)
    queue = [tid for tid, d in in_degree.items() if d == 0]
    topo = []
    while queue:
        cur = queue.pop(0)
        topo.append(cur)
        for s in successors[cur]:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)
    if len(topo) != len(tasks):
        raise ValueError(f"Cycle detected. Sorted {len(topo)} of {len(tasks)}")
    return topo, successors

def find_terminals(tasks, successors):
    """Tasks with no successors (project end candidates)"""
    return [tid for tid in tasks if not successors[tid]]

# ============================================================
# 3. CPM CALCULATIONS
# ============================================================
def compute_cpm(tasks, durations=None):
    """
    Forward + backward pass.
    durations: optional dict tid->duration (else uses task duration)
    Returns: ES, EF, LS, LF, slack, project_duration, critical_path
    """
    topo, successors = topological_sort(tasks)
    if durations is None:
        durations = {tid: t['duration'] for tid, t in tasks.items()}
    
    ES, EF = {}, {}
    for tid in topo:
        t = tasks[tid]
        d = durations[tid]
        if not t['predecessors']:
            ES[tid] = 0.0
        else:
            reqs = []
            for pred_id, rel in t['predecessors']:
                if pred_id not in tasks:
                    continue
                if rel == 'FS':
                    reqs.append(EF[pred_id])
                elif rel == 'SS':
                    reqs.append(ES[pred_id])
                elif rel == 'FF':
                    reqs.append(EF[pred_id] - d)
                else:  # SF
                    reqs.append(ES[pred_id] - d)
            ES[tid] = max(reqs) if reqs else 0.0
        EF[tid] = ES[tid] + d
    
    project_duration = max(EF.values())
    
    # Backward pass
    LF, LS = {}, {}
    for tid in reversed(topo):
        t = tasks[tid]
        d = durations[tid]
        if not successors[tid]:
            LF[tid] = project_duration
        else:
            reqs = []
            for s in successors[tid]:
                # Look at successor's pred link from us
                for pred_id, rel in tasks[s]['predecessors']:
                    if pred_id != tid:
                        continue
                    if rel == 'FS':
                        reqs.append(LS[s])
                    elif rel == 'SS':
                        reqs.append(LS[s] + d)  # Our LF must allow successor SS
                    elif rel == 'FF':
                        reqs.append(LF[s])
                    else:  # SF
                        reqs.append(LF[s] + d)
            LF[tid] = min(reqs) if reqs else project_duration
        LS[tid] = LF[tid] - d
    
    slack = {tid: LS[tid] - ES[tid] for tid in tasks}
    critical_path = [tid for tid in tasks if abs(slack[tid]) < 0.01]
    
    return {
        'ES': ES, 'EF': EF, 'LS': LS, 'LF': LF, 'slack': slack,
        'project_duration': project_duration,
        'critical_path': critical_path,
        'topo': topo, 'successors': successors,
    }

# ============================================================
# 4. DURATION SAMPLING (BY RISK CATEGORY)
# ============================================================
def sample_low_risk(m, n):
    """Triangular ±10%, simetrik"""
    a, b = 0.90 * m, 1.10 * m
    return np.random.triangular(a, m, b, n)

def sample_med_risk(m, n):
    """Beta-PERT asimetrik, -15%/+25%"""
    a, b = 0.85 * m, 1.25 * m
    if abs(b - a) < 1e-9:
        return np.full(n, m)
    alpha_p = 1 + 4 * (m - a) / (b - a)
    beta_p = 1 + 4 * (b - m) / (b - a)
    return a + (b - a) * beta_dist.rvs(alpha_p, beta_p, size=n)

def sample_high_risk(m, n):
    """Lognormal, sağa çarpık. Medyan = m"""
    if m < 1e-6:
        return np.full(n, m)
    sigma_ln = 0.25  # ~25% log-std
    mu_ln = np.log(m)  # median = m
    return np.random.lognormal(mu_ln, sigma_ln, n)

def sample_duration(tasks, tid, n):
    cat = tasks[tid]['risk_category']
    m = tasks[tid]['duration']
    if cat == 'LOW':
        return sample_low_risk(m, n)
    elif cat == 'HIGH':
        return sample_high_risk(m, n)
    else:
        return sample_med_risk(m, n)

# ============================================================
# 5. CORRELATION (Iman-Conover, phase-based)
# ============================================================
def build_correlation_matrix(tasks, topo, phase_correlations):
    """
    phase_correlations: dict {(phase1, phase2): rho}
    Same phase: high correlation
    Related phases: medium
    Unrelated: low
    """
    n = len(topo)
    C = np.eye(n)
    tid_to_idx = {tid: i for i, tid in enumerate(topo)}
    for i, tid_i in enumerate(topo):
        for j, tid_j in enumerate(topo):
            if i == j:
                continue
            ph_i = tasks[tid_i]['phase']
            ph_j = tasks[tid_j]['phase']
            key = tuple(sorted([ph_i, ph_j]))
            rho = phase_correlations.get(key, phase_correlations.get(('default',), 0.0))
            C[i, j] = rho
    # Ensure positive definite
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals = np.maximum(eigvals, 1e-6)
    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
    # Renormalize to correlation matrix
    D = np.sqrt(np.diag(C))
    C = C / np.outer(D, D)
    return C

def apply_correlation_iman_conover(samples_matrix, C):
    """
    Iman-Conover: re-rank columns to induce target correlation
    samples_matrix: (n_sim, n_tasks)
    C: (n_tasks, n_tasks) target correlation
    """
    n_sim, n_tasks = samples_matrix.shape
    # Generate correlated normals
    L = np.linalg.cholesky(C + 1e-8 * np.eye(n_tasks))
    Z = np.random.standard_normal((n_sim, n_tasks))
    correlated_Z = Z @ L.T
    
    # Re-rank original samples by correlated_Z ranks
    output = np.zeros_like(samples_matrix)
    for j in range(n_tasks):
        ranks_target = np.argsort(np.argsort(correlated_Z[:, j]))
        sorted_orig = np.sort(samples_matrix[:, j])
        output[:, j] = sorted_orig[ranks_target]
    return output

# ============================================================
# 6. MONTE CARLO SIMULATION
# ============================================================
def monte_carlo_simulation(tasks, n_sim=10000, use_correlation=True, 
                           use_rework=False, use_supply_chain=False,
                           rework_nodes=None, supply_chain_items=None,
                           verbose=True):
    """
    Returns dict with all simulation outputs
    """
    topo, successors = topological_sort(tasks)
    n = len(topo)
    tid_to_idx = {tid: i for i, tid in enumerate(topo)}
    
    # 1. Sample base durations
    D = np.zeros((n_sim, n))
    for i, tid in enumerate(topo):
        D[:, i] = sample_duration(tasks, tid, n_sim)
    
    # 2. Apply correlation
    if use_correlation:
        phase_corr = {
            # Same phase pairs (high)
            ('Tooling', 'Tooling'): 0.5,
            ('Production', 'Production'): 0.5,
            ('Design', 'Design'): 0.55,
            ('ConceptDesign', 'ConceptDesign'): 0.55,
            ('SystemsDevelopment', 'SystemsDevelopment'): 0.5,
            ('GroundTest', 'GroundTest'): 0.6,
            ('FlightTest', 'FlightTest'): 0.6,
            ('Certification', 'Certification'): 0.6,
            ('Management', 'Management'): 0.3,
            # Related (medium)
            ('Design', 'SystemsDevelopment'): 0.35,
            ('ConceptDesign', 'Design'): 0.4,
            ('Production', 'Tooling'): 0.35,
            ('GroundTest', 'FlightTest'): 0.4,
            ('FlightTest', 'Certification'): 0.4,
            ('GroundTest', 'Certification'): 0.3,
            ('SystemsDevelopment', 'Production'): 0.25,
            # Default (low)
            ('default',): 0.1,
        }
        # Add symmetric entries
        full_corr = {}
        for k, v in phase_corr.items():
            if len(k) == 2:
                full_corr[tuple(sorted(k))] = v
            else:
                full_corr[k] = v
        
        C = build_correlation_matrix(tasks, topo, full_corr)
        D = apply_correlation_iman_conover(D, C)
    
    # 3. Apply rework
    if use_rework and rework_nodes:
        for rn in rework_nodes:
            tid = rn['task_id']
            if tid not in tid_to_idx:
                continue
            idx = tid_to_idx[tid]
            p = rn['probability']
            rework_factor_range = rn.get('factor_range', (0.3, 0.6))
            
            # Bernoulli for each sim
            rework_happens = np.random.random(n_sim) < p
            factor = np.random.uniform(*rework_factor_range, n_sim)
            additional = D[:, idx] * factor * rework_happens
            D[:, idx] += additional
            
            # Also propagate to downstream activities (basit yaklaşım - bağlı subsystem'e %50 ek)
            propagate_to = rn.get('propagate_to', [])
            for ptid in propagate_to:
                if ptid in tid_to_idx:
                    pidx = tid_to_idx[ptid]
                    D[:, pidx] += additional * 0.3  # downstream impact
    
    # 4. Apply supply chain delays (bimodal)
    if use_supply_chain and supply_chain_items:
        for sci in supply_chain_items:
            tid = sci['task_id']
            if tid not in tid_to_idx:
                continue
            idx = tid_to_idx[tid]
            p_delay = sci['p_delay']
            delay_mean = sci.get('delay_mean', 90)
            delay_std = sci.get('delay_std', 30)
            
            is_delayed = np.random.random(n_sim) < p_delay
            delay = np.maximum(0, np.random.normal(delay_mean, delay_std, n_sim))
            D[:, idx] += is_delayed * delay
    
    # 5. CPM forward pass on each simulation
    ES = np.zeros((n_sim, n))
    EF = np.zeros((n_sim, n))
    
    for i, tid in enumerate(topo):
        t = tasks[tid]
        if not t['predecessors']:
            ES[:, i] = 0
        else:
            reqs = []
            for pred_id, rel in t['predecessors']:
                if pred_id not in tid_to_idx:
                    continue
                pidx = tid_to_idx[pred_id]
                if rel == 'FS':
                    reqs.append(EF[:, pidx])
                elif rel == 'SS':
                    reqs.append(ES[:, pidx])
                elif rel == 'FF':
                    reqs.append(EF[:, pidx] - D[:, i])
                else:
                    reqs.append(ES[:, pidx] - D[:, i])
            if reqs:
                ES[:, i] = np.maximum.reduce(reqs)
            else:
                ES[:, i] = 0
        EF[:, i] = ES[:, i] + D[:, i]
    
    # Project finish = max EF across terminal tasks
    terminals = find_terminals(tasks, successors)
    term_idx = [tid_to_idx[t] for t in terminals]
    proj_finish = EF[:, term_idx].max(axis=1)
    
    # Trace critical path per simulation (for CI computation)
    if verbose:
        print(f"  Computing CI for {n_sim} simulations...")
    ci_counts = np.zeros(n, dtype=np.int64)
    
    # Hangi terminal task'in EF'i max'a eşitse, oradan geri trace
    for sim in range(n_sim):
        # Hangi terminal max'a verdi?
        finish_per_term = EF[sim, term_idx]
        winner_local = np.argmax(finish_per_term)
        cur_idx = term_idx[winner_local]
        
        seen = set()
        while cur_idx not in seen:
            seen.add(cur_idx)
            ci_counts[cur_idx] += 1
            tid = topo[cur_idx]
            t = tasks[tid]
            if not t['predecessors']:
                break
            # Find predecessor that determined our ES
            best_pred = None
            best_req = -np.inf
            for pred_id, rel in t['predecessors']:
                if pred_id not in tid_to_idx:
                    continue
                pidx = tid_to_idx[pred_id]
                if rel == 'FS':
                    req = EF[sim, pidx]
                elif rel == 'SS':
                    req = ES[sim, pidx]
                elif rel == 'FF':
                    req = EF[sim, pidx] - D[sim, cur_idx]
                else:
                    req = ES[sim, pidx] - D[sim, cur_idx]
                if req > best_req:
                    best_req = req
                    best_pred = pidx
            if best_pred is None or abs(best_req - ES[sim, cur_idx]) > 0.5:
                break
            cur_idx = best_pred
    
    # Risk metrics
    ci_pct = {topo[i]: 100.0 * ci_counts[i] / n_sim for i in range(n)}
    
    # CRI: correlation of duration with project finish
    sigma_T = np.std(proj_finish)
    cri = {}
    ssi = {}
    for i, tid in enumerate(topo):
        sigma_i = np.std(D[:, i])
        if sigma_i > 1e-6 and sigma_T > 1e-6:
            cri[tid] = np.corrcoef(D[:, i], proj_finish)[0, 1]
            ssi[tid] = (ci_pct[tid] / 100) * (sigma_i / sigma_T)
        else:
            cri[tid] = 0.0
            ssi[tid] = 0.0
    
    return {
        'project_finish': proj_finish,
        'D': D, 'ES': ES, 'EF': EF,
        'topo': topo, 'tid_to_idx': tid_to_idx,
        'ci_pct': ci_pct, 'cri': cri, 'ssi': ssi,
        'n_sim': n_sim,
        'terminals': terminals,
        'mean': float(np.mean(proj_finish)),
        'std': float(np.std(proj_finish)),
        'P50': float(np.percentile(proj_finish, 50)),
        'P80': float(np.percentile(proj_finish, 80)),
        'P90': float(np.percentile(proj_finish, 90)),
        'P95': float(np.percentile(proj_finish, 95)),
    }

# ============================================================
# 7. RCPSP (Resource-Constrained Project Scheduling)
# Serial SGS + LFT priority, deterministic for given durations
# ============================================================
def rcpsp_serial_sgs(tasks, durations, resource_capacity, resource_demand,
                     cpm_lft=None):
    """
    Schedule with resource constraints.
    durations: dict tid -> duration
    resource_capacity: dict resource_type -> capacity
    resource_demand: dict tid -> dict {resource_type: amount}
    cpm_lft: precomputed LFT (priority); if None, use task IDs
    """
    topo, successors = topological_sort(tasks)
    
    # Priority: lower LFT = higher priority (urgent first)
    if cpm_lft is None:
        cpm_result = compute_cpm(tasks, durations)
        cpm_lft = cpm_result['LF']
    priorities = sorted(topo, key=lambda t: cpm_lft.get(t, 0))
    
    start_time = {}
    finish_time = {}
    
    # Resource profile: resource_type -> list of (start, end, demand)
    resource_usage = defaultdict(list)
    
    for tid in priorities:
        t = tasks[tid]
        d = durations[tid]
        demand = resource_demand.get(tid, {})
        
        # Earliest precedence-feasible start
        if not t['predecessors']:
            es = 0
        else:
            reqs = []
            for pred_id, rel in t['predecessors']:
                if pred_id not in finish_time:
                    continue
                if rel == 'FS':
                    reqs.append(finish_time[pred_id])
                elif rel == 'SS':
                    reqs.append(start_time[pred_id])
                elif rel == 'FF':
                    reqs.append(finish_time[pred_id] - d)
                else:
                    reqs.append(start_time[pred_id] - d)
            es = max(reqs) if reqs else 0
        
        # Check resource feasibility - earliest start when all required resources have capacity
        candidate = es
        while True:
            feasible = True
            for r_type, r_amt in demand.items():
                if r_amt <= 0:
                    continue
                cap = resource_capacity.get(r_type, 999)
                # At time `candidate`, how much used?
                used = sum(amt for (s, e, amt) in resource_usage[r_type]
                          if s < candidate + d and e > candidate)
                if used + r_amt > cap:
                    feasible = False
                    # Jump to next time when something frees up
                    next_free = min((e for (s, e, amt) in resource_usage[r_type]
                                    if e > candidate), default=candidate + 1)
                    candidate = max(next_free, candidate + 0.1)
                    break
            if feasible:
                break
            if candidate > 1e6:  # safety
                break
        
        start_time[tid] = candidate
        finish_time[tid] = candidate + d
        for r_type, r_amt in demand.items():
            if r_amt > 0:
                resource_usage[r_type].append((candidate, candidate + d, r_amt))
    
    proj_dur = max(finish_time.values())
    return {
        'start': start_time, 'finish': finish_time,
        'project_duration': proj_dur,
    }

def rcpsp_monte_carlo(tasks, n_sim, resource_capacity, resource_demand,
                       use_correlation=True, verbose=True):
    """RCPSP combined with stochastic durations"""
    topo, successors = topological_sort(tasks)
    n = len(topo)
    tid_to_idx = {tid: i for i, tid in enumerate(topo)}
    
    # Sample correlated durations once for all sims
    D = np.zeros((n_sim, n))
    for i, tid in enumerate(topo):
        D[:, i] = sample_duration(tasks, tid, n_sim)
    
    if use_correlation:
        phase_corr = {
            ('Tooling', 'Tooling'): 0.5,
            ('Production', 'Production'): 0.5,
            ('Design', 'Design'): 0.55,
            ('SystemsDevelopment', 'SystemsDevelopment'): 0.5,
            ('GroundTest', 'GroundTest'): 0.6,
            ('FlightTest', 'FlightTest'): 0.6,
            ('Certification', 'Certification'): 0.6,
            ('Management', 'Management'): 0.3,
            ('Design', 'SystemsDevelopment'): 0.35,
            ('ConceptDesign', 'Design'): 0.4,
            ('Production', 'Tooling'): 0.35,
            ('GroundTest', 'FlightTest'): 0.4,
            ('FlightTest', 'Certification'): 0.4,
            ('default',): 0.1,
        }
        full_corr = {}
        for k, v in phase_corr.items():
            if len(k) == 2:
                full_corr[tuple(sorted(k))] = v
            else:
                full_corr[k] = v
        C = build_correlation_matrix(tasks, topo, full_corr)
        D = apply_correlation_iman_conover(D, C)
    
    # Pre-compute CPM-based LFT for priority
    cpm_base = compute_cpm(tasks)
    base_lft = cpm_base['LF']
    
    proj_durations = np.zeros(n_sim)
    
    # RCPSP is expensive; we run it but maybe sub-sample if too slow
    # For each sim, solve RCPSP
    for sim in range(n_sim):
        durations_dict = {topo[i]: D[sim, i] for i in range(n)}
        result = rcpsp_serial_sgs(tasks, durations_dict, resource_capacity,
                                  resource_demand, cpm_lft=base_lft)
        proj_durations[sim] = result['project_duration']
        if verbose and (sim + 1) % max(1, n_sim // 10) == 0:
            print(f"    RCPSP sim {sim+1}/{n_sim}")
    
    return {
        'project_finish': proj_durations,
        'mean': float(np.mean(proj_durations)),
        'std': float(np.std(proj_durations)),
        'P50': float(np.percentile(proj_durations, 50)),
        'P80': float(np.percentile(proj_durations, 80)),
        'P90': float(np.percentile(proj_durations, 90)),
        'P95': float(np.percentile(proj_durations, 95)),
    }


# ============================================================
# 8. RESOURCE DEMAND ASSIGNMENT
# ============================================================
def assign_resource_demand(tasks):
    """Each task demands its resource_type with amount=1 (could be parameterized)"""
    demand = {}
    for tid, t in tasks.items():
        if t['duration'] <= 1:  # milestones don't consume
            demand[tid] = {}
            continue
        demand[tid] = {t['resource_type']: 1}
    return demand

# Resource capacity scenarios
RESOURCE_SCENARIOS = {
    'Lean': {
        'SystemEng': 3, 'TestEng': 2, 'ManufactEng': 4,
        'CertEng': 2, 'DesignEng': 3, 'PMOps': 5,
    },
    'Baseline': {
        'SystemEng': 5, 'TestEng': 4, 'ManufactEng': 8,
        'CertEng': 3, 'DesignEng': 6, 'PMOps': 10,
    },
    'Optimized': {
        'SystemEng': 8, 'TestEng': 7, 'ManufactEng': 14,
        'CertEng': 5, 'DesignEng': 10, 'PMOps': 15,
    },
}

# ============================================================
# 9. REWORK NODES (typical aerospace)
# ============================================================
def get_rework_nodes(tasks):
    """Identify rework nodes by name pattern"""
    rework_keywords = {
        'Preliminary Design Review': {'prob': 0.15, 'factor': (0.2, 0.4)},
        'Critical Design Review': {'prob': 0.25, 'factor': (0.3, 0.6)},
        'Test Readiness Review': {'prob': 0.10, 'factor': (0.2, 0.4)},
        'System Verification Review': {'prob': 0.20, 'factor': (0.2, 0.5)},
        'Ground Vibration Test': {'prob': 0.20, 'factor': (0.3, 0.5)},
        '1st Air Vehicle Test': {'prob': 0.30, 'factor': (0.3, 0.6)},
        'Compliance Demonstration': {'prob': 0.20, 'factor': (0.3, 0.5)},
        'Certification Flight Test': {'prob': 0.25, 'factor': (0.3, 0.6)},
    }
    nodes = []
    for tid, t in tasks.items():
        name = t['task_name']
        for kw, params in rework_keywords.items():
            if kw.lower() in str(name).lower():
                nodes.append({
                    'task_id': tid,
                    'task_name': name,
                    'probability': params['prob'],
                    'factor_range': params['factor'],
                })
                break
    return nodes

# ============================================================
# 10. SUPPLY CHAIN ITEMS (long lead)
# ============================================================
def get_supply_chain_items(tasks):
    """Identify long-lead procurement-dependent activities"""
    keywords = {
        'engine': {'p_delay': 0.20, 'delay_mean': 90, 'delay_std': 30},
        'avionics computer': {'p_delay': 0.15, 'delay_mean': 60, 'delay_std': 20},
        'mission computer': {'p_delay': 0.15, 'delay_mean': 60, 'delay_std': 20},
        'tool manufacturing': {'p_delay': 0.10, 'delay_mean': 45, 'delay_std': 20},
        'sub assembly': {'p_delay': 0.12, 'delay_mean': 30, 'delay_std': 15},
        'electrical power': {'p_delay': 0.15, 'delay_mean': 60, 'delay_std': 25},
        'flight control system development': {'p_delay': 0.15, 'delay_mean': 75, 'delay_std': 25},
    }
    items = []
    for tid, t in tasks.items():
        name = str(t['task_name']).lower()
        for kw, params in keywords.items():
            if kw in name:
                items.append({
                    'task_id': tid,
                    'task_name': t['task_name'],
                    **params,
                })
                break
    return items

# ============================================================
# 11. MITIGATION SCENARIOS
# ============================================================
MITIGATION_SCENARIOS = {
    'S1_Baseline': {
        'description': 'No mitigation, baseline model',
        'resource': 'Baseline',
        'use_rework': True,
        'use_supply_chain': True,
        'duration_modifiers': {},
    },
    'S2_ResourceBoost': {
        'description': 'Increase resources to Optimized for top-SSI activities',
        'resource': 'Optimized',
        'use_rework': True,
        'use_supply_chain': True,
        'duration_modifiers': {},
    },
    'S3_ParallelTesting': {
        'description': 'Parallelize GVT and Flight prep activities (reduce by 15%)',
        'resource': 'Baseline',
        'use_rework': True,
        'use_supply_chain': True,
        'duration_modifiers': {'parallel_test': 0.85},  # multiply by 0.85
    },
    'S4_SupplierRisk': {
        'description': 'Dual sourcing reduces supply chain delays by 60%',
        'resource': 'Baseline',
        'use_rework': True,
        'use_supply_chain': True,
        'supply_chain_mitigation': 0.4,  # multiply p_delay by 0.4
    },
}


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("v2 MODEL EXECUTION")
    print("=" * 60)
    
    print("\n[1] Loading data...")
    tasks = load_data()
    print(f"    Loaded {len(tasks)} leaf tasks")
    
    print("\n[2] CPM Baseline...")
    cpm_result = compute_cpm(tasks)
    print(f"    Project Duration: {cpm_result['project_duration']:.2f} days")
    print(f"    Critical Path Length: {len(cpm_result['critical_path'])} tasks")
    
    print("\n[3] Monte Carlo Simulation (v2: asymmetric + correlated)...")
    rework_nodes = get_rework_nodes(tasks)
    supply_items = get_supply_chain_items(tasks)
    print(f"    Identified {len(rework_nodes)} rework nodes")
    print(f"    Identified {len(supply_items)} supply-chain dependent activities")
    
    mc_basic = monte_carlo_simulation(tasks, n_sim=10000, 
                                       use_correlation=True,
                                       use_rework=False, use_supply_chain=False,
                                       verbose=True)
    print(f"\n    BASIC MC (correlated, no rework/supply):")
    print(f"      Mean: {mc_basic['mean']:.1f} days")
    print(f"      P50: {mc_basic['P50']:.1f}, P80: {mc_basic['P80']:.1f}, P90: {mc_basic['P90']:.1f}")
    
    print("\n[4] Monte Carlo with Rework + Supply Chain...")
    mc_full = monte_carlo_simulation(tasks, n_sim=10000,
                                      use_correlation=True,
                                      use_rework=True, use_supply_chain=True,
                                      rework_nodes=rework_nodes,
                                      supply_chain_items=supply_items,
                                      verbose=True)
    print(f"\n    FULL MC (all uncertainty sources):")
    print(f"      Mean: {mc_full['mean']:.1f} days")
    print(f"      P50: {mc_full['P50']:.1f}, P80: {mc_full['P80']:.1f}, P90: {mc_full['P90']:.1f}")
    
    # Save results
    import pickle
    with open('outputs/mc_results.pkl', 'wb') as f:
        # Don't save huge arrays in pickle
        slim_basic = {k: v for k, v in mc_basic.items() if k not in ['D', 'ES', 'EF']}
        slim_full = {k: v for k, v in mc_full.items() if k not in ['D', 'ES', 'EF']}
        slim_full['project_finish_full'] = mc_full['project_finish']
        slim_basic['project_finish_full'] = mc_basic['project_finish']
        pickle.dump({
            'cpm': cpm_result,
            'mc_basic': slim_basic,
            'mc_full': slim_full,
            'rework_nodes': rework_nodes,
            'supply_items': supply_items,
        }, f)
    print("\n    ✓ Results saved to outputs/mc_results.pkl")
    
    # Top SSI / CI tables
    print("\n[5] Risk Metrics (Top 15)")
    ssi_sorted = sorted(mc_full['ssi'].items(), key=lambda x: -x[1])[:15]
    print("    Top 15 by SSI:")
    print(f"    {'Task':50} {'CI(%)':>7} {'CRI':>7} {'SSI':>7}")
    for tid, ssi_val in ssi_sorted:
        name = tasks[tid]['task_name'][:48]
        ci = mc_full['ci_pct'][tid]
        cri = mc_full['cri'][tid]
        print(f"    {name:50} {ci:>7.1f} {cri:>7.3f} {ssi_val:>7.3f}")
    
    print("\n" + "=" * 60)
    print("v2 MODEL EXECUTION COMPLETE")
    print("=" * 60)

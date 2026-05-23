mod domain_tests;
pub(crate) mod partial_policy;

use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::rc::Rc;
use std::time::{Duration, Instant};

use crate::domain_description::FONDProblem;
use crate::relaxation::{OutcomeDeterminizer, RelaxedComposition};
use crate::task_network::HTN;

pub use partial_policy::{SearchMode, TiebreakerKind};
use partial_policy::{
    compute_reach_sig, delta_nearest_f_value, make_key, HCache, MemoKey, NodeKind,
    PartialPolicyState, PolicyAssignment, PolicyLink, ReachNode, ReachSig, ReachView,
};

use super::progress;
use super::{
    ConnectionLabel, HeuristicType, NodeExpansion, PolicyNode, PolicyOutput, SearchGraphNode,
    SearchResult, SearchStats, StrongPolicy,
};

// ── Reach-graph construction ───────────────────────────────────────────────

/// Build Reach(π) for the root (empty) policy: the set of (tn, state) pairs reachable
/// from (init_tn, init_state).  All nodes with UC ≠ ∅ are Compound (open OR nodes);
/// nodes with UC = ∅ but applicable primitives are Primitive (auto-executed).
///
/// Returns `(reach, index_map, out_c)` where:
/// - `reach[0]` is always the initial node.
/// - `out_c` holds indices of unassigned compound nodes (Out_C(π)).
fn compute_reach(
    init_tn: Rc<HTN>,
    init_state: Rc<HashSet<u32>>,
    relaxed: &RelaxedComposition,
    bijection: &HashMap<u32, u32>,
    h_type: &HeuristicType,
    mode: SearchMode,
    h_cache: &mut HCache,
) -> (Vec<ReachNode>, HashMap<MemoKey, usize>, Vec<usize>) {
    // reach[i] starts as None and is filled when the node is dequeued.
    let mut reach: Vec<Option<ReachNode>> = Vec::new();
    let mut index_map: HashMap<MemoKey, usize> = HashMap::new();
    let mut out_c: Vec<usize> = Vec::new();

    // Allocate slot 0 for the initial node.
    let init_key = make_key(&init_tn, &init_state);
    index_map.insert(init_key.clone(), 0);
    reach.push(None);

    // BFS queue carries the canonical (tn, state) key so we can look up the
    // heuristic cache at the Compound-node branch without recomputing make_key.
    let mut queue: VecDeque<(Rc<HTN>, Rc<HashSet<u32>>, usize, MemoKey)> = VecDeque::new();
    queue.push_back((init_tn, init_state, 0, init_key));

    while let Some((tn, state, node_idx, node_key)) = queue.pop_front() {
        // ── Goal ──────────────────────────────────────────────────────────
        if tn.is_empty() {
            reach[node_idx] = Some(ReachNode {
                tn,
                state,
                kind: NodeKind::Goal,
                prob_upper: 1.0,
                cost_lower: 0.0,
                successors: vec![],
                landmarks: vec![],
            });
            continue;
        }

        let expansions = progress(tn.clone(), state.clone());

        // ── Dead end ──────────────────────────────────────────────────────
        if expansions.is_empty() {
            reach[node_idx] = Some(ReachNode {
                tn,
                state,
                kind: NodeKind::Dead,
                prob_upper: 0.0,
                cost_lower: 0.0,
                successors: vec![],
                landmarks: vec![],
            });
            continue;
        }

        // Algorithm 3 (Holler et al.): if UC ≠ ∅ (any decomposition exists), this is a
        // Compound (open OR) node — the policy will assign one connector (method or
        // primitive action) to it.  Only when UC = ∅ do we auto-execute an applicable
        // primitive (no policy choice needed, no heuristic call).
        //
        // This avoids the exponential O(M^N) Primitive-node chains that primitive-eager
        // creates for N concurrent non-det primitives with M outcomes each.
        let has_decomposition = expansions
            .iter()
            .any(|e| e.connection_label.is_decomposition());

        if has_decomposition {
            // ── UC ≠ ∅ — open OR node; policy assigns a connector (method or primitive) ──
            let (h, landmarks) = lookup_or_compute_h(
                &node_key, tn.as_ref(), state.as_ref(), relaxed, bijection, h_type, h_cache,
            );
            let (prob_upper, cost_lower) = match mode {
                SearchMode::MinCost => {
                    let c = if h == f32::INFINITY { f64::INFINITY } else { h as f64 };
                    let p = if h == f32::INFINITY { 0.0 } else { 1.0 };
                    (p, c)
                }
                SearchMode::MaxProb => (if h == f32::INFINITY { 0.0 } else { 1.0 }, 0.0),
            };
            reach[node_idx] = Some(ReachNode {
                tn,
                state,
                kind: NodeKind::Compound,
                prob_upper,
                cost_lower,
                successors: vec![],
                landmarks,
            });
            out_c.push(node_idx);
        } else {
            // ── UC = ∅ — only applicable primitives; auto-execute first one ──
            // progress() lists primitives before decompositions and in BTreeSet-ID order,
            // giving a fixed, consistent linearisation.
            if let Some(prim_expansion) = expansions.first() {
                let mut successors: Vec<(usize, f64)> = Vec::new();
                for (i, outcome_state) in prim_expansion.states.iter().enumerate() {
                    let p = prim_expansion
                        .outcome_probabilities
                        .get(i)
                        .copied()
                        .unwrap_or(1.0);
                    let succ_key = make_key(&prim_expansion.tn, outcome_state);
                    let succ_idx = get_or_insert(
                        succ_key,
                        &mut index_map,
                        &mut reach,
                        &mut queue,
                        prim_expansion.tn.clone(),
                        outcome_state.clone(),
                    );
                    successors.push((succ_idx, p));
                }
                reach[node_idx] = Some(ReachNode {
                    tn,
                    state,
                    kind: NodeKind::Primitive,
                    prob_upper: 1.0,
                    cost_lower: 0.0,
                    successors,
                    landmarks: vec![],
                });
            }
        }
    }

    let reach = reach.into_iter().map(|n| n.unwrap()).collect();
    (reach, index_map, out_c)
}

/// Return the reach index for `(tn, state)`, inserting a placeholder and
/// enqueueing if the pair has not been seen before.
fn get_or_insert(
    key: MemoKey,
    index_map: &mut HashMap<MemoKey, usize>,
    reach: &mut Vec<Option<ReachNode>>,
    queue: &mut VecDeque<(Rc<HTN>, Rc<HashSet<u32>>, usize, MemoKey)>,
    tn: Rc<HTN>,
    state: Rc<HashSet<u32>>,
) -> usize {
    if let Some(&idx) = index_map.get(&key) {
        idx // already allocated — back-edge or convergence
    } else {
        let idx = reach.len();
        index_map.insert(key.clone(), idx);
        reach.push(None); // placeholder filled when dequeued
        queue.push_back((tn, state, idx, key));
        idx
    }
}

/// Look up the heuristic value + landmarks for `(tn, state)` in the cache;
/// on a miss compute via `h_val_with_landmarks` and insert. The cache is
/// shared across all partial policies in a search run, so the same
/// `(tn, state)` Compound node never recomputes its heuristic.
fn lookup_or_compute_h(
    key: &MemoKey,
    tn: &HTN,
    state: &HashSet<u32>,
    relaxed: &RelaxedComposition,
    bijection: &HashMap<u32, u32>,
    h_type: &HeuristicType,
    h_cache: &mut HCache,
) -> (f32, crate::heuristics::LandmarkCuts) {
    if let Some(cached) = h_cache.get(key) {
        return cached.clone();
    }
    let computed =
        SearchGraphNode::h_val_with_landmarks(tn, state, relaxed, bijection, h_type);
    h_cache.insert(key.clone(), computed.clone());
    computed
}

// ── Incremental reach-graph extension ─────────────────────────────────────

/// Extend Reach(π) to Reach(π') by assigning `node_idx` (a Compound node in
/// `parent_reach`) to the method whose decomposed TN is `expansion_tn`.
///
/// All nodes already in `parent_reach` are copied unchanged; only the
/// newly reachable portion (from `connector` onwards) is BFS-explored.
/// Back-edges into already-known nodes are detected via `parent_index_map`.
///
/// `parent_out_c[0]` must equal `node_idx` (the node being assigned).
/// The returned `out_c` starts with `parent_out_c[1..]` and appends any
/// new unassigned compound nodes discovered in the extension.
fn compute_reach_incremental(
    parent_reach: &[ReachNode],
    parent_index_map: &HashMap<MemoKey, usize>,
    parent_out_c: &[usize],
    node_idx: usize,
    connector: &NodeExpansion,
    _node_state: Rc<HashSet<u32>>,
    relaxed: &RelaxedComposition,
    bijection: &HashMap<u32, u32>,
    h_type: &HeuristicType,
    mode: SearchMode,
    method_action_idx: Option<usize>,
    h_cache: &mut HCache,
) -> (ReachNode, Vec<ReachNode>, HashMap<MemoKey, usize>, Vec<usize>) {
    // No clone of the parent reach — only the modified node and the newly
    // discovered tail (extension) are materialised. Downstream consumers read
    // the virtual reach via ReachView::Incremental { parent, modified, extension }.
    let parent_len = parent_reach.len();
    let mut modified: ReachNode = parent_reach[node_idx].clone();
    let mut extension: Vec<Option<ReachNode>> = Vec::new();
    let mut index_map = HashMap::with_capacity(parent_index_map.len() + 16);
    index_map.clone_from(parent_index_map);
    // Carry over remaining unresolved compound nodes (skip node_idx at [0]).
    let mut out_c: Vec<usize> = parent_out_c[1..].to_vec();

    // Allocate / find the successor node(s) produced by the connector.
    // For decomposition connectors: one deterministic successor (connector.states[0] == node_state).
    // For primitive connectors: one successor per non-det outcome.
    let mut queue: VecDeque<(Rc<HTN>, Rc<HashSet<u32>>, usize, MemoKey)> = VecDeque::new();
    let mut successors: Vec<(usize, f64)> = Vec::new();
    for (i, outcome_state) in connector.states.iter().enumerate() {
        let p = connector.outcome_probabilities.get(i).copied().unwrap_or(1.0);
        let sk = make_key(&connector.tn, outcome_state);
        let si = get_or_insert_ext(
            sk,
            &mut index_map,
            &mut extension,
            &mut queue,
            connector.tn.clone(),
            outcome_state.clone(),
            parent_len,
        );
        successors.push((si, p));
    }
    // First successor index is used by the LM-cut warm-start check below.
    // For decomposition connectors (single outcome) this is the only successor.
    let first_succ_idx = successors.first().map(|(i, _)| *i).unwrap_or(0);

    // Flip the node from Compound to Assigned, recording all outcome successors.
    modified.kind = NodeKind::Assigned;
    modified.successors = successors;

    // BFS only over newly enqueued nodes — the index_map prevents
    // re-entering any node already present in parent_reach.
    // Every dequeued node has `n_idx >= parent_len` (back-edges into the
    // parent are handled inline by `get_or_insert_ext`), so it always lands in
    // `extension[n_idx - parent_len]`.
    let is_decomp = connector.connection_label.is_decomposition();
    while let Some((tn, state, n_idx, node_key)) = queue.pop_front() {
        let ext_idx = n_idx - parent_len;
        if tn.is_empty() {
            extension[ext_idx] = Some(ReachNode {
                tn,
                state,
                kind: NodeKind::Goal,
                prob_upper: 1.0,
                cost_lower: 0.0,
                successors: vec![],
                landmarks: vec![],
            });
            continue;
        }

        let expansions = progress(tn.clone(), state.clone());

        if expansions.is_empty() {
            extension[ext_idx] = Some(ReachNode {
                tn,
                state,
                kind: NodeKind::Dead,
                prob_upper: 0.0,
                cost_lower: 0.0,
                successors: vec![],
                landmarks: vec![],
            });
            continue;
        }

        // Algorithm 3: UC ≠ ∅ → Compound (open OR node); UC = ∅ → Primitive (auto-execute).
        let has_decomposition = expansions
            .iter()
            .any(|e| e.connection_label.is_decomposition());

        if has_decomposition {
            // UC ≠ ∅ — open OR node.
            // First consult the heuristic cache; on hit reuse, on miss either
            // warm-start (LM-cut on the direct compound successor of a
            // decomposition) or do a full h_val_with_landmarks computation.
            let (h, landmarks) = if let Some(cached) = h_cache.get(&node_key) {
                cached.clone()
            } else {
                let computed = if matches!(h_type, HeuristicType::HLMCut)
                    && is_decomp
                    && n_idx == first_succ_idx
                    && method_action_idx.is_some()
                {
                    SearchGraphNode::h_val_lmcut_incremental(
                        tn.as_ref(),
                        state.as_ref(),
                        relaxed,
                        bijection,
                        &parent_reach[node_idx].landmarks,
                        method_action_idx.unwrap(),
                    )
                } else {
                    SearchGraphNode::h_val_with_landmarks(
                        tn.as_ref(), state.as_ref(), relaxed, bijection, h_type,
                    )
                };
                h_cache.insert(node_key.clone(), computed.clone());
                computed
            };
            let (prob_upper, cost_lower) = match mode {
                SearchMode::MinCost => {
                    let c = if h == f32::INFINITY { f64::INFINITY } else { h as f64 };
                    let p = if h == f32::INFINITY { 0.0 } else { 1.0 };
                    (p, c)
                }
                SearchMode::MaxProb => (if h == f32::INFINITY { 0.0 } else { 1.0 }, 0.0),
            };
            extension[ext_idx] = Some(ReachNode {
                tn,
                state,
                kind: NodeKind::Compound,
                prob_upper,
                cost_lower,
                successors: vec![],
                landmarks,
            });
            out_c.push(n_idx);
        } else {
            // UC = ∅ — auto-execute first applicable primitive.
            if let Some(prim_expansion) = expansions.first() {
                let mut prim_successors: Vec<(usize, f64)> = Vec::new();
                for (i, outcome_state) in prim_expansion.states.iter().enumerate() {
                    let p = prim_expansion
                        .outcome_probabilities
                        .get(i)
                        .copied()
                        .unwrap_or(1.0);
                    let sk = make_key(&prim_expansion.tn, outcome_state);
                    let si = get_or_insert_ext(
                        sk,
                        &mut index_map,
                        &mut extension,
                        &mut queue,
                        prim_expansion.tn.clone(),
                        outcome_state.clone(),
                        parent_len,
                    );
                    prim_successors.push((si, p));
                }
                extension[ext_idx] = Some(ReachNode {
                    tn,
                    state,
                    kind: NodeKind::Primitive,
                    prob_upper: 1.0,
                    cost_lower: 0.0,
                    successors: prim_successors,
                    landmarks: vec![],
                });
            }
        }
    }

    let extension: Vec<ReachNode> = extension.into_iter().map(|n| n.unwrap()).collect();
    (modified, extension, index_map, out_c)
}

/// Like `get_or_insert`, but maintains a sparse `extension` Vec of `Option<ReachNode>`
/// indexed from `parent_len`. Existing parent-side hits (idx < parent_len) return
/// immediately without touching `extension`.
fn get_or_insert_ext(
    key: MemoKey,
    index_map: &mut HashMap<MemoKey, usize>,
    extension: &mut Vec<Option<ReachNode>>,
    queue: &mut VecDeque<(Rc<HTN>, Rc<HashSet<u32>>, usize, MemoKey)>,
    tn: Rc<HTN>,
    state: Rc<HashSet<u32>>,
    parent_len: usize,
) -> usize {
    if let Some(&idx) = index_map.get(&key) {
        return idx; // back-edge into parent reach or earlier extension slot
    }
    let idx = parent_len + extension.len();
    index_map.insert(key.clone(), idx);
    extension.push(None); // placeholder filled when dequeued
    queue.push_back((tn, state, idx, key));
    idx
}

// ── MinCost helpers ─────────────────────────────────────────────────────────

/// Admissible lower bound on total plan assignments for MinCost AND*.
/// Returns true iff no Dead node is reachable from reach[0].
/// A closed FOND policy with a reachable Dead node is not a valid strong plan.
fn is_reach_proper(view: &ReachView<'_>) -> bool {
    let n = view.len();
    let mut visited = vec![false; n];
    let mut stack = vec![0usize];
    while let Some(idx) = stack.pop() {
        if visited[idx] {
            continue;
        }
        visited[idx] = true;
        let node = view.get(idx);
        if matches!(node.kind, NodeKind::Dead) {
            return false;
        }
        for &(succ, _) in &node.successors {
            if !visited[succ] {
                stack.push(succ);
            }
        }
    }
    true
}

// ── Algorithm 2 — HTN-AND* ─────────────────────────────────────────────────

/// Max-probability AND* for probabilistic HTN domains.
pub fn run(
    problem: &FONDProblem,
    h_type: HeuristicType,
    tiebreaker: TiebreakerKind,
) -> (SearchResult, SearchStats) {
    run_internal(problem, h_type, tiebreaker, SearchMode::MaxProb)
}

/// Min-cost AND* for standard FOND (non-probabilistic) domains.
/// Returns a strong plan (success probability 1.0) minimising the number of
/// compound-node assignments (policy length).  Uses the classical heuristic
/// to guide search; `--prob` is not supported in this mode.
pub fn run_fond(
    problem: &FONDProblem,
    h_type: HeuristicType,
    tiebreaker: TiebreakerKind,
) -> (SearchResult, SearchStats) {
    run_internal(problem, h_type, tiebreaker, SearchMode::MinCost)
}

fn run_internal(
    problem: &FONDProblem,
    h_type: HeuristicType,
    tiebreaker: TiebreakerKind,
    mode: SearchMode,
) -> (SearchResult, SearchStats) {
    let start_time = Instant::now();
    let (outcome_det, bijection) = OutcomeDeterminizer::from_fond_problem(problem);
    let relaxed = RelaxedComposition::new(&outcome_det);

    // Per-search heuristic cache shared across all partial policies:
    // every (canonical TN, canonical state) Compound node hits this cache,
    // closing AND*'s biggest per-successor cost gap vs the single-graph AO*.
    let mut h_cache: HCache = HashMap::new();

    // Build the initial (empty) partial policy π_I.
    let (init_reach, _init_index_map, init_out_c) = compute_reach(
        Rc::new(problem.init_tn.clone()),
        Rc::new(problem.initial_state.clone()),
        &relaxed,
        &bijection,
        &h_type,
        mode,
        &mut h_cache,
    );
    let init_view = ReachView::Full(&init_reach);
    let f_value = match mode {
        SearchMode::MaxProb => PartialPolicyState::compute_f_by_vi(&init_view),
        SearchMode::MinCost => delta_nearest_f_value(&init_view, &init_out_c, 0),
    };
    let init_sig = compute_reach_sig(&init_view, f_value, 0);
    drop(init_view);

    // Root state: no parent, so base_reach is empty; full reach lives in extension.
    let mut next_id: u64 = 0;
    let pi_i = PartialPolicyState {
        base_reach: Rc::new(Vec::new()),
        modification: None,
        extension: init_reach,
        out_c: init_out_c,
        policy_tail: None,
        f_value,
        policy_size: 0,
        tiebreaker,
        insertion_order: next_id,
    };
    next_id += 1;

    let mut open: BinaryHeap<PartialPolicyState> = BinaryHeap::new();
    let mut seen: HashSet<ReachSig> = HashSet::new();
    seen.insert(init_sig);
    open.push(pi_i);

    let mut explored = 0u32;

    // Track the best closed (fully-resolved) policy found so far.
    // We continue searching until the open list's maximum f-value proves
    // the best-known closed policy is optimal (i.e. no open policy can
    // beat it).
    let mut best_closed_prob: f64 = -1.0;
    let mut best_closed_policy: Option<PartialPolicyState> = None;

    while let Some(pi) = open.pop() {
        // ── MaxProb: stop when open top can't beat best closed ─────────────
        if mode == SearchMode::MaxProb {
            if let Some(_) = &best_closed_policy {
                if pi.f_value <= best_closed_prob {
                    break;
                }
            }
        }

        explored += 1;

        if explored % 10_000 == 0 {
            match mode {
                SearchMode::MaxProb => eprintln!(
                    "[AND*] explored={} | open={} | best_prob={:.4} | best_closed={:.4} | {:.1}s",
                    explored,
                    open.len(),
                    pi.f_value,
                    if best_closed_prob < 0.0 { 0.0 } else { best_closed_prob },
                    start_time.elapsed().as_secs_f64()
                ),
                SearchMode::MinCost => eprintln!(
                    "[AND*] explored={} | open={} | est_cost={:.1} | found_plan={} | {:.1}s",
                    explored,
                    open.len(),
                    -pi.f_value,
                    best_closed_prob >= 1.0,
                    start_time.elapsed().as_secs_f64()
                ),
            }
        }

        // ── Closed policy found ────────────────────────────────────────────
        if pi.is_closed() {
            match mode {
                SearchMode::MaxProb => {
                    let success_prob = pi.f_value;
                    if success_prob > best_closed_prob {
                        best_closed_prob = success_prob;
                        best_closed_policy = Some(pi);
                        eprintln!(
                            "[AND*] new best closed: prob={:.6} | explored={} | {:.1}s",
                            best_closed_prob,
                            explored,
                            start_time.elapsed().as_secs_f64()
                        );
                        if best_closed_prob >= 1.0 - 1e-12 {
                            break;
                        }
                    }
                    continue;
                }
                SearchMode::MinCost => {
                    let reach = pi.reconstruct_reach();
                    if is_reach_proper(&ReachView::Full(&reach)) {
                        best_closed_prob = 1.0;
                        best_closed_policy = Some(pi);
                        eprintln!(
                            "[AND*] new best closed: prob={:.6} | explored={} | {:.1}s",
                            best_closed_prob,
                            explored,
                            start_time.elapsed().as_secs_f64()
                        );
                        break; // first proper closed policy is optimal
                    }
                    continue; // improper (trapped cycle) — keep searching
                }
            }
        }

        // ── Reconstruct this state's full reach (clone-on-pop) ─────────────
        let reach = pi.reconstruct_reach();
        let reach_rc = Rc::new(reach);

        let index_map: HashMap<MemoKey, usize> = reach_rc
            .iter()
            .enumerate()
            .map(|(i, node)| (make_key(node.tn.as_ref(), node.state.as_ref()), i))
            .collect();

        // ── Expand first unresolved compound node lazily ───────────────────
        let node_idx = pi.out_c[0];
        let node_tn = reach_rc[node_idx].tn.clone();
        let node_state = reach_rc[node_idx].state.clone();

        // Algorithm 3: try ALL connectors from progress() — both compound decompositions
        // and primitive executions.  Each is a candidate assignment for the OR node at
        // out_c[0].
        let expansions = progress(node_tn.clone(), node_state.clone());
        let total_connectors = expansions.len();
        let mut last_heartbeat = Instant::now();

        for (conn_i, expansion) in expansions.iter().enumerate() {
            if last_heartbeat.elapsed() >= Duration::from_secs(2) {
                eprintln!(
                    "[AND*] explored={} | expanding {}/{} | open={} | elapsed={:.1}s",
                    explored,
                    conn_i + 1,
                    total_connectors,
                    open.len(),
                    start_time.elapsed().as_secs_f64()
                );
                last_heartbeat = Instant::now();
            }

            // For LM-cut: find the classical action index so the incremental
            // warm-start can drop landmark cuts that contain it.
            let method_action_idx = if matches!(h_type, HeuristicType::HLMCut) {
                let name = match &expansion.connection_label {
                    ConnectionLabel::Decomposition(_, m) => m.as_str(),
                    ConnectionLabel::Execution(a, _) => a.as_str(),
                };
                relaxed.domain.actions.iter().position(|a| a.name == name)
            } else {
                None
            };

            let (modified_node, extension, _new_index_map, new_out_c) =
                compute_reach_incremental(
                    &reach_rc,
                    &index_map,
                    &pi.out_c,
                    node_idx,
                    expansion,
                    node_state.clone(),
                    &relaxed,
                    &bijection,
                    &h_type,
                    mode,
                    method_action_idx,
                    &mut h_cache,
                );
            // View the parent reach + modified node + extension as a single
            // virtual reach for the f-value / signature / properness checks,
            // without ever materialising a full Vec<ReachNode>.
            let new_view = ReachView::Incremental {
                parent: &reach_rc,
                modified_idx: node_idx,
                modified: &modified_node,
                extension: &extension,
            };
            // ── Early deadlock detection (MinCost only) ──────────────────────
            // If any Dead node is already reachable through the resolved reach,
            // no extension of Out_C can fix it — prune immediately.
            if mode == SearchMode::MinCost && !is_reach_proper(&new_view) {
                continue;
            }
            let new_f = match mode {
                SearchMode::MaxProb => PartialPolicyState::compute_f_by_vi(&new_view),
                SearchMode::MinCost => {
                    delta_nearest_f_value(&new_view, &new_out_c, pi.policy_size + 1)
                }
            };

            let should_prune = match mode {
                SearchMode::MaxProb => new_f < best_closed_prob.max(problem.rho),
                SearchMode::MinCost => false, // admissible heuristic — no threshold-based pruning needed
            };
            if should_prune {
                continue;
            }

            // Signature f-value for deduplication.
            // MaxProb: use the VI probability (already computed as new_f).
            // MinCost (FOND): use delta_nearest directly — the reach graph topology
            // (compound node states + TN structures in compute_reach_sig) already
            // fully distinguishes reach graphs. VI is binary-valued in FOND and adds
            // no extra discrimination, and the reference implementation (Messa &
            // Pereira) uses only structural state IDs, not VI values, for signatures.
            let sig_f = new_f;
            let sig = compute_reach_sig(&new_view, sig_f, pi.policy_size + 1);
            drop(new_view);
            if !seen.contains(&sig) {
                seen.insert(sig);

                let new_link = Rc::new(PolicyLink {
                    parent: pi.policy_tail.clone(),
                    assignment: PolicyAssignment {
                        tn_snapshot: node_tn.clone(),
                        state: node_state.clone(),
                        label: expansion.connection_label.clone(),
                    },
                });
                open.push(PartialPolicyState {
                    base_reach: Rc::clone(&reach_rc), // shared, O(1)
                    modification: Some((node_idx, modified_node)),
                    extension,
                    out_c: new_out_c,
                    policy_tail: Some(new_link),
                    f_value: new_f,
                    policy_size: pi.policy_size + 1,
                    tiebreaker,
                    insertion_order: next_id,
                });
                next_id += 1;
            }
        }
        // reach_rc is dropped here; its refcount falls to zero once all
        // children created above are either popped or pruned.
    }

    // Return the best closed policy found (if any and above threshold).
    if let Some(best_pi) = best_closed_policy {
        if best_closed_prob >= problem.rho {
            let mut transitions: Vec<(PolicyNode, PolicyOutput)> = vec![];
            let mut cur = best_pi.policy_tail.as_ref();
            while let Some(link) = cur {
                let a = &link.assignment;
                let state_strings: HashSet<String> = a
                    .state
                    .iter()
                    .map(|id| problem.facts.get_fact(*id).clone())
                    .collect();
                let (task, method) = match &a.label {
                    ConnectionLabel::Decomposition(t, m) => (t.clone(), m.clone()),
                    ConnectionLabel::Execution(action, _) => (action.clone(), String::new()),
                };
                transitions.push((
                    PolicyNode {
                        tn: a.tn_snapshot.clone(),
                        state: state_strings,
                    },
                    PolicyOutput { task, method },
                ));
                cur = link.parent.as_ref();
            }
            let makespan = transitions.len() as u16;
            let policy = StrongPolicy {
                transitions,
                makespan,
                success_probability: best_closed_prob,
            };
            return (
                SearchResult::Success(policy),
                make_stats(explored, start_time, Some(best_closed_prob), problem.rho),
            );
        }
    }

    (
        SearchResult::NoSolution,
        make_stats(explored, start_time, None, problem.rho),
    )
}

fn make_stats(explored: u32, start_time: Instant, prob: Option<f64>, rho: f64) -> SearchStats {
    SearchStats {
        max_depth: 0,
        search_nodes: explored,
        explored_nodes: explored,
        seach_time: start_time.elapsed(),
        success_probability: prob,
        rho_threshold: rho,
    }
}

// ── Unit tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::run;
    use super::TiebreakerKind;
    use crate::domain_description::{DomainTasks, FONDProblem, Facts};
    use crate::search::{HeuristicType, SearchResult};
    use crate::task_network::{CompoundTask, Method, PrimitiveAction, Task, HTN};
    use std::collections::{BTreeSet, HashMap, HashSet};
    use std::rc::Rc;

    /// Build a domain with two sequential ND actions:
    ///   a1 (prob p1 → adds f1, 1-p1 → nothing)  →
    ///   a2 (prob p2 → adds f2, 1-p2 → nothing, precond f1)  →
    ///   a3 (deterministic gate, precond f2)
    /// Max achievable success probability = p1 * p2.
    fn build_two_nd_problem(p1: f64, p2: f64, rho: f64) -> FONDProblem {
        let a1 = Task::Primitive(PrimitiveAction::new_with_probabilities(
            "a1".to_string(),
            1,
            HashSet::new(),
            vec![HashSet::from([0]), HashSet::new()], // add f1 | nothing
            vec![HashSet::new(), HashSet::new()],
            vec![p1, 1.0 - p1],
        ));
        let a2 = Task::Primitive(PrimitiveAction::new_with_probabilities(
            "a2".to_string(),
            1,
            HashSet::from([0]),                       // precond: f1
            vec![HashSet::from([1]), HashSet::new()], // add f2 | nothing
            vec![HashSet::new(), HashSet::new()],
            vec![p2, 1.0 - p2],
        ));
        let a3 = Task::Primitive(PrimitiveAction::new(
            "a3".to_string(),
            1,
            HashSet::from([1]), // precond: f2
            vec![HashSet::new()],
            vec![HashSet::new()],
        ));
        let domain = Rc::new(DomainTasks::new(vec![a1, a2, a3]));
        let tn = HTN::new(
            BTreeSet::from([1, 2, 3]),
            vec![(1, 2), (2, 3)],
            domain.clone(),
            HashMap::from([(1, 0), (2, 1), (3, 2)]),
        );
        let mut problem = FONDProblem {
            facts: Facts::new(vec!["f1".to_string(), "f2".to_string()]),
            tasks: domain,
            initial_state: HashSet::new(),
            init_tn: tn,
            rho,
        };
        problem.collapse_tn();
        problem
    }

    #[test]
    fn two_nd_actions_quarter_prob() {
        let problem = build_two_nd_problem(0.5, 0.5, 0.0);
        let (result, _) = run(&problem, HeuristicType::HAdd, TiebreakerKind::Combined);
        match result {
            SearchResult::Success(policy) => assert!(
                (policy.success_probability - 0.25).abs() < 1e-9,
                "Expected 0.25, got {}",
                policy.success_probability
            ),
            SearchResult::NoSolution => panic!("Expected solution with prob 0.25"),
        }
    }

    #[test]
    fn two_nd_actions_above_threshold_no_solution() {
        let problem = build_two_nd_problem(0.5, 0.5, 0.5);
        let (result, _) = run(&problem, HeuristicType::HAdd, TiebreakerKind::Combined);
        assert!(matches!(result, SearchResult::NoSolution));
    }

    #[test]
    fn two_nd_actions_exact_threshold_succeeds() {
        let problem = build_two_nd_problem(0.5, 0.5, 0.25);
        let (result, _) = run(&problem, HeuristicType::HAdd, TiebreakerKind::Combined);
        match result {
            SearchResult::Success(policy) => assert!(
                (policy.success_probability - 0.25).abs() < 1e-9,
                "Expected 0.25, got {}",
                policy.success_probability
            ),
            SearchResult::NoSolution => panic!("Expected success at exact threshold"),
        }
    }

    #[test]
    fn deterministic_single_action() {
        let a = Task::Primitive(PrimitiveAction::new(
            "a".to_string(),
            1,
            HashSet::new(),
            vec![HashSet::new()],
            vec![HashSet::new()],
        ));
        let domain = Rc::new(DomainTasks::new(vec![a]));
        let tn = HTN::new(
            BTreeSet::from([1]),
            vec![],
            domain.clone(),
            HashMap::from([(1, 0)]),
        );
        let mut problem = FONDProblem {
            facts: Facts::new(vec!["f0".to_string()]),
            tasks: domain,
            initial_state: HashSet::new(),
            init_tn: tn,
            rho: 1.0,
        };
        problem.collapse_tn();
        let (result, _) = run(&problem, HeuristicType::HAdd, TiebreakerKind::Combined);
        match result {
            SearchResult::Success(policy) => assert!(
                (policy.success_probability - 1.0).abs() < 1e-9,
                "Expected 1.0, got {}",
                policy.success_probability
            ),
            SearchResult::NoSolution => panic!("Expected success for deterministic action"),
        }
    }

    #[test]
    fn single_nd_action_point_seven() {
        let a1 = Task::Primitive(PrimitiveAction::new_with_probabilities(
            "a1".to_string(),
            1,
            HashSet::new(),
            vec![HashSet::from([0]), HashSet::new()],
            vec![HashSet::new(), HashSet::new()],
            vec![0.7, 0.3],
        ));
        let gate = Task::Primitive(PrimitiveAction::new(
            "gate".to_string(),
            1,
            HashSet::from([0]),
            vec![HashSet::new()],
            vec![HashSet::new()],
        ));
        let domain = Rc::new(DomainTasks::new(vec![a1, gate]));
        let tn = HTN::new(
            BTreeSet::from([1, 2]),
            vec![(1, 2)],
            domain.clone(),
            HashMap::from([(1, 0), (2, 1)]),
        );
        let mut problem = FONDProblem {
            facts: Facts::new(vec!["f0".to_string()]),
            tasks: domain,
            initial_state: HashSet::new(),
            init_tn: tn,
            rho: 0.0,
        };
        problem.collapse_tn();
        let (result, _) = run(&problem, HeuristicType::HAdd, TiebreakerKind::Combined);
        match result {
            SearchResult::Success(policy) => assert!(
                (policy.success_probability - 0.7).abs() < 1e-9,
                "Expected 0.7, got {}",
                policy.success_probability
            ),
            SearchResult::NoSolution => panic!("Expected success with prob 0.7"),
        }
    }

    #[test]
    fn three_nd_actions_eighth_prob() {
        let a1 = Task::Primitive(PrimitiveAction::new_with_probabilities(
            "a1".to_string(),
            1,
            HashSet::new(),
            vec![HashSet::from([0]), HashSet::new()],
            vec![HashSet::new(), HashSet::new()],
            vec![0.5, 0.5],
        ));
        let a2 = Task::Primitive(PrimitiveAction::new_with_probabilities(
            "a2".to_string(),
            1,
            HashSet::from([0]),
            vec![HashSet::from([1]), HashSet::new()],
            vec![HashSet::new(), HashSet::new()],
            vec![0.5, 0.5],
        ));
        let a3 = Task::Primitive(PrimitiveAction::new_with_probabilities(
            "a3".to_string(),
            1,
            HashSet::from([1]),
            vec![HashSet::from([2]), HashSet::new()],
            vec![HashSet::new(), HashSet::new()],
            vec![0.5, 0.5],
        ));
        let gate = Task::Primitive(PrimitiveAction::new(
            "gate".to_string(),
            1,
            HashSet::from([2]),
            vec![HashSet::new()],
            vec![HashSet::new()],
        ));
        let domain = Rc::new(DomainTasks::new(vec![a1, a2, a3, gate]));
        let tn = HTN::new(
            BTreeSet::from([1, 2, 3, 4]),
            vec![(1, 2), (2, 3), (3, 4)],
            domain.clone(),
            HashMap::from([(1, 0), (2, 1), (3, 2), (4, 3)]),
        );
        let mut problem = FONDProblem {
            facts: Facts::new(vec!["f0".to_string(), "f1".to_string(), "f2".to_string()]),
            tasks: domain,
            initial_state: HashSet::new(),
            init_tn: tn,
            rho: 0.0,
        };
        problem.collapse_tn();
        let (result, _) = run(&problem, HeuristicType::HAdd, TiebreakerKind::Combined);
        match result {
            SearchResult::Success(policy) => assert!(
                (policy.success_probability - 0.125).abs() < 1e-9,
                "Expected 0.125, got {}",
                policy.success_probability
            ),
            SearchResult::NoSolution => panic!("Expected success with prob 0.125"),
        }
    }

    #[test]
    fn three_nd_actions_above_threshold_no_solution() {
        let a1 = Task::Primitive(PrimitiveAction::new_with_probabilities(
            "a1".to_string(),
            1,
            HashSet::new(),
            vec![HashSet::from([0]), HashSet::new()],
            vec![HashSet::new(), HashSet::new()],
            vec![0.5, 0.5],
        ));
        let a2 = Task::Primitive(PrimitiveAction::new_with_probabilities(
            "a2".to_string(),
            1,
            HashSet::from([0]),
            vec![HashSet::from([1]), HashSet::new()],
            vec![HashSet::new(), HashSet::new()],
            vec![0.5, 0.5],
        ));
        let a3 = Task::Primitive(PrimitiveAction::new_with_probabilities(
            "a3".to_string(),
            1,
            HashSet::from([1]),
            vec![HashSet::from([2]), HashSet::new()],
            vec![HashSet::new(), HashSet::new()],
            vec![0.5, 0.5],
        ));
        let gate = Task::Primitive(PrimitiveAction::new(
            "gate".to_string(),
            1,
            HashSet::from([2]),
            vec![HashSet::new()],
            vec![HashSet::new()],
        ));
        let domain = Rc::new(DomainTasks::new(vec![a1, a2, a3, gate]));
        let tn = HTN::new(
            BTreeSet::from([1, 2, 3, 4]),
            vec![(1, 2), (2, 3), (3, 4)],
            domain.clone(),
            HashMap::from([(1, 0), (2, 1), (3, 2), (4, 3)]),
        );
        let mut problem = FONDProblem {
            facts: Facts::new(vec!["f0".to_string(), "f1".to_string(), "f2".to_string()]),
            tasks: domain,
            initial_state: HashSet::new(),
            init_tn: tn,
            rho: 0.2,
        };
        problem.collapse_tn();
        let (result, _) = run(&problem, HeuristicType::HAdd, TiebreakerKind::Combined);
        assert!(matches!(result, SearchResult::NoSolution));
    }

    #[test]
    fn asymmetric_two_nd_actions() {
        let problem = build_two_nd_problem(0.8, 0.6, 0.0);
        let (result, _) = run(&problem, HeuristicType::HAdd, TiebreakerKind::Combined);
        match result {
            SearchResult::Success(policy) => assert!(
                (policy.success_probability - 0.48).abs() < 1e-9,
                "Expected 0.48, got {}",
                policy.success_probability
            ),
            SearchResult::NoSolution => panic!("Expected solution with prob 0.48"),
        }
    }

    #[test]
    fn method_choice_selects_best() {
        let a_hi = Task::Primitive(PrimitiveAction::new_with_probabilities(
            "a_hi".to_string(),
            1,
            HashSet::new(),
            vec![HashSet::from([0]), HashSet::new()],
            vec![HashSet::new(), HashSet::new()],
            vec![0.8, 0.2],
        ));
        let a_lo = Task::Primitive(PrimitiveAction::new_with_probabilities(
            "a_lo".to_string(),
            1,
            HashSet::new(),
            vec![HashSet::from([0]), HashSet::new()],
            vec![HashSet::new(), HashSet::new()],
            vec![0.3, 0.7],
        ));
        let a_gate = Task::Primitive(PrimitiveAction::new(
            "a_gate".to_string(),
            1,
            HashSet::from([0]),
            vec![HashSet::new()],
            vec![HashSet::new()],
        ));
        let t = Task::Compound(CompoundTask {
            name: "t".to_string(),
            methods: vec![],
        });
        let domain = Rc::new(DomainTasks::new(vec![a_hi, a_lo, a_gate, t]));

        let m_hi = Method::new(
            "m_hi".to_string(),
            HTN::new(
                BTreeSet::from([1, 2]),
                vec![(1, 2)],
                domain.clone(),
                HashMap::from([(1, 0), (2, 2)]),
            ),
        );
        let m_lo = Method::new(
            "m_lo".to_string(),
            HTN::new(
                BTreeSet::from([1, 2]),
                vec![(1, 2)],
                domain.clone(),
                HashMap::from([(1, 1), (2, 2)]),
            ),
        );
        let domain = domain.add_methods(vec![(3, m_hi), (3, m_lo)]);

        let tn = HTN::new(
            BTreeSet::from([1]),
            vec![],
            domain.clone(),
            HashMap::from([(1, 3)]),
        );
        let mut problem = FONDProblem {
            facts: Facts::new(vec!["f_done".to_string()]),
            tasks: domain,
            initial_state: HashSet::new(),
            init_tn: tn,
            rho: 0.0,
        };
        problem.collapse_tn();

        let (result, _) = run(&problem, HeuristicType::HAdd, TiebreakerKind::Combined);
        match result {
            SearchResult::Success(policy) => assert!(
                (policy.success_probability - 0.8).abs() < 1e-9,
                "Expected 0.8 (best method), got {}",
                policy.success_probability
            ),
            SearchResult::NoSolution => panic!("Expected solution with prob 0.8"),
        }
    }

    #[test]
    fn method_choice_hprob_selects_best() {
        let a_hi = Task::Primitive(PrimitiveAction::new_with_probabilities(
            "a_hi".to_string(),
            1,
            HashSet::new(),
            vec![HashSet::from([0]), HashSet::new()],
            vec![HashSet::new(), HashSet::new()],
            vec![0.8, 0.2],
        ));
        let a_lo = Task::Primitive(PrimitiveAction::new_with_probabilities(
            "a_lo".to_string(),
            1,
            HashSet::new(),
            vec![HashSet::from([0]), HashSet::new()],
            vec![HashSet::new(), HashSet::new()],
            vec![0.3, 0.7],
        ));
        let a_gate = Task::Primitive(PrimitiveAction::new(
            "a_gate".to_string(),
            1,
            HashSet::from([0]),
            vec![HashSet::new()],
            vec![HashSet::new()],
        ));
        let t = Task::Compound(CompoundTask {
            name: "t".to_string(),
            methods: vec![],
        });
        let domain = Rc::new(DomainTasks::new(vec![a_hi, a_lo, a_gate, t]));

        let m_hi = Method::new(
            "m_hi".to_string(),
            HTN::new(
                BTreeSet::from([1, 2]),
                vec![(1, 2)],
                domain.clone(),
                HashMap::from([(1, 0), (2, 2)]),
            ),
        );
        let m_lo = Method::new(
            "m_lo".to_string(),
            HTN::new(
                BTreeSet::from([1, 2]),
                vec![(1, 2)],
                domain.clone(),
                HashMap::from([(1, 1), (2, 2)]),
            ),
        );
        let domain = domain.add_methods(vec![(3, m_hi), (3, m_lo)]);

        let tn = HTN::new(
            BTreeSet::from([1]),
            vec![],
            domain.clone(),
            HashMap::from([(1, 3)]),
        );
        let mut problem = FONDProblem {
            facts: Facts::new(vec!["f_done".to_string()]),
            tasks: domain,
            initial_state: HashSet::new(),
            init_tn: tn,
            rho: 0.0,
        };
        problem.collapse_tn();

        let (result, _) = run(&problem, HeuristicType::HAdd, TiebreakerKind::Combined);
        match result {
            SearchResult::Success(policy) => assert!(
                (policy.success_probability - 0.8).abs() < 1e-9,
                "Expected 0.8 (best method), got {}",
                policy.success_probability
            ),
            SearchResult::NoSolution => panic!("Expected solution with prob 0.8"),
        }
    }

    #[test]
    fn recursive_bounded_cycle_returns_partial_probability() {
        use crate::domain_description::read_json_domain;
        let mut problem = read_json_domain("test_domains/prob_always_fail_bounded_k01.json");
        problem.rho = 0.0;
        let (result, _) = run(&problem, HeuristicType::HAdd, TiebreakerKind::Combined);
        match result {
            SearchResult::Success(policy) => assert!(
                (policy.success_probability - 0.5).abs() < 1e-9,
                "Expected 0.5, got {}",
                policy.success_probability
            ),
            SearchResult::NoSolution => panic!("Expected partial-probability solution"),
        }
    }

    // ── delta_nearest_f_value unit tests ────────────────────────────────────

    #[test]
    fn delta_nearest_closed_policy_cost() {
        // Closed policy (out_c empty) with 3 assignments → f = -(3) = -3
        let f = super::delta_nearest_f_value(&super::ReachView::Full(&[]), &[], 3);
        assert!((f - (-3.0)).abs() < 1e-9, "expected -3.0, got {}", f);
    }

    #[test]
    fn delta_nearest_two_frontier_nodes() {
        // out_c = [0, 1], cost_lower = [2.0, 5.0], policy_size = 3
        // count = 5, h_vals = [5.0, 2.0] desc
        // delta = max(5+0, 2+1) = 5
        // min_out_c_h = 2.0, min_h_term = 1, lb = max(5, 5+1) = 6  → f = -6.0
        use super::{NodeKind, ReachNode};
        let dummy_tn = || Rc::new(HTN::new(
            std::collections::BTreeSet::new(),
            vec![],
            Rc::new(DomainTasks::new(vec![])),
            HashMap::new(),
        ));
        let reach = vec![
            ReachNode {
                tn: dummy_tn(),
                state: Rc::new(HashSet::new()),
                kind: NodeKind::Compound,
                prob_upper: 1.0,
                cost_lower: 2.0,
                successors: vec![],
                landmarks: vec![],
            },
            ReachNode {
                tn: dummy_tn(),
                state: Rc::new(HashSet::new()),
                kind: NodeKind::Compound,
                prob_upper: 1.0,
                cost_lower: 5.0,
                successors: vec![],
                landmarks: vec![],
            },
        ];
        let f = super::delta_nearest_f_value(&super::ReachView::Full(&reach), &[0, 1], 3);
        assert!((f - (-6.0)).abs() < 1e-9, "expected -6.0, got {}", f);
    }

    #[test]
    fn delta_nearest_infinity_cost_lower() {
        // INFINITY cost_lower is filtered from h_vals; count = policy_size + out_c = 2
        // No finite h → delta = NEG_INFINITY → lb = count = 2  → f = -2.0
        use super::{NodeKind, ReachNode};
        let reach = vec![ReachNode {
            tn: Rc::new(HTN::new(
                std::collections::BTreeSet::new(),
                vec![],
                Rc::new(DomainTasks::new(vec![])),
                HashMap::new(),
            )),
            state: Rc::new(HashSet::new()),
            kind: NodeKind::Compound,
            prob_upper: 0.0,
            cost_lower: f64::INFINITY,
            successors: vec![],
            landmarks: vec![],
        }];
        let f = super::delta_nearest_f_value(&super::ReachView::Full(&reach), &[0], 1);
        assert!((f - (-2.0)).abs() < 1e-9, "expected -2.0, got {}", f);
    }
}

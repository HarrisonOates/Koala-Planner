use crate::heuristics::LandmarkCuts;
use crate::search::progression::ConnectionLabel;
use crate::task_network::HTN;
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::rc::Rc;

// ── MemoKey types (shared with mod.rs) ────────────────────────────────────────

#[derive(Hash, PartialEq, Eq, Clone)]
pub struct TnKey {
    pub mappings: BTreeMap<u32, u32>,
    pub orderings: Vec<(u32, u32)>,
}

#[derive(Hash, PartialEq, Eq, Clone)]
pub struct StateKey(pub Vec<u32>);

pub type MemoKey = (TnKey, StateKey);

/// Cache of heuristic values keyed by `(canonical TN, canonical state)`.
/// Shared across all partial policies in a single search run so the same
/// `(tn, state)` Compound node never re-computes its heuristic — closing
/// the gap with AO*, which by construction computes each h-value once.
pub type HCache = HashMap<MemoKey, (f32, LandmarkCuts)>;

/// Read-only view of a reach graph built from a parent reach plus a single
/// modified node plus a tail of newly discovered nodes. Lets f-value /
/// signature / properness checks read each generated successor *without*
/// materialising the full `Vec<ReachNode>`.
///
/// `Full` is the simple variant used for the root reach (no parent).
pub enum ReachView<'a> {
    Full(&'a [ReachNode]),
    Incremental {
        parent: &'a [ReachNode],
        modified_idx: usize,
        modified: &'a ReachNode,
        extension: &'a [ReachNode],
    },
}

impl<'a> ReachView<'a> {
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            ReachView::Full(r) => r.len(),
            ReachView::Incremental {
                parent, extension, ..
            } => parent.len() + extension.len(),
        }
    }

    #[inline]
    pub fn get(&self, idx: usize) -> &'a ReachNode {
        match self {
            ReachView::Full(r) => &r[idx],
            ReachView::Incremental {
                parent,
                modified_idx,
                modified,
                extension,
            } => {
                if idx == *modified_idx {
                    modified
                } else if idx < parent.len() {
                    &parent[idx]
                } else {
                    &extension[idx - parent.len()]
                }
            }
        }
    }

    pub fn iter(&self) -> ReachViewIter<'a, '_> {
        ReachViewIter { view: self, idx: 0 }
    }
}

pub struct ReachViewIter<'a, 'b> {
    view: &'b ReachView<'a>,
    idx: usize,
}

impl<'a, 'b> Iterator for ReachViewIter<'a, 'b> {
    type Item = &'a ReachNode;
    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.view.len() {
            return None;
        }
        let n = self.view.get(self.idx);
        self.idx += 1;
        Some(n)
    }
}

pub fn make_key(tn: &HTN, state: &HashSet<u32>) -> MemoKey {
    let mappings: BTreeMap<u32, u32> = tn.mappings.iter().map(|(&k, &v)| (k, v)).collect();
    let mut orderings = tn.get_orderings();
    orderings.sort();
    orderings.dedup();
    let mut sv: Vec<u32> = state.iter().copied().collect();
    sv.sort();
    (
        TnKey {
            mappings,
            orderings,
        },
        StateKey(sv),
    )
}

/// Controls whether AND* optimises for maximum success probability (MaxProb)
/// or minimum expected cost (MinCost, for standard FOND domains).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchMode {
    MaxProb,
    #[allow(dead_code)] // used by run_fond (Phase 3)
    MinCost,
}

/// Secondary ordering applied when two partial policies have equal f-value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TiebreakerKind {
    /// No secondary criterion — FIFO among equal-f policies.
    NoTiebreak,
    /// Prefer more assignments (deeper DFS-like dive).
    /// Analogous to greater-g tiebreaking in classical A*.
    PolicySize,
    /// Prefer fewer unresolved compound nodes — closer to a closed policy.
    ClosureFirst,
    /// Prefer more assignments first; break further ties by fewer unresolved nodes.
    Combined,
}

/// Classification of a node in the reach graph.
#[derive(Clone, Debug)]
pub enum NodeKind {
    /// Empty task network — goal achieved (V = 1.0).
    Goal,
    /// No applicable expansion — dead end (V = 0.0).
    Dead,
    /// Unassigned compound task node — still in Out_C(π).
    /// V is pinned at `prob_upper` (admissible upper bound).
    Compound,
    /// Compound task with an assigned method.
    /// V is computed via its single successor.
    Assigned,
    /// Primitive action step.
    /// V is computed via weighted sum over outcome successors.
    Primitive,
}

/// One node in the reach graph Reach(π).
#[derive(Clone)]
pub struct ReachNode {
    pub tn: Rc<HTN>,
    pub state: Rc<HashSet<u32>>,
    pub kind: NodeKind,
    /// Admissible upper bound on success probability (MaxProb mode).
    /// Used to pin the V-value of Compound (out_c) nodes during VI.
    pub prob_upper: f64,
    /// Admissible lower bound on cost from this node (MinCost/FOND mode).
    /// Set to h_val at Compound nodes in MinCost mode; 0.0 elsewhere.
    #[allow(dead_code)] // used by fond_f_value (Phase 3)
    pub cost_lower: f64,
    /// Outgoing edges: (successor reach-index, edge probability).
    /// Empty for Goal, Dead, and Compound nodes.
    pub successors: Vec<(usize, f64)>,
    /// LM-cut landmark cuts discovered when computing h_val for this node.
    /// Non-empty only for Compound nodes when using the HLMCut heuristic.
    /// Used to warm-start LM-cut for child compound nodes (Pommerening & Helmert 2013).
    pub landmarks: LandmarkCuts,
}

/// A single method assignment recorded in the policy.
#[derive(Clone)]
pub struct PolicyAssignment {
    pub tn_snapshot: Rc<HTN>,
    pub state: Rc<HashSet<u32>>,
    pub label: ConnectionLabel,
}

/// Persistent (shared) linked list of policy assignments.
/// Each PartialPolicyState carries only its tail pointer; ancestors
/// are shared with sibling states via Rc, so creating a child is O(1).
pub struct PolicyLink {
    pub parent: Option<Rc<PolicyLink>>,
    pub assignment: PolicyAssignment,
}

/// One node in the AND* open list — a partial policy π.
///
/// Memory layout uses structural sharing to avoid cloning the full reach
/// graph for every child state:
///
/// - `base_reach`: the parent's full reach, shared via `Rc` among siblings.
///   For the root state this is an empty `Rc<Vec<_>>` (no parent).
/// - `modification`: the single node that changed (Compound → Assigned) when
///   we extended the parent.  `None` for the root.
/// - `extension`: the newly discovered reach nodes beyond `base_reach.len()`.
///
/// When a state is *popped*, `reconstruct_reach` clones the base and applies
/// the modification + extension in O(|base| + |extension|).  States that are
/// never popped (pruned by the seen-set) pay zero clone cost.
///
/// The index map is also not stored (expensive keys); it is rebuilt cheaply
/// from the reconstructed reach on each pop.
pub struct PartialPolicyState {
    /// Parent's full reach nodes, shared with sibling states (Rc, O(1) to store).
    pub base_reach: Rc<Vec<ReachNode>>,
    /// The one node flipped from Compound to Assigned when the parent was
    /// extended.  `None` for the root state.
    pub modification: Option<(usize, ReachNode)>,
    /// New reach nodes discovered beyond `base_reach.len()`.
    pub extension: Vec<ReachNode>,
    /// Indices (in the *reconstructed* reach) of unassigned compound nodes.
    pub out_c: Vec<usize>,
    /// Singly-linked list of assignments made so far.
    pub policy_tail: Option<Rc<PolicyLink>>,
    /// f(π) = V[0] from Bellman VI with out_c nodes pinned at prob_upper.
    pub f_value: f64,
    /// Number of compound-node assignments in policy_tail (i.e. |π|).
    pub policy_size: usize,
    /// Which secondary criterion to use when f-values tie.
    pub tiebreaker: TiebreakerKind,
    /// Monotonic insertion counter for deterministic heap ordering (FIFO
    /// among otherwise-equal policies).
    pub insertion_order: u64,
}

impl PartialPolicyState {
    pub fn is_closed(&self) -> bool {
        self.out_c.is_empty()
    }

    /// Reconstruct the full `Vec<ReachNode>` for this state by cloning the
    /// base, applying the modification, and appending the extension.
    /// Only called when the state is *popped* from the open list.
    pub fn reconstruct_reach(&self) -> Vec<ReachNode> {
        let mut reach = (*self.base_reach).clone();
        if let Some((idx, ref node)) = self.modification {
            reach[idx] = node.clone();
        }
        reach.extend(self.extension.iter().cloned());
        reach
    }

    /// Compute V[0] (value of the initial reach node) via pessimistic
    /// Bellman value iteration on the reach graph:
    ///
    ///   Goal      → V = 1.0  (fixed)
    ///   Dead      → V = 0.0  (fixed)
    ///   Compound  → V = prob_upper  (fixed — admissible upper bound)
    ///   Assigned, Primitive → V = Σ p_i · V[successor_i]  (iterate)
    ///
    /// Initialises non-fixed nodes at 0.0 (pessimistic) and iterates
    /// until |ΔV| < 1e-12, giving the MaxProb-correct value for both
    /// acyclic and cyclic reach graphs.
    pub fn compute_f_by_vi(view: &ReachView<'_>) -> f64 {
        let n = view.len();
        if n == 0 {
            return 1.0; // empty problem — trivially solved
        }

        // Initialise value vector.
        let mut v: Vec<f64> = (0..n)
            .map(|i| match view.get(i).kind {
                NodeKind::Goal => 1.0,
                NodeKind::Dead => 0.0,
                NodeKind::Compound => view.get(i).prob_upper, // pinned
                _ => 0.0,                                     // pessimistic init
            })
            .collect();

        // Bellman VI — converges for both acyclic and cyclic graphs
        // when initialised pessimistically (lower fixed point = MaxProb).
        const MAX_ITERS: usize = 100_000;
        for _ in 0..MAX_ITERS {
            let mut delta = 0.0_f64;
            for i in 0..n {
                let node_i = view.get(i);
                match node_i.kind {
                    // Fixed-value nodes — skip.
                    NodeKind::Goal | NodeKind::Dead | NodeKind::Compound => continue,
                    _ => {}
                }
                let new_v: f64 = node_i.successors.iter().map(|&(j, p)| p * v[j]).sum();
                let diff = (new_v - v[i]).abs();
                if diff > delta {
                    delta = diff;
                }
                v[i] = new_v;
            }
            if delta < 1e-12 {
                break;
            }
        }

        v[0]
    }
}

// ── Delta-nearest admissible lower bound (MinCost mode) ─────────────────────

/// Admissible lower-bound on total policy size for MinCost (FOND) mode.
///
/// Implements the "delta-nearest" estimate from Messa & Pereira (2403.19883):
/// collect cost_lower from all Assigned and Compound(Out_C) nodes, sort desc,
/// then delta = max(h[i] + i).  Returns -max(delta, count + max(0, min_out_c_h-1)).
pub fn delta_nearest_f_value(view: &ReachView<'_>, out_c: &[usize], policy_size: usize) -> f64 {
    if out_c.is_empty() {
        return -(policy_size as f64);
    }

    let mut h_vals: Vec<f64> = Vec::new();
    let mut min_out_c_h = f64::INFINITY;

    for node in view.iter() {
        match node.kind {
            NodeKind::Assigned => {
                if node.cost_lower != f64::INFINITY {
                    h_vals.push(node.cost_lower);
                }
            }
            NodeKind::Compound => {
                if node.cost_lower != f64::INFINITY {
                    h_vals.push(node.cost_lower);
                    if node.cost_lower < min_out_c_h {
                        min_out_c_h = node.cost_lower;
                    }
                }
            }
            _ => {}
        }
    }

    let count = (policy_size + out_c.len()) as f64;

    h_vals.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
    let delta = h_vals
        .iter()
        .enumerate()
        .map(|(i, &h)| h + i as f64)
        .fold(f64::NEG_INFINITY, f64::max);

    let min_h_term = if min_out_c_h == f64::INFINITY {
        0.0
    } else {
        (min_out_c_h - 1.0).max(0.0)
    };

    let lb = if delta == f64::NEG_INFINITY {
        count
    } else {
        delta.max(count + min_h_term)
    };
    -lb
}

// ── Ordering — max-heap on f_value ──────────────────────────────────────────

impl PartialEq for PartialPolicyState {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for PartialPolicyState {}

impl PartialOrd for PartialPolicyState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for PartialPolicyState {
    fn cmp(&self, other: &Self) -> Ordering {
        // Primary key: higher success-probability upper bound first.
        let f_ord = self
            .f_value
            .partial_cmp(&other.f_value)
            .unwrap_or(Ordering::Equal);
        if f_ord != Ordering::Equal {
            return f_ord;
        }
        // Tiebreaker (configurable):
        let tb = match self.tiebreaker {
            // No secondary criterion.
            TiebreakerKind::NoTiebreak => Ordering::Equal,
            // Prefer more assignments — deeper DFS-like dive.
            TiebreakerKind::PolicySize => self.policy_size.cmp(&other.policy_size),
            // Prefer fewer unresolved compound nodes — closer to a closed policy.
            // Reversed because fewer is better and BinaryHeap is a max-heap.
            TiebreakerKind::ClosureFirst => other.out_c.len().cmp(&self.out_c.len()),
            // More assignments first; break further ties by fewer unresolved.
            TiebreakerKind::Combined => self
                .policy_size
                .cmp(&other.policy_size)
                .then_with(|| other.out_c.len().cmp(&self.out_c.len())),
        };
        // Final tiebreaker: FIFO — earlier insertion (lower counter) gets
        // higher priority in the max-heap.
        tb.then_with(|| other.insertion_order.cmp(&self.insertion_order))
    }
}

// ── Deduplication signature ──────────────────────────────────────────────────

/// Hashed descriptor of a single unassigned compound reach node.
/// We hash the canonical `MemoKey` (already produced for `index_map` lookups)
/// down to a `u64` so the open-list dedup set stores ~8 bytes per compound
/// instead of three Vec allocations.
pub type CompoundSig = u64;

/// Full deduplication key for a PartialPolicyState.
///
/// Two states with the same policy_size, f_value, and identical unresolved
/// compound frontier are considered equivalent and the second is dropped.
/// Including `policy_size` is critical for recursive domains: the same
/// compound frontier may appear at different recursion depths, producing
/// different closed-policy probabilities (deeper = higher probability).
#[derive(Hash, PartialEq, Eq)]
pub struct ReachSig {
    pub policy_size: usize,
    pub f_value_bits: u64,
    pub compounds: Vec<CompoundSig>, // sorted hashes
}

/// Hash a single (TN, state) pair to a u64 using a deterministic SipHash.
/// The hash matches across isomorphic TNs that differ only in node-ID labels:
/// we rename node IDs to canonical 0..N order (same renaming the original
/// `canonicalize_tn` used) before feeding the data to SipHash. Storing this
/// as a `u64` (vs three Vec allocations per compound) shrinks the `seen`
/// dedup set by orders of magnitude.
#[inline]
fn hash_compound(tn: &HTN, state: &HashSet<u32>) -> u64 {
    use std::hash::{Hash, Hasher};

    // Canonical renaming of node IDs to 0..N (independent of grounding order).
    let mut old_nodes: Vec<u32> = tn.get_nodes().iter().copied().collect();
    old_nodes.sort_unstable();
    let mut renaming = std::collections::HashMap::with_capacity(old_nodes.len());
    for (new_id, &old_id) in old_nodes.iter().enumerate() {
        renaming.insert(old_id, new_id as u32);
    }

    let mut mappings: Vec<(u32, u32)> = tn
        .mappings
        .iter()
        .map(|(&old_node, &task_id)| (renaming[&old_node], task_id))
        .collect();
    mappings.sort_unstable();

    let mut orderings: Vec<(u32, u32)> = tn
        .get_orderings()
        .into_iter()
        .map(|(src, dst)| (renaming[&src], renaming[&dst]))
        .collect();
    orderings.sort_unstable();
    orderings.dedup();

    let mut facts: Vec<u32> = state.iter().copied().collect();
    facts.sort_unstable();

    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    (mappings.len() as u32).hash(&mut hasher);
    for kv in &mappings {
        kv.hash(&mut hasher);
    }
    (orderings.len() as u32).hash(&mut hasher);
    for kv in &orderings {
        kv.hash(&mut hasher);
    }
    (facts.len() as u32).hash(&mut hasher);
    for f in &facts {
        f.hash(&mut hasher);
    }
    hasher.finish()
}

/// Build a canonical signature of the reach graph for open-list deduplication.
/// Based on: policy_size + f_value + the set of unresolved compound nodes.
/// Called with the reach and f_value computed during expansion (not stored per state).
pub fn compute_reach_sig(view: &ReachView<'_>, f_value: f64, policy_size: usize) -> ReachSig {
    let mut compounds: Vec<CompoundSig> = view
        .iter()
        .filter(|n| matches!(n.kind, NodeKind::Compound))
        .map(|n| hash_compound(n.tn.as_ref(), n.state.as_ref()))
        .collect();
    compounds.sort_unstable();
    ReachSig {
        policy_size,
        f_value_bits: f_value.to_bits(),
        compounds,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain_description::DomainTasks;
    use crate::task_network::HTN;
    use std::collections::{BTreeSet, HashMap};

    fn dummy_node(kind: NodeKind, cost_lower: f64) -> ReachNode {
        let domain = Rc::new(DomainTasks::new(vec![]));
        let tn = Rc::new(HTN::new(BTreeSet::new(), vec![], domain, HashMap::new()));
        ReachNode {
            tn,
            state: Rc::new(HashSet::new()),
            kind,
            prob_upper: 1.0,
            cost_lower,
            successors: vec![],
            landmarks: vec![],
        }
    }

    #[test]
    fn delta_nearest_tighter_than_max_h() {
        // 2 Assigned nodes (h=10, h=0), 1 Out_C node (h=3).
        // fond_f_value would give -(2 + 3) = -5.
        // delta_nearest: count=3, h_vals=[10,3,0] desc,
        //   delta = max(10+0, 3+1, 0+2) = 10
        //   min_out_c_h=3, min_h_term=2, lb = max(10, 3+2) = 10 → f=-10 (tighter).
        let reach = vec![
            dummy_node(NodeKind::Assigned, 10.0),
            dummy_node(NodeKind::Assigned, 0.0),
            dummy_node(NodeKind::Compound, 3.0),
        ];
        let out_c = vec![2usize];
        let f = delta_nearest_f_value(&ReachView::Full(&reach), &out_c, 2);
        assert!((f - (-10.0)).abs() < 1e-9, "expected -10.0, got {f}");
    }

    #[test]
    fn delta_nearest_degenerate_cases() {
        // Empty policy (policy_size=0), 2 Out_C nodes h=[5,2].
        // count=2, h_vals=[5,2], delta=max(5+0,2+1)=5
        // min_out_c_h=2, min_h_term=1, lb=max(5, 2+1)=5 → f=-5.
        let reach = vec![
            dummy_node(NodeKind::Compound, 5.0),
            dummy_node(NodeKind::Compound, 2.0),
        ];
        let f = delta_nearest_f_value(&ReachView::Full(&reach), &[0, 1], 0);
        assert!((f - (-5.0)).abs() < 1e-9, "expected -5.0, got {f}");

        // All-INFINITY out_c: count = policy_size + 1 = 4; no finite h → lb=4 → f=-4.
        let reach2 = vec![dummy_node(NodeKind::Compound, f64::INFINITY)];
        let f2 = delta_nearest_f_value(&ReachView::Full(&reach2), &[0], 3);
        assert!((f2 - (-4.0)).abs() < 1e-9, "expected -4.0, got {f2}");

        // Closed policy (out_c empty) → -(policy_size).
        let f3 = delta_nearest_f_value(&ReachView::Full(&[]), &[], 5);
        assert!((f3 - (-5.0)).abs() < 1e-9, "expected -5.0, got {f3}");
    }
}

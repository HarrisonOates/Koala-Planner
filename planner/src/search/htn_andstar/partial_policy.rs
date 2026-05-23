use std::collections::HashSet;
use std::cmp::Ordering;
use std::rc::Rc;
use crate::task_network::HTN;

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
    /// Admissible upper bound on success probability from this node.
    /// Used to pin the V-value of Compound (out_c) nodes during VI.
    pub prob_upper: f64,
    /// Outgoing edges: (successor reach-index, edge probability).
    /// Empty for Goal, Dead, and Compound nodes.
    pub successors: Vec<(usize, f64)>,
}

/// A single method assignment recorded in the policy.
#[derive(Clone)]
pub struct PolicyAssignment {
    pub tn_snapshot: Rc<HTN>,
    pub state: Rc<HashSet<u32>>,
    pub task_name: String,
    pub method_name: String,
}

/// Persistent (shared) linked list of policy assignments.
/// Each PartialPolicyState carries only its tail pointer; ancestors
/// are shared with sibling states via Rc, so creating a child is O(1).
pub struct PolicyLink {
    pub parent: Option<Rc<PolicyLink>>,
    pub assignment: PolicyAssignment,
}

/// One node in the AND* open list — a partial policy π.
pub struct PartialPolicyState {
    /// Forward-reachable nodes from (init_tn, init_state) under π.
    /// reach[0] is always the initial node.
    pub reach: Vec<ReachNode>,
    /// Indices into `reach` for unassigned compound nodes (Out_C(π)).
    pub out_c: Vec<usize>,
    /// Singly-linked list of assignments made so far.
    pub policy_tail: Option<Rc<PolicyLink>>,
    /// f(π) = V[0] from Bellman VI with out_c nodes pinned at prob_upper.
    /// Equals the exact success probability when the policy is closed.
    pub f_value: f64,
    /// Number of compound-node assignments in policy_tail (i.e. |π|).
    pub policy_size: usize,
    /// Which secondary criterion to use when f-values tie.
    pub tiebreaker: TiebreakerKind,
}

impl PartialPolicyState {
    pub fn is_closed(&self) -> bool {
        self.out_c.is_empty()
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
    pub fn compute_f_by_vi(reach: &[ReachNode]) -> f64 {
        let n = reach.len();
        if n == 0 {
            return 1.0; // empty problem — trivially solved
        }

        // Initialise value vector.
        let mut v: Vec<f64> = reach.iter().map(|node| match node.kind {
            NodeKind::Goal     => 1.0,
            NodeKind::Dead     => 0.0,
            NodeKind::Compound => node.prob_upper, // pinned
            _                  => 0.0,             // pessimistic init
        }).collect();

        // Bellman VI — converges for both acyclic and cyclic graphs
        // when initialised pessimistically (lower fixed point = MaxProb).
        const MAX_ITERS: usize = 100_000;
        for _ in 0..MAX_ITERS {
            let mut delta = 0.0_f64;
            for i in 0..n {
                match reach[i].kind {
                    // Fixed-value nodes — skip.
                    NodeKind::Goal | NodeKind::Dead | NodeKind::Compound => continue,
                    _ => {}
                }
                let new_v: f64 = reach[i].successors.iter()
                    .map(|&(j, p)| p * v[j])
                    .sum();
                let diff = (new_v - v[i]).abs();
                if diff > delta { delta = diff; }
                v[i] = new_v;
            }
            if delta < 1e-12 { break; }
        }

        v[0]
    }
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
        let f_ord = self.f_value.partial_cmp(&other.f_value).unwrap_or(Ordering::Equal);
        if f_ord != Ordering::Equal {
            return f_ord;
        }
        // Tiebreaker (configurable):
        match self.tiebreaker {
            // No secondary criterion.
            TiebreakerKind::NoTiebreak => Ordering::Equal,
            // Prefer more assignments — deeper DFS-like dive.
            TiebreakerKind::PolicySize =>
                self.policy_size.cmp(&other.policy_size),
            // Prefer fewer unresolved compound nodes — closer to a closed policy.
            // Reversed because fewer is better and BinaryHeap is a max-heap.
            TiebreakerKind::ClosureFirst =>
                other.out_c.len().cmp(&self.out_c.len()),
            // More assignments first; break further ties by fewer unresolved.
            TiebreakerKind::Combined =>
                self.policy_size.cmp(&other.policy_size)
                    .then_with(|| other.out_c.len().cmp(&self.out_c.len())),
        }
    }
}

// ── Deduplication signature ──────────────────────────────────────────────────

/// Canonical descriptor of a single unassigned compound reach node.
#[derive(Hash, PartialEq, Eq, PartialOrd, Ord, Clone)]
pub struct CompoundSig {
    pub tn_mappings:  Vec<(u32, u32)>, // sorted by renamed node id
    pub tn_orderings: Vec<(u32, u32)>, // sorted, deduped
    pub state_facts:  Vec<u32>,        // sorted fact IDs
}

/// Full deduplication key for a PartialPolicyState.
/// Two states with the same f_value and identical unresolved compound
/// frontier are considered equivalent and the second is dropped.
#[derive(Hash, PartialEq, Eq)]
pub struct ReachSig {
    pub f_value_bits: u64,
    pub compounds:    Vec<CompoundSig>, // sorted for canonical form
}

impl PartialPolicyState {
    /// Produce a canonical renaming of TN node IDs (independent of
    /// the arbitrary integer IDs assigned during grounding).
    fn canonicalize_tn(tn: &HTN) -> (Vec<(u32, u32)>, Vec<(u32, u32)>) {
        let mut old_nodes: Vec<u32> = tn.get_nodes().iter().copied().collect();
        old_nodes.sort();
        let renaming: std::collections::HashMap<u32, u32> = old_nodes
            .iter()
            .enumerate()
            .map(|(new_id, &old_id)| (old_id, new_id as u32))
            .collect();

        let mut tn_mappings: Vec<(u32, u32)> = tn.mappings.iter()
            .map(|(&old_node, &task_id)| (renaming[&old_node], task_id))
            .collect();
        tn_mappings.sort();

        let mut tn_orderings: Vec<(u32, u32)> = tn.get_orderings().into_iter()
            .map(|(src, dst)| (renaming[&src], renaming[&dst]))
            .collect();
        tn_orderings.sort();
        tn_orderings.dedup();

        (tn_mappings, tn_orderings)
    }

    /// Build a canonical signature of this partial policy for open-list
    /// deduplication. Based on: f_value + the set of unresolved compound nodes.
    pub fn reach_sig(&self) -> ReachSig {
        let mut compounds: Vec<CompoundSig> = self.reach.iter()
            .filter(|n| matches!(n.kind, NodeKind::Compound))
            .map(|n| {
                let (tn_mappings, tn_orderings) = Self::canonicalize_tn(n.tn.as_ref());
                let mut state_facts: Vec<u32> = n.state.iter().copied().collect();
                state_facts.sort();
                CompoundSig { tn_mappings, tn_orderings, state_facts }
            })
            .collect();
        compounds.sort();
        ReachSig {
            f_value_bits: self.f_value.to_bits(),
            compounds,
        }
    }
}

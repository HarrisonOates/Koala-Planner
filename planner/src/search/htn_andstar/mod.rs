mod partial_policy;
mod domain_tests;

use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet, VecDeque};
use std::rc::Rc;
use std::time::Instant;

use crate::domain_description::FONDProblem;
use crate::relaxation::{OutcomeDeterminizer, RelaxedComposition};
use crate::task_network::HTN;

use partial_policy::{
    NodeKind, PartialPolicyState, PolicyAssignment, PolicyLink, ReachNode, ReachSig,
};
pub use partial_policy::TiebreakerKind;

use super::{
    ConnectionLabel, HeuristicType, PolicyNode, PolicyOutput, SearchGraphNode, SearchResult,
    SearchStats, StrongPolicy,
};
use super::progress;

// ── Key types ──────────────────────────────────────────────────────────────

#[derive(Hash, PartialEq, Eq, Clone)]
struct TnKey {
    mappings: BTreeMap<u32, u32>,
    orderings: Vec<(u32, u32)>,
}

#[derive(Hash, PartialEq, Eq, Clone)]
struct StateKey(Vec<u32>);

type MemoKey = (TnKey, StateKey);

fn make_key(tn: &HTN, state: &HashSet<u32>) -> MemoKey {
    let mappings: BTreeMap<u32, u32> = tn.mappings.iter().map(|(&k, &v)| (k, v)).collect();
    let mut orderings = tn.get_orderings();
    orderings.sort();
    orderings.dedup();
    let mut sv: Vec<u32> = state.iter().copied().collect();
    sv.sort();
    (TnKey { mappings, orderings }, StateKey(sv))
}

// ── Reach-graph construction ───────────────────────────────────────────────

/// Build Reach(π): the set of (tn, state) pairs reachable from
/// (init_tn, init_state) by following `assignments` at compound nodes
/// and branching over all outcomes at primitive nodes.
///
/// Returns `(reach, out_c)` where:
/// - `reach[0]` is always the initial node.
/// - `out_c` holds indices of unassigned compound nodes (Out_C(π)).
fn compute_reach(
    init_tn: Rc<HTN>,
    init_state: Rc<HashSet<u32>>,
    assignments: &HashMap<MemoKey, (String, String)>,
    relaxed: &RelaxedComposition,
    bijection: &HashMap<u32, u32>,
    h_type: &HeuristicType,
) -> (Vec<ReachNode>, Vec<usize>) {
    // reach[i] starts as None and is filled when the node is dequeued.
    let mut reach: Vec<Option<ReachNode>> = Vec::new();
    let mut index_map: HashMap<MemoKey, usize> = HashMap::new();
    let mut out_c: Vec<usize> = Vec::new();

    // Allocate slot 0 for the initial node.
    let init_key = make_key(&init_tn, &init_state);
    index_map.insert(init_key, 0);
    reach.push(None);

    // BFS queue: (tn, state, pre-allocated reach index).
    let mut queue: VecDeque<(Rc<HTN>, Rc<HashSet<u32>>, usize)> = VecDeque::new();
    queue.push_back((init_tn, init_state, 0));

    while let Some((tn, state, node_idx)) = queue.pop_front() {
        // ── Goal ──────────────────────────────────────────────────────────
        if tn.is_empty() {
            reach[node_idx] = Some(ReachNode {
                tn, state,
                kind: NodeKind::Goal,
                prob_upper: 1.0,
                successors: vec![],
            });
            continue;
        }

        let expansions = progress(tn.clone(), state.clone());

        // ── Dead end ──────────────────────────────────────────────────────
        if expansions.is_empty() {
            reach[node_idx] = Some(ReachNode {
                tn, state,
                kind: NodeKind::Dead,
                prob_upper: 0.0,
                successors: vec![],
            });
            continue;
        }

        let has_decomposition = expansions.iter().any(|e| e.connection_label.is_decomposition());

        if has_decomposition {
            // ── Compound task ─────────────────────────────────────────────
            let h = SearchGraphNode::h_val(
                tn.as_ref(), state.as_ref(), relaxed, bijection, h_type,
            );
            let prob_upper = if h == f32::INFINITY { 0.0 } else { 1.0 };
            let key = make_key(&tn, &state);

            if let Some((task_name, method_name)) = assignments.get(&key) {
                // Assigned: follow the chosen method.
                let matching = expansions.iter().find(|e| match &e.connection_label {
                    ConnectionLabel::Decomposition(t, m) => t == task_name && m == method_name,
                    _ => false,
                });

                if let Some(decomp) = matching {
                    let succ_key = make_key(&decomp.tn, &state);
                    let succ_idx = get_or_insert(
                        succ_key, &mut index_map, &mut reach, &mut queue,
                        decomp.tn.clone(), state.clone(),
                    );
                    reach[node_idx] = Some(ReachNode {
                        tn, state,
                        kind: NodeKind::Assigned,
                        prob_upper,
                        successors: vec![(succ_idx, 1.0)],
                    });
                } else {
                    // Assigned method not found — treat as dead end.
                    reach[node_idx] = Some(ReachNode {
                        tn, state,
                        kind: NodeKind::Dead,
                        prob_upper: 0.0,
                        successors: vec![],
                    });
                }
            } else {
                // Unassigned: add to Out_C.
                reach[node_idx] = Some(ReachNode {
                    tn, state,
                    kind: NodeKind::Compound,
                    prob_upper,
                    successors: vec![],
                });
                out_c.push(node_idx);
            }
        } else {
            // ── Primitive action ──────────────────────────────────────────
            // progress() returns one expansion per unconstrained primitive.
            // We execute only the first one — the task network ordering
            // determines which primitive runs next; summing all expansions
            // would make probabilities exceed 1.0.
            let mut successors: Vec<(usize, f64)> = Vec::new();
            if let Some(expansion) = expansions.first() {
                for (i, outcome_state) in expansion.states.iter().enumerate() {
                    let p = expansion.outcome_probabilities.get(i).copied().unwrap_or(1.0);
                    let succ_key = make_key(&expansion.tn, outcome_state);
                    let succ_idx = get_or_insert(
                        succ_key, &mut index_map, &mut reach, &mut queue,
                        expansion.tn.clone(), outcome_state.clone(),
                    );
                    successors.push((succ_idx, p));
                }
            }
            reach[node_idx] = Some(ReachNode {
                tn, state,
                kind: NodeKind::Primitive,
                prob_upper: 1.0,
                successors,
            });
        }
    }

    let reach = reach.into_iter().map(|n| n.unwrap()).collect();
    (reach, out_c)
}

/// Return the reach index for `(tn, state)`, inserting a placeholder and
/// enqueueing if the pair has not been seen before.
fn get_or_insert(
    key: MemoKey,
    index_map: &mut HashMap<MemoKey, usize>,
    reach: &mut Vec<Option<ReachNode>>,
    queue: &mut VecDeque<(Rc<HTN>, Rc<HashSet<u32>>, usize)>,
    tn: Rc<HTN>,
    state: Rc<HashSet<u32>>,
) -> usize {
    if let Some(&idx) = index_map.get(&key) {
        idx // already allocated — back-edge or convergence
    } else {
        let idx = reach.len();
        index_map.insert(key, idx);
        reach.push(None); // placeholder filled when dequeued
        queue.push_back((tn, state, idx));
        idx
    }
}

// ── Assignment reconstruction ──────────────────────────────────────────────

/// Walk the PolicyLink chain and reconstruct the assignments as a HashMap.
fn collect_assignments(
    policy_tail: &Option<Rc<PolicyLink>>,
) -> HashMap<MemoKey, (String, String)> {
    let mut map = HashMap::new();
    let mut cur = policy_tail.as_ref();
    while let Some(link) = cur {
        let a = &link.assignment;
        let key = make_key(a.tn_snapshot.as_ref(), a.state.as_ref());
        map.insert(key, (a.task_name.clone(), a.method_name.clone()));
        cur = link.parent.as_ref();
    }
    map
}

// ── Algorithm 2 — HTN-AND* ─────────────────────────────────────────────────

pub fn run(problem: &FONDProblem, h_type: HeuristicType, tiebreaker: TiebreakerKind) -> (SearchResult, SearchStats) {
    let start_time = Instant::now();
    let (outcome_det, bijection) = OutcomeDeterminizer::from_fond_problem(problem);
    let relaxed = RelaxedComposition::new(&outcome_det);

    // Build the initial empty partial policy π_I.
    let empty_assignments: HashMap<MemoKey, (String, String)> = HashMap::new();
    let (reach, out_c) = compute_reach(
        Rc::new(problem.init_tn.clone()),
        Rc::new(problem.initial_state.clone()),
        &empty_assignments,
        &relaxed, &bijection, &h_type,
    );
    let f_value = PartialPolicyState::compute_f_by_vi(&reach);
    let pi_i = PartialPolicyState { reach, out_c, policy_tail: None, f_value, policy_size: 0, tiebreaker };

    let mut open: BinaryHeap<PartialPolicyState> = BinaryHeap::new();
    let mut seen: HashSet<ReachSig> = HashSet::new();
    seen.insert(pi_i.reach_sig());
    open.push(pi_i);

    let mut explored = 0u32;

    while let Some(pi) = open.pop() {
        explored += 1;

        if explored % 10 == 0 {
            eprintln!(
                "[AND*] explored={} | open={} | best_f={:.4} | {:.1}s",
                explored, open.len(), pi.f_value,
                start_time.elapsed().as_secs_f64()
            );
        }

        // ── Closed policy found ────────────────────────────────────────────
        if pi.is_closed() {
            let success_prob = pi.f_value;
            let stats = make_stats(explored, start_time, Some(success_prob), problem.rho);
            if success_prob < problem.rho {
                // Best achievable probability is below threshold.
                return (SearchResult::NoSolution, stats);
            }
            // Collect assignments from the PolicyLink chain.
            let mut assignments: Vec<(Rc<HTN>, Rc<HashSet<u32>>, String, String)> = vec![];
            let mut cur = pi.policy_tail.as_ref();
            while let Some(link) = cur {
                let a = &link.assignment;
                assignments.push((
                    a.tn_snapshot.clone(), a.state.clone(),
                    a.task_name.clone(), a.method_name.clone(),
                ));
                cur = link.parent.as_ref();
            }

            let transitions: Vec<(PolicyNode, PolicyOutput)> = assignments.iter()
                .map(|(tn, state, task_name, method_name)| {
                    let state_strings: HashSet<String> = state.iter()
                        .map(|id| problem.facts.get_fact(*id).clone())
                        .collect();
                    (
                        PolicyNode { tn: tn.clone(), state: state_strings },
                        PolicyOutput { task: task_name.clone(), method: method_name.clone() },
                    )
                })
                .collect();

            let policy = StrongPolicy {
                transitions,
                makespan: assignments.len() as u16,
                success_probability: success_prob,
            };
            return (SearchResult::Success(policy), stats);
        }

        // ── Expand first unresolved compound node ──────────────────────────
        let node_idx   = pi.out_c[0];
        let node_tn    = pi.reach[node_idx].tn.clone();
        let node_state = pi.reach[node_idx].state.clone();

        let expansions = progress(node_tn.clone(), node_state.clone());

        for expansion in expansions.iter().filter(|e| e.connection_label.is_decomposition()) {
            let (task_name, method_name) = match &expansion.connection_label {
                ConnectionLabel::Decomposition(t, m) => (t.clone(), m.clone()),
                _ => unreachable!(),
            };

            // New assignments = parent's assignments + this one.
            let node_key = make_key(node_tn.as_ref(), node_state.as_ref());
            let mut new_assignments = collect_assignments(&pi.policy_tail);
            new_assignments.insert(node_key, (task_name.clone(), method_name.clone()));

            let (new_reach, new_out_c) = compute_reach(
                Rc::new(problem.init_tn.clone()),
                Rc::new(problem.initial_state.clone()),
                &new_assignments,
                &relaxed, &bijection, &h_type,
            );
            let new_f = PartialPolicyState::compute_f_by_vi(&new_reach);

            // Prune closed policies that cannot meet the threshold.
            if new_out_c.is_empty() && new_f < problem.rho {
                continue;
            }

            let new_link = Rc::new(PolicyLink {
                parent: pi.policy_tail.clone(),
                assignment: PolicyAssignment {
                    tn_snapshot: node_tn.clone(),
                    state: node_state.clone(),
                    task_name,
                    method_name,
                },
            });

            let pi_prime = PartialPolicyState {
                reach: new_reach,
                out_c: new_out_c,
                policy_tail: Some(new_link),
                f_value: new_f,
                policy_size: pi.policy_size + 1,
                tiebreaker,
            };

            let sig = pi_prime.reach_sig();
            if !seen.contains(&sig) {
                seen.insert(sig);
                open.push(pi_prime);
            }
        }
    }

    (SearchResult::NoSolution, make_stats(explored, start_time, None, problem.rho))
}

fn make_stats(
    explored: u32,
    start_time: Instant,
    prob: Option<f64>,
    rho: f64,
) -> SearchStats {
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
    use std::collections::{BTreeSet, HashMap, HashSet};
    use std::rc::Rc;
    use crate::domain_description::{Facts, DomainTasks, FONDProblem};
    use crate::task_network::{Task, PrimitiveAction, CompoundTask, Method, HTN};
    use crate::search::{SearchResult, HeuristicType};
    use super::run;
    use super::TiebreakerKind;

    /// Build a domain with two sequential ND actions:
    ///   a1 (prob p1 → adds f1, 1-p1 → nothing)  →
    ///   a2 (prob p2 → adds f2, 1-p2 → nothing, precond f1)  →
    ///   a3 (deterministic gate, precond f2)
    /// Max achievable success probability = p1 * p2.
    fn build_two_nd_problem(p1: f64, p2: f64, rho: f64) -> FONDProblem {
        let a1 = Task::Primitive(PrimitiveAction::new_with_probabilities(
            "a1".to_string(), 1,
            HashSet::new(),
            vec![HashSet::from([0]), HashSet::new()],  // add f1 | nothing
            vec![HashSet::new(), HashSet::new()],
            vec![p1, 1.0 - p1],
        ));
        let a2 = Task::Primitive(PrimitiveAction::new_with_probabilities(
            "a2".to_string(), 1,
            HashSet::from([0]),                        // precond: f1
            vec![HashSet::from([1]), HashSet::new()],  // add f2 | nothing
            vec![HashSet::new(), HashSet::new()],
            vec![p2, 1.0 - p2],
        ));
        let a3 = Task::Primitive(PrimitiveAction::new(
            "a3".to_string(), 1,
            HashSet::from([1]),     // precond: f2
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
                "Expected 0.25, got {}", policy.success_probability
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
                "Expected 0.25, got {}", policy.success_probability
            ),
            SearchResult::NoSolution => panic!("Expected success at exact threshold"),
        }
    }

    #[test]
    fn deterministic_single_action() {
        let a = Task::Primitive(PrimitiveAction::new(
            "a".to_string(), 1,
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
                "Expected 1.0, got {}", policy.success_probability
            ),
            SearchResult::NoSolution => panic!("Expected success for deterministic action"),
        }
    }

    #[test]
    fn single_nd_action_point_seven() {
        let a1 = Task::Primitive(PrimitiveAction::new_with_probabilities(
            "a1".to_string(), 1,
            HashSet::new(),
            vec![HashSet::from([0]), HashSet::new()],
            vec![HashSet::new(), HashSet::new()],
            vec![0.7, 0.3],
        ));
        let gate = Task::Primitive(PrimitiveAction::new(
            "gate".to_string(), 1,
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
                "Expected 0.7, got {}", policy.success_probability
            ),
            SearchResult::NoSolution => panic!("Expected success with prob 0.7"),
        }
    }

    #[test]
    fn three_nd_actions_eighth_prob() {
        let a1 = Task::Primitive(PrimitiveAction::new_with_probabilities(
            "a1".to_string(), 1, HashSet::new(),
            vec![HashSet::from([0]), HashSet::new()],
            vec![HashSet::new(), HashSet::new()],
            vec![0.5, 0.5],
        ));
        let a2 = Task::Primitive(PrimitiveAction::new_with_probabilities(
            "a2".to_string(), 1, HashSet::from([0]),
            vec![HashSet::from([1]), HashSet::new()],
            vec![HashSet::new(), HashSet::new()],
            vec![0.5, 0.5],
        ));
        let a3 = Task::Primitive(PrimitiveAction::new_with_probabilities(
            "a3".to_string(), 1, HashSet::from([1]),
            vec![HashSet::from([2]), HashSet::new()],
            vec![HashSet::new(), HashSet::new()],
            vec![0.5, 0.5],
        ));
        let gate = Task::Primitive(PrimitiveAction::new(
            "gate".to_string(), 1, HashSet::from([2]),
            vec![HashSet::new()], vec![HashSet::new()],
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
                "Expected 0.125, got {}", policy.success_probability
            ),
            SearchResult::NoSolution => panic!("Expected success with prob 0.125"),
        }
    }

    #[test]
    fn three_nd_actions_above_threshold_no_solution() {
        let a1 = Task::Primitive(PrimitiveAction::new_with_probabilities(
            "a1".to_string(), 1, HashSet::new(),
            vec![HashSet::from([0]), HashSet::new()],
            vec![HashSet::new(), HashSet::new()],
            vec![0.5, 0.5],
        ));
        let a2 = Task::Primitive(PrimitiveAction::new_with_probabilities(
            "a2".to_string(), 1, HashSet::from([0]),
            vec![HashSet::from([1]), HashSet::new()],
            vec![HashSet::new(), HashSet::new()],
            vec![0.5, 0.5],
        ));
        let a3 = Task::Primitive(PrimitiveAction::new_with_probabilities(
            "a3".to_string(), 1, HashSet::from([1]),
            vec![HashSet::from([2]), HashSet::new()],
            vec![HashSet::new(), HashSet::new()],
            vec![0.5, 0.5],
        ));
        let gate = Task::Primitive(PrimitiveAction::new(
            "gate".to_string(), 1, HashSet::from([2]),
            vec![HashSet::new()], vec![HashSet::new()],
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
            tasks: domain, initial_state: HashSet::new(), init_tn: tn, rho: 0.2,
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
                "Expected 0.48, got {}", policy.success_probability
            ),
            SearchResult::NoSolution => panic!("Expected solution with prob 0.48"),
        }
    }

    #[test]
    fn method_choice_selects_best() {
        let a_hi = Task::Primitive(PrimitiveAction::new_with_probabilities(
            "a_hi".to_string(), 1, HashSet::new(),
            vec![HashSet::from([0]), HashSet::new()],
            vec![HashSet::new(), HashSet::new()],
            vec![0.8, 0.2],
        ));
        let a_lo = Task::Primitive(PrimitiveAction::new_with_probabilities(
            "a_lo".to_string(), 1, HashSet::new(),
            vec![HashSet::from([0]), HashSet::new()],
            vec![HashSet::new(), HashSet::new()],
            vec![0.3, 0.7],
        ));
        let a_gate = Task::Primitive(PrimitiveAction::new(
            "a_gate".to_string(), 1, HashSet::from([0]),
            vec![HashSet::new()], vec![HashSet::new()],
        ));
        let t = Task::Compound(CompoundTask { name: "t".to_string(), methods: vec![] });
        let domain = Rc::new(DomainTasks::new(vec![a_hi, a_lo, a_gate, t]));

        let m_hi = Method::new("m_hi".to_string(), HTN::new(
            BTreeSet::from([1, 2]), vec![(1, 2)], domain.clone(),
            HashMap::from([(1, 0), (2, 2)]),
        ));
        let m_lo = Method::new("m_lo".to_string(), HTN::new(
            BTreeSet::from([1, 2]), vec![(1, 2)], domain.clone(),
            HashMap::from([(1, 1), (2, 2)]),
        ));
        let domain = domain.add_methods(vec![(3, m_hi), (3, m_lo)]);

        let tn = HTN::new(
            BTreeSet::from([1]), vec![], domain.clone(),
            HashMap::from([(1, 3)]),
        );
        let mut problem = FONDProblem {
            facts: Facts::new(vec!["f_done".to_string()]),
            tasks: domain, initial_state: HashSet::new(), init_tn: tn, rho: 0.0,
        };
        problem.collapse_tn();

        let (result, _) = run(&problem, HeuristicType::HAdd, TiebreakerKind::Combined);
        match result {
            SearchResult::Success(policy) => assert!(
                (policy.success_probability - 0.8).abs() < 1e-9,
                "Expected 0.8 (best method), got {}", policy.success_probability
            ),
            SearchResult::NoSolution => panic!("Expected solution with prob 0.8"),
        }
    }

    #[test]
    fn method_choice_hprob_selects_best() {
        let a_hi = Task::Primitive(PrimitiveAction::new_with_probabilities(
            "a_hi".to_string(), 1, HashSet::new(),
            vec![HashSet::from([0]), HashSet::new()],
            vec![HashSet::new(), HashSet::new()],
            vec![0.8, 0.2],
        ));
        let a_lo = Task::Primitive(PrimitiveAction::new_with_probabilities(
            "a_lo".to_string(), 1, HashSet::new(),
            vec![HashSet::from([0]), HashSet::new()],
            vec![HashSet::new(), HashSet::new()],
            vec![0.3, 0.7],
        ));
        let a_gate = Task::Primitive(PrimitiveAction::new(
            "a_gate".to_string(), 1, HashSet::from([0]),
            vec![HashSet::new()], vec![HashSet::new()],
        ));
        let t = Task::Compound(CompoundTask { name: "t".to_string(), methods: vec![] });
        let domain = Rc::new(DomainTasks::new(vec![a_hi, a_lo, a_gate, t]));

        let m_hi = Method::new("m_hi".to_string(), HTN::new(
            BTreeSet::from([1, 2]), vec![(1, 2)], domain.clone(),
            HashMap::from([(1, 0), (2, 2)]),
        ));
        let m_lo = Method::new("m_lo".to_string(), HTN::new(
            BTreeSet::from([1, 2]), vec![(1, 2)], domain.clone(),
            HashMap::from([(1, 1), (2, 2)]),
        ));
        let domain = domain.add_methods(vec![(3, m_hi), (3, m_lo)]);

        let tn = HTN::new(
            BTreeSet::from([1]), vec![], domain.clone(),
            HashMap::from([(1, 3)]),
        );
        let mut problem = FONDProblem {
            facts: Facts::new(vec!["f_done".to_string()]),
            tasks: domain, initial_state: HashSet::new(), init_tn: tn, rho: 0.0,
        };
        problem.collapse_tn();

        let (result, _) = run(&problem, HeuristicType::HProb, TiebreakerKind::Combined);
        match result {
            SearchResult::Success(policy) => assert!(
                (policy.success_probability - 0.8).abs() < 1e-9,
                "Expected 0.8 (best method), got {}", policy.success_probability
            ),
            SearchResult::NoSolution => panic!("Expected solution with prob 0.8"),
        }
    }

    #[test]
    fn two_nd_actions_hprob_quarter_prob() {
        let problem = build_two_nd_problem(0.5, 0.5, 0.0);
        let (result, _) = run(&problem, HeuristicType::HProb, TiebreakerKind::Combined);
        match result {
            SearchResult::Success(policy) => assert!(
                (policy.success_probability - 0.25).abs() < 1e-9,
                "Expected 0.25, got {}", policy.success_probability
            ),
            SearchResult::NoSolution => panic!("Expected solution with prob 0.25"),
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
}

use std::collections::{HashSet, VecDeque};

use crate::domain_description::ClassicalDomain;

/// A landmark cut: the set of action indices in the cut, and the cost deducted.
pub type LandmarkCuts = Vec<(Vec<usize>, u32)>;

/// Standard LM-cut heuristic value (landmarks discarded after computation).
pub fn h_lmcut(domain: &ClassicalDomain, state: &HashSet<u32>, goal: &HashSet<u32>) -> f32 {
    if goal.iter().all(|g| state.contains(g)) {
        return 0.0;
    }
    h_lmcut_core(domain, state, goal, &[] as &[(Vec<usize>, u32)]).0
}

/// LM-cut returning (value, discovered landmark cuts) for storage and incremental reuse.
pub fn h_lmcut_full(
    domain: &ClassicalDomain,
    state: &HashSet<u32>,
    goal: &HashSet<u32>,
) -> (f32, LandmarkCuts) {
    if goal.iter().all(|g| state.contains(g)) {
        return (0.0, vec![]);
    }
    h_lmcut_core(domain, state, goal, &[] as &[(Vec<usize>, u32)])
}

/// Incremental LM-cut: warm-start from `parent_cuts`, dropping cuts that contain
/// `exclude_action` (the classical action corresponding to the applied method).
///
/// Admissibility: any carried cut L with `exclude_action ∉ L` is still a disjunctive
/// action landmark for the child node (Pommerening & Helmert 2013, §"Incremental
/// Computation"), provided the child's state is at most as hard as the parent's.
/// In our HTN encoding this holds when no primitive actions were executed between
/// the parent compound node and the child (the no-primitives path).
pub fn h_lmcut_incremental(
    domain: &ClassicalDomain,
    state: &HashSet<u32>,
    goal: &HashSet<u32>,
    parent_cuts: &[(Vec<usize>, u32)],
    exclude_action: usize,
) -> (f32, LandmarkCuts) {
    if goal.iter().all(|g| state.contains(g)) {
        return (0.0, vec![]);
    }
    let carried: LandmarkCuts = parent_cuts
        .iter()
        .filter(|(actions, _)| !actions.contains(&exclude_action))
        .cloned()
        .collect();
    h_lmcut_core(domain, state, goal, &carried)
}

/// Core LM-cut computation.
///
/// `carried_cuts` are landmark cuts already paid for (from a parent node).
/// Initial action costs are deflated by their accumulated deductions, and
/// `h_total` starts at the sum of carried cut costs.  All new cuts found are
/// appended to `carried_cuts` and returned alongside the total value.
fn h_lmcut_core(
    domain: &ClassicalDomain,
    state: &HashSet<u32>,
    goal: &HashSet<u32>,
    carried_cuts: &[(Vec<usize>, u32)],
) -> (f32, LandmarkCuts) {
    let n_facts = domain.facts.count() as usize;
    let n_actions = domain.actions.len();

    let mut achievers: Vec<Vec<usize>> = vec![Vec::new(); n_facts];
    for (i, action) in domain.actions.iter().enumerate() {
        if action.add_effects.is_empty() {
            continue;
        }
        for &f in &action.add_effects[0] {
            let fi = f as usize;
            if fi < n_facts {
                achievers[fi].push(i);
            }
        }
    }

    // Recreate costs from carried cuts: costs[a] = 1 - Σ{c : a ∈ L, (L,c) ∈ carried}
    let mut costs: Vec<u32> = vec![1; n_actions];
    let mut h_total: u32 = 0;
    for (cut_actions, cut_cost) in carried_cuts {
        h_total += cut_cost;
        for &a in cut_actions {
            if a < n_actions {
                costs[a] = costs[a].saturating_sub(*cut_cost);
            }
        }
    }

    let mut all_cuts: LandmarkCuts = carried_cuts.to_vec();

    loop {
        // === 1. h_max fixed-point propagation ===
        let mut hmax = vec![u32::MAX; n_facts];
        for &f in state {
            let fi = f as usize;
            if fi < n_facts {
                hmax[fi] = 0;
            }
        }
        let mut changed = true;
        while changed {
            changed = false;
            for (idx, action) in domain.actions.iter().enumerate() {
                if action.add_effects.is_empty() {
                    continue;
                }
                let act_hmax = action.pre_cond.iter().fold(0u32, |acc, &p| {
                    let pv = if (p as usize) < n_facts { hmax[p as usize] } else { u32::MAX };
                    if acc == u32::MAX || pv == u32::MAX { u32::MAX } else { acc.max(pv) }
                });
                let fact_val = if act_hmax == u32::MAX {
                    u32::MAX
                } else {
                    costs[idx].saturating_add(act_hmax)
                };
                for &f in &action.add_effects[0] {
                    let fi = f as usize;
                    if fi < n_facts && fact_val < hmax[fi] {
                        hmax[fi] = fact_val;
                        changed = true;
                    }
                }
            }
        }

        let goal_hmax = goal.iter().fold(0u32, |acc, &g| {
            let gv = if (g as usize) < n_facts { hmax[g as usize] } else { u32::MAX };
            if acc == u32::MAX || gv == u32::MAX { u32::MAX } else { acc.max(gv) }
        });
        if goal_hmax == u32::MAX {
            return (f32::INFINITY, all_cuts);
        }
        if goal_hmax == 0 {
            return (h_total as f32, all_cuts);
        }

        // === 2. Build goal zone Z via backward BFS ===
        let mut z_zone = vec![false; n_facts];
        let mut queue: VecDeque<u32> = VecDeque::new();
        for &g in goal {
            let gi = g as usize;
            if gi < n_facts && hmax[gi] > 0 && !z_zone[gi] {
                z_zone[gi] = true;
                queue.push_back(g);
            }
        }
        while let Some(f) = queue.pop_front() {
            let fi = f as usize;
            let f_hmax = hmax[fi];
            for &idx in &achievers[fi] {
                let action = &domain.actions[idx];
                let act_hmax = action.pre_cond.iter().fold(0u32, |acc, &p| {
                    let pv = if (p as usize) < n_facts { hmax[p as usize] } else { u32::MAX };
                    if acc == u32::MAX || pv == u32::MAX { u32::MAX } else { acc.max(pv) }
                });
                if act_hmax == u32::MAX {
                    continue;
                }
                if costs[idx].saturating_add(act_hmax) != f_hmax {
                    continue;
                }
                if act_hmax > 0 {
                    for &p in &action.pre_cond {
                        let pi = p as usize;
                        if pi < n_facts && hmax[pi] == act_hmax && !z_zone[pi] {
                            z_zone[pi] = true;
                            queue.push_back(p);
                        }
                    }
                }
            }
        }

        // === 3. Find cut: hmax=0, cost>0, achieves a Z-fact ===
        let mut cut: Vec<usize> = Vec::new();
        let mut cut_cost = u32::MAX;
        for (idx, action) in domain.actions.iter().enumerate() {
            if costs[idx] == 0 || action.add_effects.is_empty() {
                continue;
            }
            let act_hmax = action.pre_cond.iter().fold(0u32, |acc, &p| {
                let pv = if (p as usize) < n_facts { hmax[p as usize] } else { u32::MAX };
                if acc == u32::MAX || pv == u32::MAX { u32::MAX } else { acc.max(pv) }
            });
            if act_hmax != 0 {
                continue;
            }
            if action.add_effects[0]
                .iter()
                .any(|&f| (f as usize) < n_facts && z_zone[f as usize])
            {
                cut.push(idx);
                cut_cost = cut_cost.min(costs[idx]);
            }
        }
        if cut.is_empty() || cut_cost == 0 {
            break;
        }

        h_total += cut_cost;
        all_cuts.push((cut.clone(), cut_cost));
        for idx in cut {
            costs[idx] -= cut_cost;
        }
    }

    (h_total as f32, all_cuts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain_description::{ClassicalDomain, Facts};
    use crate::task_network::PrimitiveAction;

    fn chain_domain() -> ClassicalDomain {
        let facts =
            Facts::new(["0", "1", "2", "3", "4"].iter().map(|s| s.to_string()).collect());
        let mk = |pre: HashSet<u32>, add: HashSet<u32>| -> PrimitiveAction {
            PrimitiveAction::new("".into(), 1, pre, vec![add], vec![HashSet::new()])
        };
        ClassicalDomain::new(
            facts,
            vec![
                mk(HashSet::from([0]), HashSet::from([1])),
                mk(HashSet::from([1]), HashSet::from([2])),
                mk(HashSet::from([1]), HashSet::from([3])),
                mk(HashSet::from([1, 2, 3]), HashSet::from([4])),
            ],
        )
    }

    #[test]
    fn goal_already_satisfied() {
        let d = chain_domain();
        assert_eq!(h_lmcut(&d, &HashSet::from([4u32]), &HashSet::from([4u32])), 0.0);
    }

    #[test]
    fn unreachable_returns_infinity() {
        let d = chain_domain();
        assert_eq!(h_lmcut(&d, &HashSet::from([0u32]), &HashSet::from([99u32])), f32::INFINITY);
    }

    #[test]
    fn single_step() {
        let d = chain_domain();
        assert_eq!(h_lmcut(&d, &HashSet::from([0u32]), &HashSet::from([1u32])), 1.0);
    }

    #[test]
    fn four_action_chain_gives_three() {
        let d = chain_domain();
        assert_eq!(h_lmcut(&d, &HashSet::from([0u32]), &HashSet::from([4u32])), 3.0);
    }

    #[test]
    fn admissible_not_greater_than_hadd() {
        use super::super::add::h_add;
        let d = chain_domain();
        let s = HashSet::from([0u32]);
        let g = HashSet::from([4u32]);
        assert!(h_lmcut(&d, &s, &g) <= h_add(&d, &s, &g));
    }

    #[test]
    fn full_returns_same_value_as_standard() {
        let d = chain_domain();
        let s = HashSet::from([0u32]);
        let g = HashSet::from([4u32]);
        let (val, cuts) = h_lmcut_full(&d, &s, &g);
        assert_eq!(val, 3.0);
        assert_eq!(cuts.len(), 3); // three landmark cuts found
    }

    #[test]
    fn incremental_with_no_carried_matches_full() {
        let d = chain_domain();
        let s = HashSet::from([0u32]);
        let g = HashSet::from([4u32]);
        let (v_full, _) = h_lmcut_full(&d, &s, &g);
        // exclude_action = 99 (nonexistent) → no filtering → should equal full
        let (v_inc, _) = h_lmcut_incremental(&d, &s, &g, &[] as &[(Vec<usize>, u32)], 99);
        assert_eq!(v_full, v_inc);
    }

    #[test]
    fn incremental_with_carried_cuts_admissible() {
        let d = chain_domain();
        let s = HashSet::from([0u32]);
        let g = HashSet::from([4u32]);
        let (v_full, cuts) = h_lmcut_full(&d, &s, &g);
        // Carry all cuts with a non-existent exclude_action → should reproduce full value
        let (v_inc, _) = h_lmcut_incremental(&d, &s, &g, &cuts, 99);
        assert_eq!(v_full, v_inc);
    }
}

use std::collections::{HashSet, VecDeque};

use crate::domain_description::ClassicalDomain;

pub fn h_lmcut(domain: &ClassicalDomain, state: &HashSet<u32>, goal: &HashSet<u32>) -> f32 {
    if goal.iter().all(|g| state.contains(g)) {
        return 0.0;
    }

    let n_facts = domain.facts.count() as usize;
    let n_actions = domain.actions.len();

    // achievers[f] = indices of actions whose first add-effect set contains f
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

    // Unit costs — consistent with h_add / h_max / h_ff conventions
    let mut costs: Vec<u32> = vec![1; n_actions];
    let mut h_total: u32 = 0;

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
            return f32::INFINITY;
        }
        if goal_hmax == 0 {
            return h_total as f32;
        }

        // === 2. Build goal zone Z via backward BFS ===
        //
        // Z contains facts that must be "newly achieved" by the cut.
        // Seed from goal facts with hmax > 0, then follow best-achiever edges
        // backward: for each best-achiever a of f (costs[a]+hmax(a)=hmax(f)),
        //   if hmax(a) > 0 add the peak preconditions { p | hmax(p) = hmax(a) }.
        //   if hmax(a) = 0 the action is a cut candidate (don't extend Z).
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
                    continue; // not a best achiever of f
                }
                if act_hmax > 0 {
                    // Extend Z through peak preconditions
                    for &p in &action.pre_cond {
                        let pi = p as usize;
                        if pi < n_facts && hmax[pi] == act_hmax && !z_zone[pi] {
                            z_zone[pi] = true;
                            queue.push_back(p);
                        }
                    }
                }
                // act_hmax == 0 → action is a cut candidate; don't extend Z
            }
        }

        // === 3. Find cut: actions with hmax = 0, cost > 0, achieving some Z-fact ===
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
        for idx in cut {
            costs[idx] -= cut_cost;
        }
    }

    h_total as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain_description::{ClassicalDomain, Facts};
    use crate::task_network::PrimitiveAction;

    fn chain_domain() -> ClassicalDomain {
        // p1: pre={0} add={1}
        // p2: pre={1} add={2}
        // p3: pre={1} add={3}
        // p4: pre={1,2,3} add={4}
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
        // Goal {1}: only p1 needed — one cut {p1}, h = 1
        let d = chain_domain();
        assert_eq!(h_lmcut(&d, &HashSet::from([0u32]), &HashSet::from([1u32])), 1.0);
    }

    #[test]
    fn four_action_chain_gives_three() {
        // iter1: cut={p1}       h=1  (p1 reduces to cost 0)
        // iter2: cut={p2,p3}    h=2  (both reduce to cost 0)
        // iter3: cut={p4}       h=3  (p4 reduces to cost 0)
        // iter4: goal_hmax=0  → return 3
        let d = chain_domain();
        assert_eq!(h_lmcut(&d, &HashSet::from([0u32]), &HashSet::from([4u32])), 3.0);
    }

    #[test]
    fn admissible_not_greater_than_hadd() {
        use super::super::add::h_add;
        let d = chain_domain();
        let s = HashSet::from([0u32]);
        let g = HashSet::from([4u32]);
        // h_add overcounts; h_lmcut is admissible ⇒ h_lmcut ≤ h_add
        assert!(h_lmcut(&d, &s, &g) <= h_add(&d, &s, &g));
    }
}

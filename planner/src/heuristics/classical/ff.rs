use std::collections::{HashSet};
use super::*;

pub fn h_ff(domain: &ClassicalDomain, state: &HashSet<u32>, goal: &HashSet<u32>) -> f32 {
    let graphplan = GraphPlan::build_graph(domain, state, goal);
    match graphplan {
        Some(graph) => return plan_length(domain, graph, goal) as f32,
        None => {
            return f32::INFINITY;
        }
    }
}

fn plan_length(domain: &ClassicalDomain, graphplan: GraphPlan, goal_state: &HashSet<u32>) -> u32 {
    let mut len = 0;
    let mut g = graphplan.compute_goal_indices(goal_state);
    let depth = graphplan.depth as usize;
    let n_facts = domain.facts.count() as usize;

    // Flat bool array: marks[layer * n_facts + fact_id].
    // Replaces Vec<HashSet<u32>> to avoid per-layer HashSet allocations.
    let mut marks = vec![false; (depth + 1) * n_facts];

    // Initial-state facts as a bool array for O(1) membership tests.
    let initial_facts_set = graphplan.get_fact_layer(0);
    let mut in_initial = vec![false; n_facts];
    for &f in &initial_facts_set {
        if (f as usize) < n_facts {
            in_initial[f as usize] = true;
        }
    }

    for i in (1..=depth).rev() {
        let i = i as u32;
        let Some(goals_at_layer) = g.get(&i) else { continue };
        let layer_off = i as usize * n_facts;
        let prev_off = (i as usize - 1) * n_facts;

        let mut open_goals: Vec<u32> = goals_at_layer
            .iter()
            .filter(|&&f| !marks[layer_off + f as usize])
            .cloned()
            .collect();
        open_goals.sort_unstable(); // deterministic order: eliminates HashSet iteration variance

        for &open_goal in &open_goals {
            // Intra-layer re-check: an earlier goal in this layer may have already
            // covered open_goal as a side-effect of its chosen action.
            if marks[layer_off + open_goal as usize] {
                continue;
            }

            // Direct lookup via effect_to_actions: find the cheapest action at layer i-1
            // that produces open_goal. Tiebreak by action index for determinism.
            let min_action_idx = graphplan.effect_to_actions[open_goal as usize]
                .iter()
                .filter(|&&a| graphplan.action_layer[a] == i - 1)
                .min_by_key(|&&a| (domain.actions[a].cost, a))
                .copied()
                .unwrap();
            let min_action = &domain.actions[min_action_idx];
            len += 1;

            // Add unsatisfied preconditions as new subgoals at their graphplan layer.
            for &precond in &min_action.pre_cond {
                let pi = precond as usize;
                if pi < n_facts && !in_initial[pi] && !marks[prev_off + pi] {
                    let layer = graphplan.fact_placed[precond as usize];
                    g.entry(layer).or_default().insert(precond);
                }
            }

            // Mark all effects: at layer_off (prevents re-selection as goal at this layer)
            // and at prev_off (prevents re-adding as subgoal for other goals at this layer).
            for &add in &min_action.add_effects[0] {
                let ai = add as usize;
                if ai < n_facts {
                    marks[layer_off + ai] = true;
                    marks[prev_off + ai] = true;
                }
            }
        }
    }
    len
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::domain_description::Facts;
    use crate::task_network::PrimitiveAction;

    pub fn generate_domain() -> ClassicalDomain {
        let p1 = PrimitiveAction::new(
            "p1".to_string(),
            1,
            HashSet::from([0]),
            vec![HashSet::from([1])],
            vec![HashSet::from([3])],
        );
        let p2 = PrimitiveAction::new(
            "p2".to_string(),
            1,
            HashSet::from([1]),
            vec![HashSet::from([2])],
            vec![HashSet::new()],
        );
        let p3 = PrimitiveAction::new(
            "p3".to_string(),
            1,
            HashSet::from([1]),
            vec![HashSet::from([3])],
            vec![HashSet::new()],
        );
        let p4 = PrimitiveAction::new(
            "p4".to_string(),
            1,
            HashSet::from([1, 2, 3]),
            vec![HashSet::from([4])],
            vec![HashSet::new()],
        );
        let facts = Facts::new(vec![
            "0".to_owned(),
            "1".to_owned(),
            "2".to_owned(),
            "3".to_owned(),
            "4".to_owned(),
        ]);
        let actions = vec![p1, p2, p3, p4];
        ClassicalDomain::new(facts, actions)
    }

    #[test]
    pub fn h_val_test() {
        let domain = generate_domain();
        let h = h_ff(&domain, &HashSet::from([0]), &HashSet::from([4]));
        assert_eq!(h, 4.0);
    }

    /// An action that simultaneously achieves two goal facts must be counted once,
    /// not twice. This verifies the intra-layer marks re-check.
    #[test]
    pub fn multi_goal_same_action_counted_once() {
        // p_joint: pre={0}, add={1, 2}  — achieves both goals in one step
        let p_joint = PrimitiveAction::new(
            "p_joint".to_string(),
            1,
            HashSet::from([0]),
            vec![HashSet::from([1, 2])],
            vec![HashSet::new()],
        );
        let facts = Facts::new(vec![
            "f0".to_owned(),
            "f1".to_owned(),
            "f2".to_owned(),
        ]);
        let domain = ClassicalDomain::new(facts, vec![p_joint]);
        // Both goals 1 and 2 are achieved by p_joint in one step: h_ff should be 1.
        let h = h_ff(&domain, &HashSet::from([0u32]), &HashSet::from([1u32, 2u32]));
        assert_eq!(h, 1.0);
    }
}

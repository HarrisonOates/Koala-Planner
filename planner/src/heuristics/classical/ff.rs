use std::collections::{HashMap, HashSet};
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

    // Vec-indexed marks — layers are dense integers 0..=depth
    let mut marks: Vec<HashSet<u32>> = vec![HashSet::new(); depth + 1];

    // Initial state facts are constant — hoist out of the per-goal loop
    let initial_facts = graphplan.get_fact_layer(0);

    for i in (1..=depth).rev() {
        let i = i as u32;
        let Some(goals_at_layer) = g.get(&i) else { continue };
        let mut open_goals: Vec<u32> = goals_at_layer
            .difference(&marks[i as usize])
            .cloned()
            .collect();
        open_goals.sort_unstable(); // deterministic order: eliminates HashSet iteration variance

        for open_goal in &open_goals {
            // Direct lookup via effect_to_actions: find the cheapest action at layer i-1
            // that produces open_goal, without scanning the whole layer.
            // Tiebreak by action index for determinism when action costs are equal.
            let min_action_idx = graphplan.effect_to_actions[*open_goal as usize]
                .iter()
                .filter(|&&a| graphplan.action_layer[a] == i - 1)
                .min_by_key(|&&a| (domain.actions[a].cost, a))
                .copied()
                .unwrap();
            let min_action = &domain.actions[min_action_idx];
            len += 1;

            // Add preconditions as new goals: skip those in the initial state or already marked
            for &precond in min_action.pre_cond.difference(&initial_facts) {
                if !marks[(i - 1) as usize].contains(&precond) {
                    let layer = *graphplan.facts.get(&precond).unwrap();
                    g.entry(layer).or_default().insert(precond);
                }
            }

            // Mark all effects of min_action at layers i and i-1
            for &add in &min_action.add_effects[0] {
                marks[i as usize].insert(add);
                marks[(i - 1) as usize].insert(add);
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
}

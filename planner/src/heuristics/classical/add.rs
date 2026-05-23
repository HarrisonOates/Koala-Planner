use super::*;
use std::collections::{HashMap, HashSet, VecDeque};

pub fn h_add(domain: &ClassicalDomain, state: &HashSet<u32>, goal: &HashSet<u32>) -> f32 {
    let n = domain.actions.len();
    let fact_to_actions = &domain.fact_to_actions;
    let mut precond_remaining: Vec<u32> = domain.precond_counts.clone();
    let mut precond_cost_sum: Vec<u32> = vec![0; n];

    let mut fact_cost: HashMap<u32, u32> = HashMap::new();
    let mut remaining_goals = goal.len();

    let mut queue: VecDeque<(u32, u32)> = VecDeque::new();
    for &f in state.iter() {
        fact_cost.insert(f, 0);
        if goal.contains(&f) {
            remaining_goals -= 1;
        }
        queue.push_back((f, 0));
    }

    // Fire zero-precondition actions immediately (applicable from any state)
    for (i, action) in domain.actions.iter().enumerate() {
        if action.pre_cond.is_empty() && !action.add_effects.is_empty() {
            let weight = 1u32;
            for &effect in action.add_effects[0].iter() {
                if !fact_cost.contains_key(&effect) {
                    fact_cost.insert(effect, weight);
                    if goal.contains(&effect) {
                        remaining_goals -= 1;
                    }
                    queue.push_back((effect, weight));
                }
            }
        }
    }

    // Event-driven propagation: process newly achieved facts, trigger dependent actions
    while let Some((fact_id, fact_cost_val)) = queue.pop_front() {
        if remaining_goals == 0 {
            break;
        }
        let f_idx = fact_id as usize;
        if f_idx < fact_to_actions.len() {
            for &action_idx in &fact_to_actions[f_idx] {
                if precond_remaining[action_idx] == 0 {
                    continue; // already fired
                }
                precond_cost_sum[action_idx] += fact_cost_val;
                precond_remaining[action_idx] -= 1;
                if precond_remaining[action_idx] == 0 {
                    let action = &domain.actions[action_idx];
                    if action.add_effects.is_empty() {
                        continue;
                    }
                    let weight = 1 + precond_cost_sum[action_idx];
                    for &effect in action.add_effects[0].iter() {
                        if !fact_cost.contains_key(&effect) {
                            fact_cost.insert(effect, weight);
                            if goal.contains(&effect) {
                                remaining_goals -= 1;
                            }
                            queue.push_back((effect, weight));
                        }
                    }
                }
            }
        }
    }

    if remaining_goals > 0 {
        return f32::INFINITY;
    }
    goal.iter().map(|f| fact_cost[f]).sum::<u32>() as f32
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
        let h = h_add(&domain, &HashSet::from([0]), &HashSet::from([4, 0]));
        assert_eq!(h, 6.0);
        let h = h_add(&domain, &HashSet::from([0]), &HashSet::from([4, 2]));
        assert_eq!(h, 8.0);
    }
    #[test]
    pub fn safety_test() {
        let domain = generate_domain();
        let h = h_add(&domain, &HashSet::from([0]), &HashSet::from([5]));
        assert_eq!(h, f32::INFINITY);
    }
    #[test]
    pub fn goal_awareness_test() {
        let domain = generate_domain();
        let h = h_add(&domain, &HashSet::from([0]), &HashSet::from([0]));
        assert_eq!(h, 0.0);
    }
}

#![allow(dead_code, unused_must_use)]
use crate::domain_description::ClassicalDomain;
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Debug, Clone)]
pub struct GraphPlan<'a> {
    /// fact_placed[fact_id] = layer at which the fact was first placed (u32::MAX = unplaced).
    pub fact_placed: Vec<u32>,
    pub depth: u32,
    domain: &'a ClassicalDomain,
    layer_to_actions: HashMap<u32, Vec<usize>>,
    layer_to_facts: HashMap<u32, Vec<u32>>,
    pub(crate) effect_to_actions: Vec<Vec<usize>>, // fact_id -> actions that produce it
    pub(crate) action_layer: Vec<u32>,             // action_idx -> layer (u32::MAX = unplaced)
}

impl<'a> GraphPlan<'a> {
    // returns the membership index for alternating facts and actions layer as a
    // tuple ((action_id->first occurance layer number), (fact_id -> first occurance index))
    // returns None if there is no solution
    pub fn build_graph(
        domain: &'a ClassicalDomain,
        state: &HashSet<u32>,
        goal: &HashSet<u32>,
    ) -> Option<GraphPlan<'a>> {
        let n_actions = domain.actions.len();
        let n_facts = domain.facts.count() as usize;

        // Per-action and per-fact placement layers (u32::MAX = unplaced)
        let mut action_placed: Vec<u32> = vec![u32::MAX; n_actions];
        let mut fact_placed: Vec<u32> = vec![u32::MAX; n_facts];

        let fact_to_actions = &domain.fact_to_actions;
        let mut precond_remaining: Vec<u32> = domain.precond_counts.clone();

        let mut layer_to_actions: HashMap<u32, Vec<usize>> = HashMap::new();
        let mut layer_to_facts: HashMap<u32, Vec<u32>> = HashMap::new();
        let mut effect_to_actions: Vec<Vec<usize>> = vec![vec![]; n_facts];
        let mut action_layer_vec: Vec<u32> = vec![u32::MAX; n_actions];

        // Seed with initial state facts at layer 0
        let mut queue: VecDeque<(u32, u32)> = VecDeque::new();
        for &f in state.iter() {
            let fi = f as usize;
            if fi < n_facts && fact_placed[fi] == u32::MAX {
                fact_placed[fi] = 0;
                layer_to_facts.entry(0).or_default().push(f);
                queue.push_back((f, 0));
            }
        }

        // Count goals not yet satisfied by the initial state
        let mut remaining_goals: usize = goal
            .iter()
            .filter(|&&f| (f as usize) >= n_facts || fact_placed[f as usize] == u32::MAX)
            .count();

        // Event-driven BFS: propagate facts through actions
        while let Some((fact_id, fact_layer)) = queue.pop_front() {
            if remaining_goals == 0 {
                break;
            }
            let f_idx = fact_id as usize;
            if f_idx >= n_facts {
                continue;
            }
            for &action_idx in &fact_to_actions[f_idx] {
                if action_placed[action_idx] != u32::MAX {
                    continue; // already placed
                }
                precond_remaining[action_idx] -= 1;
                if precond_remaining[action_idx] == 0 {
                    let action_layer = fact_layer + 1;
                    action_placed[action_idx] = action_layer;
                    action_layer_vec[action_idx] = action_layer;
                    layer_to_actions.entry(action_layer).or_default().push(action_idx);
                    let new_fact_layer = action_layer + 1;
                    let action = &domain.actions[action_idx];
                    if action.add_effects.is_empty() {
                        continue;
                    }
                    for &effect in &action.add_effects[0] {
                        let ei = effect as usize;
                        if ei < n_facts {
                            effect_to_actions[ei].push(action_idx);
                        }
                        if ei < n_facts && fact_placed[ei] == u32::MAX {
                            fact_placed[ei] = new_fact_layer;
                            layer_to_facts.entry(new_fact_layer).or_default().push(effect);
                            queue.push_back((effect, new_fact_layer));
                            if goal.contains(&effect) {
                                remaining_goals -= 1;
                            }
                        }
                    }
                }
            }
        }

        if remaining_goals > 0 {
            return None;
        }

        let depth = fact_placed.iter().filter(|&&l| l != u32::MAX).cloned().max().unwrap_or(0);

        Some(GraphPlan {
            fact_placed,
            depth,
            domain,
            layer_to_actions,
            layer_to_facts,
            effect_to_actions,
            action_layer: action_layer_vec,
        })
    }

    // computes the goals that are satisfied in each layer
    pub fn compute_goal_indices(&self, goal: &HashSet<u32>) -> HashMap<u32, HashSet<u32>> {
        let mut mapping: HashMap<u32, HashSet<u32>> = HashMap::new();
        let n = self.fact_placed.len();
        for &g in goal.iter() {
            let gi = g as usize;
            if gi < n {
                let layer = self.fact_placed[gi];
                if layer != u32::MAX {
                    mapping.entry(layer).or_default().insert(g);
                }
            }
        }
        mapping
    }

    pub fn get_action_layer(&self, index: u32) -> HashSet<usize> {
        if index % 2 == 0 {
            panic!("actions are odd layers")
        }
        self.layer_to_actions
            .get(&index)
            .map(|v| v.iter().cloned().collect())
            .unwrap_or_default()
    }

    pub fn get_fact_layer(&self, index: u32) -> HashSet<u32> {
        if index % 2 == 1 {
            panic!("facts are even layer")
        }
        self.layer_to_facts
            .get(&index)
            .map(|v| v.iter().cloned().collect())
            .unwrap_or_default()
    }
}

impl<'a> std::fmt::Display for GraphPlan<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "digraph G {{\n\trankdir=\"LR\"\n");
        for layer in 0..self.depth {
            if layer % 2 == 0 {
                let ids = self.get_fact_layer(layer);
                let facts: Vec<&String> =
                    ids.iter().map(|x| self.domain.facts.get_fact(*x)).collect();
                write!(f, "\tsubgraph cluster{} {{\n\t\tlabel=\"layer{}\"\n\t\tstyle=filled;\n\t\tcolor=lightgrey;\n", layer, layer);
                for (id, fact) in ids.iter().zip(facts.iter()) {
                    write!(f, "\t\t{} [label=\"{}\",shape=box]\n", id, fact);
                }
            } else {
                let ids = self.get_action_layer(layer);
                let actions: Vec<&String> =
                    ids.iter().map(|x| &self.domain.actions[*x].name).collect();
                write!(
                    f,
                    "\tsubgraph cluster{} {{\n\t\tlabel=\"layer{}\"\n",
                    layer, layer
                );
                for (id, action) in ids.iter().zip(actions.iter()) {
                    write!(f, "\t\t{} [label=\"{}\",shape=box]\n", id, action);
                }
            }
            write!(f, "\t}}\n");
        }
        write!(f, "}}\n");
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{domain_description::Facts, task_network::PrimitiveAction};

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
    pub fn graph_correctness_test() {
        let domain = generate_domain();
        let graphplan =
            GraphPlan::build_graph(&domain, &HashSet::from([0]), &HashSet::from([4])).unwrap();

        // Action layers (action_layer Vec)
        assert_eq!(graphplan.action_layer[0], 1);
        assert_eq!(graphplan.action_layer[1], 3);
        assert_eq!(graphplan.action_layer[2], 3);
        assert_eq!(graphplan.action_layer[3], 5);

        // Fact layers (fact_placed Vec)
        assert_eq!(graphplan.fact_placed[0], 0);
        assert_eq!(graphplan.fact_placed[1], 2);
        assert_eq!(graphplan.fact_placed[2], 4);
        assert_eq!(graphplan.fact_placed[3], 4);
        assert_eq!(graphplan.fact_placed[4], 6);

        assert_eq!(graphplan.depth, 6);
        assert_eq!(graphplan.get_action_layer(1), HashSet::from([0]));
        assert_eq!(graphplan.get_action_layer(3), HashSet::from([1, 2]));
        assert_eq!(graphplan.get_action_layer(5), HashSet::from([3]));
    }

    #[test]
    pub fn termination_test() {
        let mut domain = generate_domain();
        domain.actions[3].add_effects = vec![];
        let graphplan = GraphPlan::build_graph(&domain, &HashSet::from([0]), &HashSet::from([4]));
        assert_eq!(graphplan.is_none(), true);
    }
}

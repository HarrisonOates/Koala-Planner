use crate::domain_description::ClassicalDomain;
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Debug, Clone)]
pub struct GraphPlan<'a> {
    pub actions: HashMap<usize, u32>,
    pub facts: HashMap<u32, u32>,
    pub depth: u32,
    domain: &'a ClassicalDomain,
    layer_to_actions: HashMap<u32, Vec<usize>>,
    layer_to_facts: HashMap<u32, Vec<u32>>,
    pub(crate) effect_to_actions: Vec<Vec<usize>>, // fact_id -> actions that produce it
    pub(crate) action_layer: Vec<u32>,             // action_idx -> layer (Vec mirror of actions)
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

        // Reverse index: fact_id -> action indices that need it as a precondition
        let mut fact_to_actions: Vec<Vec<usize>> = vec![vec![]; n_facts];
        let mut precond_remaining: Vec<u32> = vec![0; n_actions];
        for (i, action) in domain.actions.iter().enumerate() {
            precond_remaining[i] = action.pre_cond.len() as u32;
            for &f in action.pre_cond.iter() {
                if (f as usize) < n_facts {
                    fact_to_actions[f as usize].push(i);
                }
            }
        }

        let mut layer_to_actions: HashMap<u32, Vec<usize>> = HashMap::new();
        let mut layer_to_facts: HashMap<u32, Vec<u32>> = HashMap::new();
        let mut facts_map: HashMap<u32, u32> = HashMap::new();
        let mut effect_to_actions: Vec<Vec<usize>> = vec![vec![]; n_facts];
        let mut action_layer_vec: Vec<u32> = vec![u32::MAX; n_actions];

        // Seed with initial state facts at layer 0
        let mut queue: VecDeque<(u32, u32)> = VecDeque::new();
        for &f in state.iter() {
            if (f as usize) < n_facts && fact_placed[f as usize] == u32::MAX {
                fact_placed[f as usize] = 0;
                facts_map.insert(f, 0);
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
                        if (effect as usize) < n_facts {
                            effect_to_actions[effect as usize].push(action_idx);
                        }
                        if (effect as usize) < n_facts && fact_placed[effect as usize] == u32::MAX {
                            fact_placed[effect as usize] = new_fact_layer;
                            facts_map.insert(effect, new_fact_layer);
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

        // All action indices must be present in the map (unplaced ones keep u32::MAX)
        let actions_map: HashMap<usize, u32> = action_placed
            .iter()
            .enumerate()
            .map(|(i, &l)| (i, l))
            .collect();
        // All fact IDs must be present in the map (unplaced ones keep u32::MAX)
        for f in domain.facts.get_all_ids() {
            facts_map.entry(f).or_insert(u32::MAX);
        }

        let depth = facts_map
            .values()
            .filter(|&&l| l != u32::MAX)
            .cloned()
            .max()
            .unwrap_or(0);

        Some(GraphPlan {
            actions: actions_map,
            facts: facts_map,
            depth,
            domain,
            layer_to_actions,
            layer_to_facts,
            effect_to_actions,
            action_layer: action_layer_vec,
        })
    }

    fn all_goals_satisfied(indices: &HashMap<u32, u32>, goal: &HashSet<u32>) -> bool {
        for fact in goal.iter() {
            let val = indices.get(fact).unwrap();
            if *val == u32::MAX {
                return false;
            }
        }
        return true;
    }

    // computes the goals that are satisfied in each layer
    pub fn compute_goal_indices(&self, goal: &HashSet<u32>) -> HashMap<u32, HashSet<u32>> {
        let mut mapping: HashMap<u32, HashSet<u32>> = HashMap::new();
        for &g in goal.iter() {
            if let Some(&layer) = self.facts.get(&g) {
                mapping.entry(layer).or_default().insert(g);
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
    use crate::{domain_description::Facts, heuristics::PrimitiveAction};

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
        let (actions, facts) = (graphplan.actions.clone(), graphplan.facts.clone());
        for action_id in 0..actions.len() {
            assert_eq!(actions.contains_key(&(action_id as usize)), true)
        }
        for fact_id in 0..facts.len() {
            assert_eq!(facts.contains_key(&(fact_id as u32)), true)
        }
        assert_eq!(*actions.get(&0).unwrap(), 1);
        assert_eq!(*actions.get(&1).unwrap(), 3);
        assert_eq!(*actions.get(&2).unwrap(), 3);
        assert_eq!(*actions.get(&3).unwrap(), 5);
        assert_eq!(*facts.get(&0).unwrap(), 0);
        assert_eq!(*facts.get(&1).unwrap(), 2);
        assert_eq!(*facts.get(&2).unwrap(), 4);
        assert_eq!(*facts.get(&3).unwrap(), 4);
        assert_eq!(*facts.get(&4).unwrap(), 6);
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

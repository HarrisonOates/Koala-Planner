#![allow(dead_code)]
use std::collections::HashSet;

use crate::task_network::PrimitiveAction;

use super::Facts;

#[derive(Debug)]
pub struct ClassicalDomain {
    pub facts: Facts,
    pub actions: Vec<PrimitiveAction>,
    pub fact_to_actions: Vec<Vec<usize>>,
    pub precond_counts: Vec<u32>,
}

impl ClassicalDomain {
    pub fn new(facts: Facts, actions: Vec<PrimitiveAction>) -> ClassicalDomain {
        let n_facts = facts.count() as usize;
        let n_actions = actions.len();
        let mut fact_to_actions: Vec<Vec<usize>> = vec![vec![]; n_facts];
        let mut precond_counts: Vec<u32> = vec![0; n_actions];
        for (i, action) in actions.iter().enumerate() {
            precond_counts[i] = action.pre_cond.len() as u32;
            for &f in action.pre_cond.iter() {
                if (f as usize) < n_facts {
                    fact_to_actions[f as usize].push(i);
                }
            }
        }
        ClassicalDomain { facts, actions, fact_to_actions, precond_counts }
    }

    pub fn delete_relax(&self) -> ClassicalDomain {
        let new_actions = self.actions.iter().map(|a| a.delete_relax()).collect();
        ClassicalDomain::new(self.facts.clone(), new_actions)
    }

    pub fn get_actions_by_index(&self, indices: HashSet<usize>) -> Vec<&PrimitiveAction> {
        self.actions
            .iter()
            .enumerate()
            .filter(|(i, _action)| indices.contains(i))
            .map(|(_i, action)| action)
            .collect()
    }

    pub fn get_fact(&self, index: u32) -> &String {
        self.facts.get_fact(index)
    }
}
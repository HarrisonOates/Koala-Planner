use std::collections::{HashMap, HashSet};

use crate::domain_description::ClassicalDomain;
use crate::relaxation::RelaxedComposition;
use crate::task_network::HTN;


pub type ClassicalHeuristic = fn(&ClassicalDomain, &HashSet<u32>, &HashSet<u32>) -> f32;

pub type HeuristicFn =
    Box<dyn Fn(&HTN, &HashSet<u32>, &RelaxedComposition, &HashMap<u32, u32>) -> f32>;

pub fn create_function_with_heuristic(h_input: ClassicalHeuristic) -> HeuristicFn {
    Box::new(move |tn, state, encoder, bijection| {
        let occurances = tn.count_tasks_with_frequency(); // Assuming this returns something iterable
        let task_ids: Vec<u32> = occurances
            .iter()
            .map(|(task, _)| *bijection.get(task).unwrap())
            .collect();
        let relaxed_state = encoder.compute_relaxed_state(&task_ids, state);
        let goal_state = encoder.compute_goal_state(&task_ids);
        let mut val = h_input(&encoder.domain, &relaxed_state, &goal_state);

        // Compensate for the repetition of tasks
        for (_, count) in occurances {
            if count > 1 {
                val += (count - 1) as f32;
            }
        }
        val
    })
}
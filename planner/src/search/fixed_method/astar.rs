#![allow(dead_code, unused_must_use)]
use super::*;
use crate::{
    domain_description::FONDProblem,
    relaxation::RelaxedComposition,
    search::StrongPolicy,
    task_network::HTN,
};
use priority_queue::PriorityQueue;
use search_node::{AStarStatus, SearchNode};
use search_space::SearchSpace;
use std::time::Instant;
use std::{
    cell::RefCell,
    collections::{BTreeMap, HashMap, HashSet},
    rc::Rc,
};
use weak_linearization::WeakLinearization;

pub enum AStarResult {
    Strong(StrongPolicy),
    Linear(WeakLinearization),
    NoSolution,
}

// different users of A* may want entirely different statistics
pub enum CustomStatistic {
    Value(u32),
    FloatValue(f64),
    List(Vec<u32>),
}
pub type CustomStatistics = BTreeMap<String, CustomStatistic>;

pub struct AStarStatistics {
    pub space: SearchSpace,
    pub goal_node: Option<Rc<RefCell<SearchNode>>>,
    pub custom_statistics: CustomStatistics,
}

impl std::fmt::Display for AStarStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        for (key, value) in self.custom_statistics.iter() {
            match value {
                CustomStatistic::Value(val) => writeln!(f, "{}: {}", key, val),
                CustomStatistic::List(vec) => writeln!(f, "{}: {:?}", key, vec),
                CustomStatistic::FloatValue(val) => writeln!(f, "{}: {}", key, val),
            };
        };
        writeln!(f, "# of search nodes: {}", self.space.total_nodes);
        writeln!(f, "# of explored nodes: {}", self.space.explored_nodes)
    }
}

impl CustomStatistic {
    pub fn accumulate(&mut self, y: u32) {
        match self {
            CustomStatistic::Value(x) => *x += y,
            CustomStatistic::List(v) => v.push(y),
            _ => panic!("Cannot accumulate any other type"),
        }
    }
}

pub fn calculate_ipc_score(statistics: &mut CustomStatistics, start_time: Instant) {
    let duration = start_time.elapsed().as_secs_f64();
    statistics.insert(
        String::from("duration (seconds)"),
        CustomStatistic::FloatValue(duration),
    );
    let ipc_score = if duration > 1.0 {
        1.0 - (duration.log(10.0) / (1800.0 as f64).log(10.0))
    } else {1.0};
    statistics.insert(
        String::from("IPC Score"),
        CustomStatistic::FloatValue(ipc_score),
    );
}

pub fn a_star_search(
    problem: &FONDProblem,
    heuristic_fn: impl Fn(&HTN, &HashSet<u32>, &RelaxedComposition, &HashMap<u32, u32>) -> f32,
    successor_fn: fn(
        &mut SearchSpace,
        Rc<RefCell<SearchNode>>,
    ) -> Vec<(u32, String, Option<String>, SearchNode)>,
    edge_weight_fn: fn() -> f32,
    goal_check_fn: fn(&FONDProblem, Rc<RefCell<SearchNode>>, &mut CustomStatistics) -> AStarResult,
) -> (AStarResult, AStarStatistics) {
    let start_time = Instant::now();
    let mut space = SearchSpace::new(
        problem.init_tn.clone(),
        problem.initial_state.clone(),
        problem,
    );
    space
        .initial_search_node
        .borrow_mut()
        .compute_h_value(&space, &heuristic_fn);
    space.initial_search_node.borrow_mut().g_value = Some(0.0);

    let mut custom_statistics: CustomStatistics = BTreeMap::new();

    let mut open = PriorityQueue::new();
    open.insert(space.initial_search_node.clone());

    while let Some(parent) = open.pop_least() {
        parent.borrow_mut().status = AStarStatus::Closed;
        space.explored_nodes += 1; // closed set increased in size by 1
        let result = goal_check_fn(problem, parent.clone(), &mut custom_statistics);
        match result {
            AStarResult::NoSolution => (),
            _ => {
                let duration = start_time.elapsed().as_secs_f64();
                custom_statistics.insert(
                    String::from("IPC Score"),
                    CustomStatistic::FloatValue(
                        1.0 - (duration.log(2.0) / (1800.0 as f64).log(2.0)),
                    ),
                );
                calculate_ipc_score(&mut custom_statistics, start_time);
                return (
                    result,
                    AStarStatistics {
                        space: space,
                        goal_node: Some(parent.clone()),
                        custom_statistics: custom_statistics,
                    },
                );
            }
        }
        let successors = successor_fn(&mut space, parent.clone());
        space.install_successors(parent.clone(), successors, false);
        for edge in parent.borrow().progressions.iter() {
            // Remove from open with old f value (before updating)
            if edge.next_node.borrow().status == AStarStatus::Open {
                open.remove(edge.next_node.clone());
            }

            {
                // succ_ref lifetime
                let mut succ_ref = edge.next_node.borrow_mut();
                match succ_ref.status {
                    AStarStatus::Open => {
                        if parent.borrow().g_value.unwrap() + edge_weight_fn()
                            < succ_ref.g_value.unwrap()
                        {
                            (*succ_ref).parent = Some(parent.clone());
                            (*succ_ref).g_value =
                                Some(parent.borrow().g_value.unwrap() + edge_weight_fn());
                            (*succ_ref).compute_h_value(&space, &heuristic_fn);
                        }
                    }
                    AStarStatus::Closed => {
                        if parent.borrow().g_value.unwrap() + edge_weight_fn()
                            < succ_ref.g_value.unwrap()
                        {
                            (*succ_ref).parent = Some(parent.clone());
                            (*succ_ref).g_value =
                                Some(parent.borrow().g_value.unwrap() + edge_weight_fn());
                            (*succ_ref).compute_h_value(&space, &heuristic_fn);
                            (*succ_ref).status = AStarStatus::Open;
                            space.explored_nodes -= 1; // closed set decreased in size by 1
                        }
                    }
                    AStarStatus::New => {
                        (*succ_ref).parent = Some(parent.clone());
                        (*succ_ref).g_value =
                            Some(parent.borrow().g_value.unwrap() + edge_weight_fn());
                        (*succ_ref).compute_h_value(&space, &heuristic_fn);
                        (*succ_ref).status = AStarStatus::Open;
                    }
                }
            } // succ_ref lifetime

            // Insert back into open with new f value
            if edge.next_node.borrow().status == AStarStatus::Open  {
                open.insert(edge.next_node.clone());
            }
        }
    }
    calculate_ipc_score(&mut custom_statistics, start_time);
    return (
        AStarResult::NoSolution,
        AStarStatistics {
            space: space,
            goal_node: None,
            custom_statistics: custom_statistics,
        },
    );
}

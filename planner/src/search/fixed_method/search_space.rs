#![allow(dead_code)]
use search_node::{AStarStatus, Edge, SearchNode};

use super::*;
use crate::{
    domain_description::FONDProblem,
    relaxation::{OutcomeDeterminizer, RelaxedComposition},
    task_network::HTN,
};
use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    rc::Rc,
};

pub struct SearchSpace {
    /*
        SearchNodes in the same bucket are *maybe* isomorphic
        SearchNodes in different buckets are *definitely not* isomorphic
    */
    pub maybe_isomorphic_buckets: HashMap<u32, Vec<Rc<RefCell<SearchNode>>>>,
    pub initial_search_node: Rc<RefCell<SearchNode>>,
    pub relaxed_domain: (RelaxedComposition, HashMap<u32, u32>),
    pub next_node_id: u32,
    pub explored_nodes: u32,
    pub total_nodes: u32,
}

impl SearchSpace {
    pub fn new(init_tn: HTN, init_state: HashSet<u32>, problem: &FONDProblem) -> SearchSpace {
        let node = Rc::new(RefCell::new(SearchNode::new(0, init_tn, init_state)));
        node.borrow_mut().status = AStarStatus::Open;
        let buckets = HashMap::from([(node.borrow().maybe_isomorphic_hash(), vec![node.clone()])]);
        let (outcome_det, bijection) = OutcomeDeterminizer::from_fond_problem(problem);
        let relaxed = RelaxedComposition::new(&outcome_det);
        SearchSpace {
            maybe_isomorphic_buckets: buckets,
            initial_search_node: node,
            relaxed_domain: (relaxed, bijection),
            next_node_id: 0,
            explored_nodes: 0,
            total_nodes: 1,
        }
    }

    /*
        Either finds an isomorphic node or creates a new one
    */
    pub fn find_isomorphic(&mut self, new_node: SearchNode) -> Rc<RefCell<SearchNode>> {
        let hash = new_node.maybe_isomorphic_hash();
        let ret = match self.maybe_isomorphic_buckets.get_mut(&hash) {
            Some(bucket) => {
                let mut ret = None;
                'find_isomorphic: for maybe_isomorphic_node in bucket.iter() {
                    if new_node.is_isomorphic(maybe_isomorphic_node.clone()) {
                        ret = Some(maybe_isomorphic_node.clone());
                        break 'find_isomorphic;
                    }
                }
                match ret {
                    Some(isomorphic_node) => {
                        // Found an isomorphic node
                        isomorphic_node
                    }
                    None => {
                        // No isomorphic node, add this to the bucket
                        self.total_nodes += 1;
                        if self.total_nodes % 200 == 0  {
                            println!("[DEBUG] Explored {} search nodes", self.total_nodes);
                        }
                        let ret = Rc::new(RefCell::new(new_node));
                        bucket.push(ret.clone());
                        ret
                    }
                }
            }
            None => {
                // No bucket exists for this hash, so make one
                self.total_nodes += 1;
                if self.total_nodes % 200 == 0  {
                    println!("[DEBUG] Explored {} search nodes", self.total_nodes);
                }
                let ret = Rc::new(RefCell::new(new_node));
                self.maybe_isomorphic_buckets
                    .insert(hash, vec![ret.clone()]);
                ret
            }
        };
        ret
    }

    pub fn install_successors(
        &mut self,
        node: Rc<RefCell<SearchNode>>,
        successors: Vec<(u32, String, Option<String>, SearchNode)>,
        loop_detection_enbaled: bool,
    ) {
        for (id, task_name, method_name, successor) in successors {
            let successor_in_graph: Rc<RefCell<SearchNode>>;
            if loop_detection_enbaled {
                successor_in_graph = self.find_isomorphic(successor);
            } else {
                successor_in_graph = Rc::new(RefCell::new(successor));
                self.total_nodes += 1;
            }
            node.borrow_mut().progressions.push(Edge {
                task_id: id,
                task_name: task_name,
                method_name: method_name,
                next_node: successor_in_graph.clone(),
            });
        }
    }

    pub fn to_string(&self, problem: &FONDProblem) -> String {
        let mut node_number = 0;
        SearchSpace::to_string_helper(
            problem,
            self.initial_search_node.clone(),
            &mut HashMap::new(),
            String::from(""),
            &mut node_number,
        )
    }

    pub fn to_string_helper(
        problem: &FONDProblem,
        current: Rc<RefCell<SearchNode>>,
        visited: &mut HashMap<u32, u32>, // node ID to line number in printout
        indentation: String,
        node_number: &mut u32,
    ) -> String {
        let lookup = visited.get(&current.borrow().unique_id);
        if let Some(prev_number) = lookup {
            return format!("{}GOTO NODE_{}", indentation, *prev_number);
        }
        *node_number += 1;
        visited.insert(current.borrow().unique_id, *node_number);
        let mut result = format!(
            "{}NODE_{} {}",
            indentation,
            *node_number,
            current.borrow().to_string(problem)
        );
        for edge in current.borrow().progressions.iter() {
            result = format!(
                "{}\n{}",
                result,
                SearchSpace::to_string_helper(
                    problem,
                    edge.next_node.clone(),
                    visited,
                    format!("{}|  ", indentation),
                    node_number
                )
            );
        }
        return result;
    }
}

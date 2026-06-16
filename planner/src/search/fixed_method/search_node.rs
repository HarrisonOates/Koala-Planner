#![allow(dead_code)]
use search_space::SearchSpace;

use super::*;
use crate::{
    domain_description::FONDProblem,
    relaxation::RelaxedComposition,
    task_network::{Applicability, Task, HTN},
};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    rc::Rc,
};

#[derive(PartialEq, Eq)]
pub enum AStarStatus {
    Closed,
    Open,
    New,
}
pub struct Edge {
    pub task_id: u32,
    pub task_name: String,
    pub method_name: Option<String>,
    pub next_node: Rc<RefCell<SearchNode>>,
}
pub struct SearchNode {
    pub tn: HTN,
    pub state: HashSet<u32>,
    pub progressions: Vec<Edge>,
    pub unique_id: u32,
    pub status: AStarStatus,
    pub parent: Option<Rc<RefCell<SearchNode>>>,
    pub g_value: Option<f32>,
    pub h_value: Option<f32>,
    pub goal_tested: bool,
}

impl SearchNode {
    pub fn new(next_node_id: u32, tn: HTN, state: HashSet<u32>) -> SearchNode {
        return SearchNode {
            tn: tn,
            state: state,
            progressions: vec![],
            unique_id: next_node_id,
            status: AStarStatus::New,
            parent: None,
            g_value: None,
            h_value: None,
            goal_tested: false,
        };
    }

    /*
        Same hash -> *maybe* isomorphic
        Different hash -> *definitely not* isomorphic
    */
    pub fn maybe_isomorphic_hash(&self) -> u32 {
        let mut hasher = DefaultHasher::new();
        let mut sorted_set: Vec<_> = self.state.iter().collect();
        sorted_set.sort();
        for &elem in &sorted_set {
            elem.hash(&mut hasher);
        }
        self.tn.count_tasks().hash(&mut hasher);
        hasher.finish() as u32
    }

    pub fn is_isomorphic(&self, other: Rc<RefCell<SearchNode>>) -> bool {
        self.state == other.borrow().state && HTN::is_isomorphic(&self.tn, &other.borrow().tn)
    }

    pub fn to_string(&self, problem: &FONDProblem) -> String {
        format!("{} {}",
            SearchNode::to_string_structure(&self.state, &self.tn, problem),
            if self.goal_tested {"GOAL_TESTED"} else {""}
        )
    }

    pub fn to_string_tn(tn: &HTN, _problem: &FONDProblem) -> Vec<String> {
        let uncon_ids = tn.get_unconstrained_tasks();
        let mut sorted_uncon_ids: Vec<&u32> = uncon_ids.iter().collect();
        sorted_uncon_ids.sort_by(|a, b| a.cmp(b));
        let mut uncon_names = Vec::new();
        for id in uncon_ids {
            let name = tn.get_task(id).borrow().get_name();
            uncon_names.push(format!("{}:{}", name, id));
        }
        uncon_names
    }

    pub fn to_string_structure(state: &HashSet<u32>, tn: &HTN, problem: &FONDProblem) -> String {
        // Sorting is needed so order is predictable (for tests to pass)
        let mut sorted_state: Vec<&u32> = state.iter().collect();
        sorted_state.sort_by(|a, b| a.cmp(b));
        let mut state_names = Vec::new();
        for id in sorted_state {
            let name = problem.facts.get_fact(*id);
            state_names.push(name);
        }
        format!(
            "state={:?} uncon={:?}",
            state_names,
            SearchNode::to_string_tn(tn, problem)
        )
    }

    pub fn compute_h_value(
        &mut self,
        s: &SearchSpace,
        h: &impl Fn(&HTN, &HashSet<u32>, &RelaxedComposition, &HashMap<u32, u32>) -> f32,
    ) {
        self.h_value = Some(h(
            &self.tn,
            &self.state,
            &s.relaxed_domain.0,
            &s.relaxed_domain.1,
        ));
    }

    pub fn f_value(&self) -> f32 {
        if let (Some(g), Some(h)) = (self.g_value, self.h_value) {
            g + h
        } else {
            panic!("Cannot compute f value of a node unless both g and h have been instantiated.")
        }
    }

    pub fn find_edge(&self, child: &Rc<RefCell<SearchNode>>) -> &Edge {
        for edge in &self.progressions {
            if Rc::ptr_eq(&edge.next_node, child) {
                return edge;
            }
        }
        panic!("This function should never be called unless self == child.parent");
    }
}

impl Ord for SearchNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.f_value()
            .partial_cmp(&other.f_value())
            .expect("Unable to compare the f values of two search nodes.")
    }
}

impl PartialOrd for SearchNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.f_value().partial_cmp(&other.f_value())
    }
}

impl Eq for SearchNode {}

impl PartialEq for SearchNode {
    fn eq(&self, other: &Self) -> bool {
        self.f_value() == other.f_value()
    }
}

pub fn get_successors_systematic(
    space: &mut SearchSpace,
    node: Rc<RefCell<SearchNode>>,
) -> Vec<(u32, String, Option<String>, SearchNode)> {
    let mut result = vec![];

    let unconstrained = node.borrow().tn.get_unconstrained_tasks();
    let (compounds, actions) = node.borrow().tn.separate_tasks(&unconstrained);

    // Expand a compound task if there is one
    if let Some(id) = compounds.first() {
        if let Task::Compound(cmp) = &*node.borrow().tn.get_task(*id).borrow() {
            for method in cmp.methods.iter() {
                let new_tn = node.borrow().tn.decompose(*id, method);
                space.next_node_id += 1;
                let node = SearchNode::new(space.next_node_id, new_tn, node.borrow().state.clone());
                result.push((*id, cmp.name.clone(), Some(method.name.clone()), node));
            }
        }
    }

    // If a compound task was progressed, exit
    if !result.is_empty() {
        return result;
    }

    // If not, expand primitive tasks
    'prim_loop: for prim in actions.iter() {
        if let Task::Primitive(act) = &*node.borrow().tn.get_task(*prim).borrow() {
            if !act.is_applicable(&node.borrow().state) {
                continue 'prim_loop;
            }
            let new_tn = node.borrow().tn.apply_action(*prim);
            let outcomes = act.transition(&node.borrow().state);
            for outcome in outcomes {
                space.next_node_id += 1;
                let node = SearchNode::new(space.next_node_id, new_tn.clone(), outcome);
                result.push((*prim, act.name.clone(), None, node));
            }
        }
    }

    return result;
}

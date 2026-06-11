#![allow(dead_code)]
use super::search_node::SearchNode;
use crate::{domain_description::FONDProblem, task_network::HTN};
use std::{
    cell::RefCell,
    collections::HashSet,
    rc::Rc,
};

pub struct WeakLinearization {
    linearization: Vec<(HashSet<u32>, HTN)>,
}

impl WeakLinearization {
    pub fn new() -> Self {
        WeakLinearization {
            linearization: Vec::new(),
        }
    }

    pub fn push(&mut self, node: Rc<RefCell<SearchNode>>) {
        self.linearization
            .push((node.borrow().state.clone(), node.borrow().tn.clone()));
    }

    pub fn to_string(&self, problem: &FONDProblem) -> String {
        let mut ret = String::from("Linearization:");
        for (state, tn) in self.linearization.iter() {
            ret = format!(
                "{ret}\n{}",
                SearchNode::to_string_structure(&state, &tn, problem)
            );
        }
        ret
    }

    pub fn build(&mut self, node: Rc<RefCell<SearchNode>>) {
        // Insertion into index 0 of a vector is inefficient, however, weak LD solutions
        // are only used for testing purposes, so this code will not be used in experiments
        // TODO for future, use a better data structure, or store the elements in reverse
        // order
        self.linearization
            .insert(0, (node.borrow().state.clone(), node.borrow().tn.clone()));
        if let Some(parent) = node.borrow().parent.clone() {
            self.build(parent.clone());
        }
    }
}

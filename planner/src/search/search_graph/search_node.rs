#![allow(dead_code)]
use std::{
    collections::{HashMap, HashSet},
    rc::Rc,
};

use super::*;
use super::{HeuristicType, HTN};

use crate::{heuristics::*, relaxation::RelaxedComposition};

#[derive(Debug)]
pub struct SearchGraphNode {
    pub parents: Option<Vec<u32>>,
    pub tn: Rc<HTN>,
    pub state: Rc<HashSet<u32>>,
    pub connections: Option<NodeConnections>,
    pub cost: f32,
    pub status: NodeStatus,
    pub depth: u16,
    pub success_probability: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum NodeStatus {
    Solved,
    OnGoing,
    Failed,
}

impl NodeStatus {
    pub fn is_terminal(&self) -> bool {
        match self {
            Self::Failed => true,
            Self::Solved => true,
            Self::OnGoing => false,
        }
    }
}

impl SearchGraphNode {
    pub fn mark(&mut self, i: u32) {
        self.clear_marks();
        self.connections.as_mut().unwrap().mark(i)
    }
    pub fn get_marked_connection(&self) -> Option<&Connector> {
        for item in self.connections.as_ref().unwrap().children.iter() {
            if item.is_marked {
                return Some(item);
            }
        }
        None
    }
    pub fn clear_marks(&mut self) {
        self.connections.as_mut().unwrap().clear_marks()
    }

    pub fn has_children(&self) -> bool {
        match self.connections {
            Some(_) => true,
            None => false,
        }
    }

    pub fn add_parent(&mut self, _id: u32) {
        // no-op: parent lists for cycle detection are set at node creation
    }

    pub fn is_terminal(&self) -> bool {
        self.status.is_terminal()
    }

    pub fn is_goal(&self) -> bool {
        self.tn.is_empty()
    }

    pub fn h_val(
        tn: &HTN,
        state: &HashSet<u32>,
        encoder: &RelaxedComposition,
        bijection: &HashMap<u32, u32>,
        h_type: &HeuristicType,
    ) -> f32 {
        let occurances = tn.count_tasks_with_frequency();
        let task_ids = occurances
            .iter()
            .map(|(task, _)| *bijection.get(task).unwrap())
            .collect();
        let relaxed_state = encoder.compute_relaxed_state(&task_ids, state);
        let goal_state = encoder.compute_goal_state(&task_ids);
        let mut val = match h_type {
            HeuristicType::HFF    => h_ff   (&encoder.domain, &relaxed_state, &goal_state),
            HeuristicType::HAdd   => h_add  (&encoder.domain, &relaxed_state, &goal_state),
            HeuristicType::HMax   => h_max  (&encoder.domain, &relaxed_state, &goal_state),
            HeuristicType::HLMCut => h_lmcut(&encoder.domain, &relaxed_state, &goal_state),
        };

        // Compensate for the repetition of tasks
        for (_, count) in occurances {
            if count > 1 {
                val += (count - 1) as f32
            }
        }
        val
    }

    /// Incremental LM-cut: warm-start from `parent_cuts`, dropping any cut
    /// that contains `exclude_action` (the method's classical action index).
    /// For use on the direct compound successor of an assigned compound node
    /// when no primitives intervened — admissible by Pommerening & Helmert 2013.
    pub fn h_val_lmcut_incremental(
        tn: &HTN,
        state: &HashSet<u32>,
        encoder: &RelaxedComposition,
        bijection: &HashMap<u32, u32>,
        parent_cuts: &[(Vec<usize>, u32)],
        exclude_action: usize,
    ) -> (f32, LandmarkCuts) {
        let occurances = tn.count_tasks_with_frequency();
        let task_ids: Vec<u32> = occurances
            .iter()
            .map(|(task, _)| *bijection.get(task).unwrap())
            .collect();
        let relaxed_state = encoder.compute_relaxed_state(&task_ids, state);
        let goal_state = encoder.compute_goal_state(&task_ids);
        let (mut val, landmarks) =
            h_lmcut_incremental(&encoder.domain, &relaxed_state, &goal_state, parent_cuts, exclude_action);
        for (_, count) in occurances {
            if count > 1 {
                val += (count - 1) as f32;
            }
        }
        (val, landmarks)
    }

    /// Like `h_val`, but for LM-cut also returns the discovered landmark cuts
    /// (empty for all other heuristics).  Used to store cuts on Compound reach
    /// nodes for incremental warm-starting.
    pub fn h_val_with_landmarks(
        tn: &HTN,
        state: &HashSet<u32>,
        encoder: &RelaxedComposition,
        bijection: &HashMap<u32, u32>,
        h_type: &HeuristicType,
    ) -> (f32, LandmarkCuts) {
        let occurances = tn.count_tasks_with_frequency();
        let task_ids: Vec<u32> = occurances
            .iter()
            .map(|(task, _)| *bijection.get(task).unwrap())
            .collect();
        let relaxed_state = encoder.compute_relaxed_state(&task_ids, state);
        let goal_state = encoder.compute_goal_state(&task_ids);
        let (mut val, landmarks) = match h_type {
            HeuristicType::HLMCut => h_lmcut_full(&encoder.domain, &relaxed_state, &goal_state),
            HeuristicType::HFF    => (h_ff   (&encoder.domain, &relaxed_state, &goal_state), vec![]),
            HeuristicType::HAdd   => (h_add  (&encoder.domain, &relaxed_state, &goal_state), vec![]),
            HeuristicType::HMax   => (h_max  (&encoder.domain, &relaxed_state, &goal_state), vec![]),
        };
        for (_, count) in occurances {
            if count > 1 {
                val += (count - 1) as f32;
            }
        }
        (val, landmarks)
    }

}

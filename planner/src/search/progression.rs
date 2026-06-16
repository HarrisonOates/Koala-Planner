use super::*;
use std::{
    collections::HashSet,
    rc::Rc,
};

pub fn progress(tn: Rc<HTN>, state: Rc<HashSet<u32>>) -> Vec<NodeExpansion> {
    if tn.is_goal() {
        return vec![];
    }
    let unconstrained = tn.get_unconstrained_tasks();
    let (abstract_tasks, primitive_tasks) = tn.separate_tasks(&unconstrained);
    let mut expansions = vec![];
    
    // Following Alg. 3 of Höller et al. (2020): while an unconstrained abstract
    // task remains, decompose one of them and progress no action. Only once no
    // unconstrained abstract task is left do we progress the primitive tasks.
    match abstract_tasks.first() {
        // The choice of which abstract task to decompose is immaterial to the
        // resulting solution, so a single task is fixed arbitrarily; branching
        // occurs only over its methods.
        Some(abstract_id) => {
            if let Task::Compound(CompoundTask { name, methods }) =
                &*tn.get_task(*abstract_id).borrow()
            {
                for method in methods.iter() {
                    let new_tn = Rc::new(tn.decompose(*abstract_id, method));
                    expansions.push(NodeExpansion {
                        connection_label: ConnectionLabel::Decomposition(
                            name.clone(),
                            method.name.clone(),
                        ),
                        tn: new_tn,
                        states: vec![state.clone()]
                    });
                }
            }
        }
        // No unconstrained abstract task remains: progress the unconstrained
        // primitive tasks. A non-deterministic action yields one successor
        // state per outcome, bundled together as a single (AND) expansion.
        None => {
            for p in primitive_tasks.iter() {
                if let Task::Primitive(a) = &*tn.get_task(*p).borrow() {
                    if a.is_applicable(state.as_ref()) {
                        let new_tn = Rc::new(tn.apply_action(*p));
                        let new_states = a
                            .transition(state.as_ref())
                            .into_iter()
                            .map(Rc::new)
                            .collect();
                        expansions.push(NodeExpansion {
                            connection_label: ConnectionLabel::Execution(
                                a.name.clone(),
                                a.cost,
                            ),
                            tn: new_tn,
                            states: new_states
                        });
                    }
                }
            }
        }
    }
    expansions
}

#[derive(Debug)]
pub struct NodeExpansion {
    pub connection_label: ConnectionLabel,
    pub tn: Rc<HTN>,
    pub states: Vec<Rc<HashSet<u32>>>
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConnectionLabel {
    Execution(String, u32),
    // task name - method name
    Decomposition(String, String),
}

impl ConnectionLabel {
    pub fn is_decomposition(&self) -> bool {
        match &self {
            ConnectionLabel::Decomposition(_, _) => true,
            _ => false,
        }
    }

    pub fn get_label(&self) -> String {
        match self {
            Self::Execution(name, _) => name.clone(),
            Self::Decomposition(name, method) => name.clone() + &format!("_{}", method),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{domain_description::DomainTasks, task_network::{Method, PrimitiveAction}};
    use std::collections::{BTreeSet, HashMap};

    #[test]
    pub fn expansion_correctness_test() {
        let p1 = Task::Primitive(PrimitiveAction::new(
            "p1".to_string(),
            1,
            HashSet::from([0]),
            vec![HashSet::from([1]), HashSet::from([2, 4])],
            vec![HashSet::from([3]), HashSet::new()],
        ));
        let p2 = Task::Primitive(PrimitiveAction::new(
            "p2".to_string(),
            1,
            HashSet::from([1, 2, 4]),
            vec![HashSet::from([1])],
            vec![HashSet::from([3])],
        ));
        let p3 = Task::Primitive(PrimitiveAction::new(
            "p3".to_string(),
            1,
            HashSet::from([0, 3]),
            vec![HashSet::from([1])],
            vec![HashSet::from([3])],
        ));
        let p4 = Task::Primitive(PrimitiveAction::new(
            "p4".to_string(),
            1,
            HashSet::from([4]),
            vec![HashSet::from([2])],
            vec![HashSet::new()],
        ));
        let t1 = Task::Compound(CompoundTask::new("t1".to_string(), vec![]));
        let domain = Rc::new(DomainTasks::new(vec![p1, p2, p3, p4, t1]));
        let m1 = Method::new(
            "m1".to_string(),
            HTN::new(
                BTreeSet::from([1, 2]),
                vec![(1, 2)],
                domain.clone(),
                HashMap::from([(1, domain.get_id("p1")), (2, domain.get_id("p2"))]),
            ),
        );
        let m2 = Method::new(
            "m2".to_string(),
            HTN::new(
                BTreeSet::from([1, 2]),
                vec![],
                domain.clone(),
                HashMap::from([(1, domain.get_id("p3")), (2, domain.get_id("p4"))]),
            ),
        );
        let id = domain.get_id("t1");
        let domain = domain.add_methods(vec![(id, m1), (id, m2)]);
        let tn = HTN::new(
            BTreeSet::from([1, 2, 3, 4]),
            vec![(1, 4), (2, 4), (3, 4)],
            domain.clone(),
            HashMap::from([
                (1, domain.get_id("p1")),
                (2, domain.get_id("t1")),
                (3, domain.get_id("p3")),
                (4, domain.get_id("p4")),
            ]),
        );
        let state = Rc::new(HashSet::from([0, 3]));
        let expansion = progress(Rc::new(tn), Rc::clone(&state));
        assert_eq!(expansion.len(), 2); // m1 and m2

        let tn2 = HTN::new(
            BTreeSet::from([1, 4]),
            vec![(1, 4)],
            domain.clone(),
            HashMap::from([
                (1, domain.get_id("p1")),
                (4, domain.get_id("p4")),
            ]),
        );
        let expansion2 = progress(Rc::new(tn2), Rc::clone(&state));
        assert_eq!(expansion2.len(), 1); // two states, resulting from p1 and p2 in one AND node
        let p1_expansion = &expansion2[0];
        // p1 is an execution edge, not a decomposition
        assert!(matches!(
            p1_expansion.connection_label,
            ConnectionLabel::Execution(_, _)
        ));
        // p1 is non-deterministic with two outcomes, bundled together
        assert_eq!(p1_expansion.states.len(), 2);
        // p1 applied to state {0, 3}:
        //   outcome 1: add {1}, del {3} => {0, 1}
        //   outcome 2: add {2, 4}, del {}  => {0, 2, 3, 4}
        let resulting_states: Vec<HashSet<_>> = p1_expansion
            .states
            .iter()
            .map(|s| s.as_ref().clone())
            .collect();
        assert!(resulting_states.contains(&HashSet::from([0, 1])));
        assert!(resulting_states.contains(&HashSet::from([0, 2, 3, 4])));
    }
}

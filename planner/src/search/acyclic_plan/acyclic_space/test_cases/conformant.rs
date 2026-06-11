use crate::domain_description::FONDProblem;
use crate::search::acyclic_plan::acyclic_space::AOStarSearch;
use crate::task_network::{PrimitiveAction, Task, HTN};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

#[cfg(test)]
#[test]
pub fn conformant_test() {
    use crate::domain_description::{DomainTasks, Facts};
    use std::collections::BTreeSet;

    let p1 = Task::Primitive(PrimitiveAction::new(
        "p1".to_string(),
        1,
        HashSet::from([0]),
        vec![HashSet::from([1])],
        vec![HashSet::from([0])],
    ));
    let p2 = Task::Primitive(PrimitiveAction::new(
        "p2".to_string(),
        1,
        HashSet::from([0]),
        vec![HashSet::from([1]), HashSet::from([1, 4])],
        vec![HashSet::from([2]), HashSet::from([3])],
    ));
    let p3 = Task::Primitive(PrimitiveAction::new(
        "p3".to_string(),
        1,
        HashSet::from([1]),
        vec![HashSet::new()],
        vec![HashSet::new()],
    ));
    let domain = Rc::new(DomainTasks::new(vec![p1, p2, p3]));
    let tn = HTN::new(
        BTreeSet::from([1, 2, 3]),
        vec![(1, 3), (2, 3)],
        domain.clone(),
        HashMap::from([(1, 0), (2, 1), (3, 2)]),
    );
    let mut problem = FONDProblem {
        facts: Facts::new(vec![
            "1".to_string(),
            "2".to_string(),
            "3".to_string(),
            "4".to_string(),
            "5".to_string(),
        ]),
        tasks: domain,
        initial_state: HashSet::from([0]),
        init_tn: tn,
    };
    problem.collapse_tn();
    let (result, _) = AOStarSearch::run(&problem, crate::search::acyclic_plan::HeuristicType::HAdd);
    assert_eq!(result.is_success(), true);
}

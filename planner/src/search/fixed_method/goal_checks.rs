#![allow(dead_code)]
use astar::AStarResult;
use search_node::{Edge, SearchNode};
use weak_linearization::WeakLinearization;

use super::astar::{CustomStatistic, CustomStatistics};
use super::*;
use crate::{
    domain_description::FONDProblem,
    search::{
        acyclic_plan::{PolicyNode, PolicyOutput, StrongPolicy},
        AOStarSearch, HeuristicType, SearchResult,
    },
    task_network::{Applicability, PrimitiveAction, Task, HTN},
};
use std::{
    cell::RefCell,
    collections::{BTreeSet, HashMap, HashSet},
    rc::Rc,
};

pub fn is_goal_weak_ld(
    _problem: &FONDProblem,
    leaf_node: Rc<RefCell<SearchNode>>,
    _custom_statistics: &mut CustomStatistics,
) -> AStarResult {
    if leaf_node.borrow().tn.is_empty() {
        let mut lin = WeakLinearization::new();
        lin.build(leaf_node.clone());
        return AStarResult::Linear(lin);
    } else {
        return AStarResult::NoSolution;
    }
}

pub enum TaggedTask {
    Primitive(NewID),
    Compound(OldID),
}

pub fn is_goal_strong_od(
    problem: &FONDProblem,
    leaf_node: Rc<RefCell<SearchNode>>,
    custom_statistics: &mut CustomStatistics,
) -> AStarResult {
    if !leaf_node.clone().borrow().tn.is_empty() {
        return AStarResult::NoSolution;
    }

    // a weak LD solution was found and will be attempted
    custom_statistics
        .entry(String::from("# of attempted weak LD solutions"))
        .or_insert(CustomStatistic::Value(0))
        .accumulate(1);

    // construct new FONDProblem for the AO* subproblem
    let mut sub_problem = FONDProblem {
        facts: problem.facts.clone(),
        tasks: problem.tasks.clone(),
        initial_state: problem.initial_state.clone(),
        init_tn: deorder(leaf_node.clone()),
    };

    // make initial task network just one abstract task
    sub_problem.collapse_tn();

    // call AO* algorithm
    let (solution, stats) = AOStarSearch::run(&sub_problem, HeuristicType::HAdd);

    // accumulate total subroutine search node count
    custom_statistics
        .entry(String::from("# of search nodes in all AO* calls"))
        .or_insert(CustomStatistic::Value(0))
        .accumulate(stats.search_nodes);
    leaf_node.borrow_mut().goal_tested = true;

    match solution {
        SearchResult::Success(policy) => {
            custom_statistics.insert(
                String::from("makespan"),
                CustomStatistic::Value(policy.makespan as u32),
            );
            AStarResult::Strong(policy)
        }
        SearchResult::NoSolution => AStarResult::NoSolution,
    }
}

/// Strong LD ("linear dovetailing") goal check.
///
/// Returns `Strong` if the weak linearization produced by A* is a strong plan —
/// i.e. every action in the fixed sequence is applicable regardless of which
/// nondeterministic outcomes occurred in all previous steps.
///
/// Verification by forward propagation: maintain the set of all reachable states
/// and check that each primitive action is applicable in every one of them.
pub fn is_goal_strong_ld(
    problem: &FONDProblem,
    leaf_node: Rc<RefCell<SearchNode>>,
    _custom_statistics: &mut CustomStatistics,
) -> AStarResult {
    if !leaf_node.borrow().tn.is_empty() {
        return AStarResult::NoSolution;
    }

    struct PathStep {
        parent_state: HashSet<u32>,
        parent_tn: HTN,
        task_name: String,
        method_name: Option<String>,
        action: Option<PrimitiveAction>,
    }

    // Walk from leaf to root, collecting one step per edge.
    let mut steps: Vec<PathStep> = Vec::new();
    let mut child = leaf_node.clone();
    let mut parent = child.borrow().parent.clone();

    while let Some(parent_rc) = parent {
        {
            let par = parent_rc.borrow();
            let edge = par.find_edge(&child);
            let action: Option<PrimitiveAction> = if edge.method_name.is_none() {
                let task_cell = par.tn.get_task(edge.task_id);
                let task_guard = task_cell.borrow();
                if let Task::Primitive(a) = &*task_guard {
                    Some(a.clone())
                } else {
                    unreachable!("primitive edge must point to a primitive task")
                }
            } else {
                None
            };
            steps.push(PathStep {
                parent_state: par.state.clone(),
                parent_tn: par.tn.clone(),
                task_name: edge.task_name.clone(),
                method_name: edge.method_name.clone(),
                action,
            });
        }
        child = parent_rc;
        parent = child.borrow().parent.clone();
    }
    steps.reverse(); // root-to-leaf order

    // Forward propagation: track all reachable states (stored as sorted vecs
    // for deduplication) and check that each primitive is universally applicable.
    let mut reachable: HashSet<Vec<u32>> = {
        let mut init: Vec<u32> = problem.initial_state.iter().copied().collect();
        init.sort();
        let mut s = HashSet::new();
        s.insert(init);
        s
    };

    for step in &steps {
        let Some(action) = &step.action else { continue };
        let mut next: HashSet<Vec<u32>> = HashSet::new();
        for sv in &reachable {
            let state: HashSet<u32> = sv.iter().copied().collect();
            if !action.is_applicable(&state) {
                return AStarResult::NoSolution;
            }
            for ns in action.transition(&state) {
                let mut nsv: Vec<u32> = ns.into_iter().collect();
                nsv.sort();
                next.insert(nsv);
            }
        }
        reachable = next;
    }

    // All actions applicable in all reachable states — build the linear policy.
    let transitions = steps
        .iter()
        .map(|step| {
            let state: HashSet<String> = step
                .parent_state
                .iter()
                .map(|&id| problem.facts.get_fact(id).clone())
                .collect();
            let pn = PolicyNode {
                tn: Rc::new(step.parent_tn.clone()),
                state,
            };
            let po = PolicyOutput {
                task: step.task_name.clone(),
                method: step.method_name.clone().unwrap_or_else(|| "ε".to_string()),
            };
            (pn, po)
        })
        .collect();

    AStarResult::Strong(StrongPolicy {
        transitions,
        makespan: steps.len() as u16,
        success_probability: 1.0,
    })
}

type NewID = u32; // ID of a task in the new HTN which we are building
type OldID = u32; // ID of a task in any HTN inside the search space
type TaskName = u32; // Actual task names (same for all HTNs)

pub fn deorder(leaf_node: Rc<RefCell<SearchNode>>) -> HTN {
    // data structures needed for the task network we're building
    let domain = leaf_node.borrow().tn.domain.clone();
    let mut tasks: BTreeSet<NewID> = BTreeSet::new();
    let mut alpha: HashMap<NewID, TaskName> = HashMap::new();
    let mut orderings: Vec<(NewID, NewID)> = Vec::new();

    // data structures to map IDs between our task network and the ones in the search space
    let mut equivalent_ids: HashMap<OldID, NewID> = HashMap::new();
    let mut compound_mapping: HashMap<OldID, Vec<TaggedTask>> = HashMap::new();

    // not yet handling edge case where initial search node *is* the leaf node
    let mut child = leaf_node.clone();
    let mut parent = child.borrow().parent.clone();
    let mut next_new_id: NewID = 0;

    while parent != None {
        let parent_unwrap = parent.unwrap();
        {
            // parent_node lifetime
            let parent_node = parent_unwrap.borrow();
            let edge: &Edge = parent_node.find_edge(&child);
            let old_id: OldID = edge.task_id;

            match &edge.method_name {
                Some(_name) => {
                    compound_mapping.insert(old_id, Vec::new());
                    let child_set: HashSet<OldID> = child.borrow().tn.get_task_id_set();
                    let parent_set: HashSet<OldID> = parent_node.tn.get_task_id_set();
                    let method_tasks: HashSet<OldID> =
                        child_set.difference(&parent_set).cloned().collect();
                    for method_task in method_tasks {
                        match *child.borrow().tn.get_task(method_task).borrow() {
                            Task::Primitive(_) => compound_mapping.get_mut(&old_id).unwrap().push(
                                TaggedTask::Primitive(*equivalent_ids.get(&method_task).unwrap()),
                            ),
                            Task::Compound(_) => compound_mapping
                                .get_mut(&old_id)
                                .unwrap()
                                .push(TaggedTask::Compound(method_task)),
                        }
                    }
                }
                None => {
                    let new_id: NewID = next_new_id;
                    next_new_id += 1;
                    tasks.insert(new_id);
                    alpha.insert(new_id, *parent_node.tn.mappings.get(&old_id).unwrap());
                    equivalent_ids.insert(old_id, new_id);
                    for greater in parent_node.tn.get_outgoing_edges(old_id) {
                        match *parent_node.tn.get_task(greater).borrow() {
                            Task::Primitive(_) => {
                                orderings.push((new_id, *equivalent_ids.get(&greater).unwrap()));
                            }
                            Task::Compound(_) => {
                                rec_hlpr(&mut orderings, &compound_mapping, new_id, greater);
                            }
                        }
                    }
                }
            }
        } // parent_node lifetime
        child = parent_unwrap;
        parent = child.borrow().parent.clone();
    }

    return HTN::new(tasks, orderings, domain, alpha);
}

fn rec_hlpr(
    orderings: &mut Vec<(NewID, NewID)>,
    compound_mapping: &HashMap<OldID, Vec<TaggedTask>>,
    predecessor_id: NewID,
    compound_task: OldID,
) {
    for task in compound_mapping.get(&compound_task).unwrap() {
        match task {
            TaggedTask::Primitive(id) => {
                orderings.push((predecessor_id, *id));
            }
            TaggedTask::Compound(id) => rec_hlpr(orderings, compound_mapping, predecessor_id, *id),
        }
    }
}

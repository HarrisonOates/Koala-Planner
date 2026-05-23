use astar::AStarResult;
use search_node::{Edge, SearchNode};
use weak_linearization::WeakLinearization;

use super::astar::{CustomStatistics, CustomStatistic};
use super::search_space::SearchSpace;
use super::*;
use crate::{
    domain_description::{ClassicalDomain, DomainTasks, FONDProblem, Facts},
    search::{AOStarSearch, HeuristicType, NodeStatus, SearchResult, StrongPolicy},
    task_network::{Method, Task, HTN},
};
use std::{
    cell::RefCell,
    collections::{BTreeSet, HashMap, HashSet},
    rc::Rc,
    string,
};

pub fn is_goal_weak_ld(
    problem: &FONDProblem,
    leaf_node: Rc<RefCell<SearchNode>>,
    custom_statistics: &mut CustomStatistics,
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
        .or_insert(CustomStatistic::Value(0)).accumulate(1);

    // construct new FONDProblem for the AO* subproblem
    let mut sub_problem = FONDProblem {
        facts: problem.facts.clone(),
        tasks: problem.tasks.clone(),
        initial_state: problem.initial_state.clone(),
        init_tn: deorder(leaf_node.clone()),
        rho: problem.rho,
    };

    // make initial task network just one abstract task
    sub_problem.collapse_tn();

    // call AO* algorithm
    let (solution, stats) = AOStarSearch::run(&sub_problem, HeuristicType::HAdd);

    // accumulate total subroutine search node count
    custom_statistics
        .entry(String::from("# of search nodes in all AO* calls"))
        .or_insert(CustomStatistic::Value(0)).accumulate(stats.search_nodes);
    leaf_node.borrow_mut().goal_tested = true;

    match solution {
        SearchResult::Success(policy) => {
            // search node count for final successful subroutine call
            // custom_statistics.insert(
            //     String::from("# of search nodes in final (successful) AO* call"),
            //     CustomStatistic::Value(stats.search_nodes),
            // );
            custom_statistics.insert(String::from("makespan"), CustomStatistic::Value(policy.makespan as u32));
            AStarResult::Strong(policy)
        }
        SearchResult::NoSolution => {
            AStarResult::NoSolution
        }
    }
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
                Some(name) => {
                    // println!("[CREATING COMPOUND ORDERING SET AT] {}", old_id);
                    compound_mapping.insert(old_id, Vec::new());
                    // iterate over them, check their type; if primitive, map to new ID and insert; if compound, insert with Old ID
                    let mut child_set: HashSet<OldID> = child.borrow().tn.get_task_id_set();
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
                    // println!("[NEW TASK] {}", new_id);
                    alpha.insert(new_id, *parent_node.tn.mappings.get(&old_id).unwrap());
                    equivalent_ids.insert(old_id, new_id);
                    // println!("[MAPPING] {} -> {}", old_id, new_id);
                    for greater in parent_node.tn.get_outgoing_edges(old_id) {
                        match *parent_node.tn.get_task(greater).borrow() {
                            Task::Primitive(_) => {
                                // error at this unwrap
                                // println!("[INSERT ORDERING] {} < {}", new_id, *equivalent_ids.get(&greater).unwrap());
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
                // println!("[INSERT ORDERING] {} < {}", predecessor_id, *id);
                orderings.push((predecessor_id, *id));
            }
            TaggedTask::Compound(id) => rec_hlpr(orderings, compound_mapping, predecessor_id, *id),
        }
    }
}

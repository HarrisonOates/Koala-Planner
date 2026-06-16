use std::collections::HashSet;

use super::TDG;
use crate::domain_description::{ClassicalDomain, Facts};
use crate::task_network::{PrimitiveAction, Task};
use crate::domain_description::FONDProblem;
use regex::Regex;

#[derive(Debug)]
pub struct RelaxedComposition {
    pub domain: ClassicalDomain,
    task_reachable_facts: Vec<Vec<u32>>,
    task_goal_facts: Vec<u32>,
}

impl RelaxedComposition {
    pub fn new(domain: &FONDProblem) -> RelaxedComposition {
        let mut new_facts = domain.facts.clone();
        // top down encoding
        let tasks = domain.tasks.get_all_tasks();
        let top_down_facts = tasks.iter().map(|x| x.borrow().get_name()).collect();
        new_facts = new_facts.extend(top_down_facts);
        // bottom-up encoding
        let bottom_up_facts: Vec<String> = domain
            .tasks
            .get_all_tasks()
            .iter()
            .filter(|x| x.borrow().is_primitive())
            .map(|x| x.borrow().get_name() + "_reachable")
            .collect();
        new_facts = new_facts.extend(bottom_up_facts);

        let new_actions = RelaxedComposition::encode(&domain, &new_facts);
        let classic_domain = ClassicalDomain::new(new_facts, new_actions);
        let tdg = TDG::new(&domain.init_tn);

        let n_tasks = domain.tasks.count_tasks() as usize;

        // Precompute: for each domain task_id, the _reachable fact IDs of all
        // primitives reachable from it via the TDG.
        let mut task_reachable_facts: Vec<Vec<u32>> = vec![vec![]; n_tasks];
        for task_id in 0..n_tasks as u32 {
            let reachables = tdg.task_reachability(task_id);
            let mut fact_ids: Vec<u32> = Vec::new();
            for reach_id in reachables {
                let task = domain.tasks.get_task(reach_id);
                if let Task::Primitive(prim) = &*task.borrow() {
                    if !prim.is_deterministic() {
                        let base = prim.name.clone() + "__determinized";
                        let n_effects = prim.add_effects.len() as u32;
                        for i in 0..n_effects {
                            let name = base.clone() + "_" + &i.to_string() + "_reachable";
                            fact_ids.push(classic_domain.facts.get_id(&name));
                        }
                    } else {
                        fact_ids.push(classic_domain.facts.get_id(&(prim.name.clone() + "_reachable")));
                    }
                }
            }
            task_reachable_facts[task_id as usize] = fact_ids;
        }

        // Precompute: for each domain task_id, the fact ID of its task-name in the
        // classical encoding (used by compute_goal_state).
        let mut task_goal_facts: Vec<u32> = vec![0; n_tasks];
        for task_id in 0..n_tasks as u32 {
            let name = domain.tasks.get_task(task_id).borrow().get_name();
            task_goal_facts[task_id as usize] = classic_domain.facts.get_id(&name);
        }

        RelaxedComposition {
            domain: classic_domain,
            task_reachable_facts,
            task_goal_facts,
        }
    }

    fn encode(domain: &FONDProblem, facts: &Facts) -> Vec<PrimitiveAction> {
        let mut result = vec![];
        let tasks = domain.tasks.get_all_tasks();
        for task in tasks.iter() {
            match &*task.borrow() {
                Task::Compound(c) => {
                    for method in c.methods.iter() {
                        let subtasks = method.decomposition.get_all_tasks();
                        let mut ids = HashSet::new();
                        for subtask in subtasks.iter() {
                            let task_name = subtask.borrow().get_name();
                            ids.insert(facts.get_id(&task_name));
                        }
                        let task_id = facts.get_id(&task.borrow().get_name());
                        let new_action = PrimitiveAction::new(
                            method.name.clone(),
                            0,
                            ids,
                            vec![HashSet::from([task_id])],
                            vec![HashSet::new()],
                        );
                        result.push(new_action);
                    }
                }
                Task::Primitive(p) => {
                    if p.add_effects.len() > 1 {
                        panic!("Relaxation assumes an all outcome determinized FOND problem");
                    }
                    // action executed effect
                    let mut add_effects = HashSet::from([facts.get_id(&p.name)]);
                    // canonical effects
                    if p.add_effects.len() == 1 {
                        add_effects.extend(p.add_effects[0].clone());
                    }
                    if p.name.contains("__determinized_") {
                        let re = Regex::new(r"__determinized_[0-9]+").unwrap();
                        let cleansed_name = re.replace(&p.name, "__determinized").to_string();
                        let fact_id = facts.get_id(&cleansed_name);
                        add_effects.insert(fact_id);
                    }
                    let top_down_precond = facts.get_id(&(p.name.clone() + "_reachable"));
                    let mut preconds = HashSet::from([top_down_precond]);
                    preconds.extend(p.pre_cond.clone());
                    let new_action = PrimitiveAction::new_with_probabilities(
                        p.name.clone(),
                        p.cost,
                        preconds,
                        vec![add_effects],
                        p.del_effects.clone(),
                    );
                    result.push(new_action);
                }
            }
        }
        result
    }

    pub fn compute_relaxed_state(&self, task_ids: &Vec<u32>, state: &HashSet<u32>) -> HashSet<u32> {
        let mut satisfied_preconds = state.clone();
        for &task_id in task_ids {
            satisfied_preconds.extend(&self.task_reachable_facts[task_id as usize]);
        }
        satisfied_preconds
    }

    pub fn compute_goal_state(&self, task_ids: &Vec<u32>) -> HashSet<u32> {
        task_ids.iter().map(|&id| self.task_goal_facts[id as usize]).collect()
    }

}

#[cfg(test)]
mod tests {
    use crate::task_network::{CompoundTask, HTN, Method};
    use super::*;
    use crate::domain_description::DomainTasks;
    use std::cell::RefCell;
    use std::collections::{BTreeSet, HashMap};
    use std::rc::Rc;
    fn generate_problem() -> FONDProblem {
        let p1 = Task::Primitive(PrimitiveAction::new(
            "p1".to_string(),
            1,
            HashSet::new(),
            vec![HashSet::from([1])],
            vec![HashSet::new()],
        ));
        let p2 = Task::Primitive(PrimitiveAction::new(
            "p2".to_string(),
            1,
            HashSet::from([2]),
            vec![HashSet::from([3])],
            vec![HashSet::new()],
        ));
        let p3 = Task::Primitive(PrimitiveAction::new(
            "p3".to_string(),
            1,
            HashSet::from([3]),
            vec![HashSet::from([2])],
            vec![HashSet::new()],
        ));
        let p4 = Task::Primitive(PrimitiveAction::new(
            "p4".to_string(),
            1,
            HashSet::from([1]),
            vec![HashSet::from([2])],
            vec![HashSet::from([1])],
        ));
        let t4 = Task::Compound(CompoundTask {
            name: "t4".to_string(),
            methods: vec![],
        });
        let t3 = Task::Compound(CompoundTask {
            name: "t3".to_string(),
            methods: vec![],
        });
        let t2 = Task::Compound(CompoundTask {
            name: "t2".to_string(),
            methods: vec![],
        });

        let t1 = Task::Compound(CompoundTask {
            name: "t1".to_string(),
            methods: vec![],
        });
        let domain = Rc::new(DomainTasks::new(vec![p1, p2, p3, p4, t1, t2, t3, t4]));
        let t4_m = Method::new(
            "t4_m".to_string(),
            HTN::new(
                BTreeSet::from([2, 3]),
                vec![],
                domain.clone(),
                HashMap::from([(2, domain.get_id("p2")), (3, domain.get_id("p3"))]),
            ),
        );
        let t3_m = Method::new(
            "t3_m".to_string(),
            HTN::new(
                BTreeSet::from([1, 2]),
                vec![(1, 2)],
                domain.clone(),
                HashMap::from([(1, domain.get_id("p2")), (2, domain.get_id("p2"))]),
            ),
        );
        let t2_m = Method::new(
            "t2_m".to_string(),
            HTN::new(
                BTreeSet::from([4, 3]),
                vec![(4, 3)],
                domain.clone(),
                HashMap::from([(4, domain.get_id("p4")), (3, domain.get_id("p3"))]),
            ),
        );
        let t1_m = Method::new(
            "t1_m".to_string(),
            HTN::new(
                BTreeSet::from([1, 4]),
                vec![],
                domain.clone(),
                HashMap::from([(1, domain.get_id("p1")), (4, domain.get_id("t4"))]),
            ),
        );
        let domain = domain.add_methods(vec![(4, t1_m), (5, t2_m), (6, t3_m), (7, t4_m)]);
        let init_tn = HTN::new(
            BTreeSet::from([1, 2, 3]),
            vec![(1, 3), (2, 3)],
            domain.clone(),
            HashMap::from([
                (1, domain.get_id("t1")),
                (2, domain.get_id("t2")),
                (3, domain.get_id("t3")),
            ]),
        );
        let mut p = FONDProblem {
            facts: Facts::new(vec![
                "1".to_string(),
                "2".to_string(),
                "3".to_string(),
                "4".to_string(),
            ]),
            tasks: domain,
            initial_state: HashSet::new(),
            init_tn: init_tn.clone(),
        };
        p.collapse_tn();
        p
    }

    #[test]
    pub fn encoding_test() {
        let problem = generate_problem();
        let to_classical = RelaxedComposition::new(&problem);
        let encoded = to_classical.domain;
        assert_eq!(encoded.facts.count(), 17);
        assert_eq!(encoded.actions.len(), 9);
        for action in encoded.actions.iter() {
            let mut name = action.name.clone();
            let flag = name.ends_with("_m");
            if flag {
                name = name.replace("_m", "");
            }
            let effect_id = encoded.facts.get_id(&name);
            assert_eq!(action.add_effects[0].contains(&effect_id), true);
            if !flag {
                let precond_id = encoded.facts.get_id(&(name + "_reachable"));
                assert_eq!(action.pre_cond.contains(&precond_id), true);
            }
        }
    }

    #[test]
    pub fn state_computation_test() {
        let problem = generate_problem();
        let to_classical = RelaxedComposition::new(&problem);
        let _t1 = &problem
            .tasks
            .get_all_tasks()
            .iter()
            .filter(|x| x.borrow().get_name() == "t1")
            .cloned()
            .collect::<Vec<RefCell<Task>>>()[0];
        let state = HashSet::from([to_classical.domain.facts.get_id("1")]);
        let _tn = HTN::new(
            BTreeSet::from([1]),
            vec![],
            problem.tasks.clone(),
            HashMap::from([(1, problem.tasks.get_id("t1"))]),
        );
        let relaxed_state =
            to_classical.compute_relaxed_state(&vec![problem.tasks.get_id("t1")], &state);
        assert_eq!(relaxed_state.len(), 4);
        let names = vec!["p1_reachable", "p2_reachable", "p3_reachable", "1"];
        for fact in relaxed_state {
            let name = to_classical.domain.facts.get_fact(fact);
            let mut is_contained = false;
            for item in names.iter() {
                if name == item {
                    is_contained = true;
                }
            }
            assert_eq!(is_contained, true);
        }
    }

    #[test]
    pub fn goal_state_test() {
        let problem = generate_problem();
        let to_classical = RelaxedComposition::new(&problem);
        let all_tasks = problem.tasks.get_all_tasks();
        let _t1 = &all_tasks
            .iter()
            .filter(|x| x.borrow().get_name() == "t1")
            .cloned()
            .collect::<Vec<RefCell<Task>>>()[0];
        let _p2 = &all_tasks
            .iter()
            .filter(|x| x.borrow().get_name() == "p2")
            .cloned()
            .collect::<Vec<RefCell<Task>>>()[0];
        let _state = HashSet::from([to_classical.domain.facts.get_id("1")]);
        let _tn = HTN::new(
            BTreeSet::from([1, 2]),
            vec![],
            problem.tasks.clone(),
            HashMap::from([
                (1, problem.tasks.get_id("t1")),
                (2, problem.tasks.get_id("p2")),
            ]),
        );
        let goal = to_classical.compute_goal_state(&vec![
            problem.tasks.get_id("t1"),
            problem.tasks.get_id("p2"),
        ]);
        assert_eq!(goal.len(), 2);
        let id_t1 = to_classical.domain.facts.get_id("t1");
        let id_p2 = to_classical.domain.facts.get_id("p2");
        assert_eq!(goal.contains(&id_t1), true);
        assert_eq!(goal.contains(&id_p2), true);
    }
}

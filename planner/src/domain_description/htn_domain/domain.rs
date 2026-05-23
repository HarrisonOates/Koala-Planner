use std::collections::BTreeSet;
use std::collections::HashMap;
use std::collections::HashSet;
use std::rc::Weak;

use crate::task_network::CompoundTask;
use crate::task_network::Method;

use super::DomainTasks;
use super::{HTN, PrimitiveAction, Facts, Task};
use rand::distributions::Alphanumeric;
use rand::distributions::DistString;
use std::rc::Rc;

#[derive(Debug)]
pub struct FONDProblem{
    pub facts: Facts,
    pub tasks: Rc<DomainTasks>,
    pub initial_state: HashSet<u32>,
    pub init_tn: HTN,
    pub rho: f64,
}

impl FONDProblem {
    pub fn new(literals: Vec<String>,
                // Vector of tuples in the form (action name, preconds, Vec<(Vec<add>, Vec<del>)>, probabilities)
                actions: Vec<(String, Vec<String>, Vec<(Vec<String>, Vec<String>)>, Vec<f64>)>,
                // Vector of tuples in the form (method name, task name, vec<subtasks>, vec<orderings>)
                methods: Vec<(String, String, Vec<String>, Vec<(u32, u32)>)>,
                abstract_tasks: Vec<String>,
                init: HashSet<String>,
                first_task: String
    ) -> FONDProblem {
        let facts =  Facts::new(literals);
        let initial_state = init.iter().map(|x| facts.get_id(x)).collect();
        let mut processed_tasks  = Vec::new();
        // Process Tasks
        for (name, precond, effects, probabilities) in actions.into_iter() {
            let mut add_effs = vec![];
            let mut del_effs = vec![];
            for (add_effect, del_effect) in effects.into_iter() {
                let add_set_i: HashSet<u32> = HashSet::from_iter(add_effect.into_iter()
                    .map(|x| facts.get_id(&x)));
                let del_set_i: HashSet<u32> = HashSet::from_iter(del_effect.into_iter()
                    .map(|x| facts.get_id(&x)));
                add_effs.push(add_set_i);
                del_effs.push(del_set_i);
            }
            let action = PrimitiveAction::new_with_probabilities(
                name,
                1,
                precond.into_iter().map(|x| facts.get_id(&x)).collect(),
                add_effs,
                del_effs,
                probabilities
            );
            processed_tasks.push(Task::Primitive(action));
        }

        // Process Abstract tasks
        for task in abstract_tasks.into_iter() {
            let new_task = Task::Compound(
                CompoundTask { name: task, methods: vec![] }
            );
            processed_tasks.push(new_task);
        }
        let mut domain_tasks = Rc::new(DomainTasks::new(processed_tasks));

        // Process methods
        let mut parsed_methods = vec![];
        for (name, task, subtasks, orderings) in methods.into_iter() {
            // Use sequential node IDs (0, 1, 2, ...) so duplicate task names
            // (e.g. two "attempt[]" subtasks) each get a distinct node.
            let n = subtasks.len() as u32;
            let node_ids: BTreeSet<u32> = (0..n).collect();
            let mappings: HashMap<u32, u32> = (0..n)
                .map(|i| (i, domain_tasks.get_id(&subtasks[i as usize])))
                .collect();
            // Orderings from the JSON are already subtask indices, which equal node IDs.
            let processed_orderings: Vec<(u32, u32)> = orderings.into_iter().collect();
            let decomposition = HTN::new(node_ids, processed_orderings, domain_tasks.clone(), mappings);
            let method = Method::new(name, decomposition);
            let task_id = domain_tasks.get_id(&task);
            parsed_methods.push((task_id, method));
        }
        let domain_tasks = domain_tasks.add_methods(parsed_methods);
        // initial abstract task
        let initial_task_id = domain_tasks.get_id(&first_task);
        let tn = HTN::new(
            BTreeSet::from([initial_task_id]),
            vec![],
            domain_tasks.clone(),
            HashMap::from([
                (initial_task_id, initial_task_id)
            ])
        );
        FONDProblem {
            facts,
            tasks: domain_tasks,
            initial_state,
            init_tn: tn,
            rho: 1.0,
        }
    }

    // Converts init tn into a single compound task
    pub fn collapse_tn(&mut self) {
        let rand_s: String = Alphanumeric.sample_string(&mut rand::thread_rng(), 4);
        let task_name = "collapsed_top__".to_string() + &rand_s;
        let method_name = task_name.clone() + "_m";
        let new_task = Task::Compound(CompoundTask {
            name: task_name.clone(),
            methods: vec![Method::new(
                method_name,
                self.init_tn.clone()
            )]
        });
        let mut new_domain = self.tasks.as_ref().clone();
        new_domain.add_task(new_task);
        let new_domain = Rc::new(new_domain);
        self.tasks = new_domain.clone();
        let new_tn = HTN::new(
            BTreeSet::from([1]), vec![],
            new_domain.clone(),
            HashMap::from([(1, self.tasks.get_id(&task_name))])
        );
        self.init_tn = new_tn;
    }
}
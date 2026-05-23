use crate::domain_description::FONDProblem;

use super::{HeuristicType, SearchGraph, SearchResult, SearchStats};
use std::time::Instant;

pub struct AOStarSearch {}
impl AOStarSearch {
    // the initial TN is assumed to be in collapsed format (i.e., with a single abstract task)
    pub fn run(problem: &FONDProblem, h_type: HeuristicType) -> (SearchResult, SearchStats) {
        let mut explored_nodes: u32 = 0;
        let mut max_depth = 0;
        let start_time = Instant::now();
        let mut search_graph = SearchGraph::new(problem);
        while !search_graph.is_terminated() {
            let n = search_graph.find_a_tip_node();
            search_graph.expand(n, &h_type, false);
            search_graph.backward_cost_revision(n);
            explored_nodes += 1;
            let depth = search_graph.ids.get(&n).unwrap().borrow().depth;
            if depth > max_depth {
                max_depth = depth;
            }
            if explored_nodes % 10_000 == 0 {
                eprintln!(
                    "[AO*] explored={} | nodes={} | depth={} | {:.1}s",
                    explored_nodes,
                    search_graph.ids.len(),
                    max_depth,
                    start_time.elapsed().as_secs_f64()
                );
            }
        }
        let root_prob = search_graph
            .ids
            .get(&search_graph.root)
            .unwrap()
            .borrow()
            .success_probability;
        let result = search_graph.search_result(&problem.facts);
        let stats = SearchStats {
            max_depth: max_depth,
            search_nodes: search_graph.ids.len() as u32,
            explored_nodes: explored_nodes,
            seach_time: start_time.elapsed(),
            success_probability: Some(root_prob),
            rho_threshold: problem.rho,
        };
        (result, stats)
    }
}

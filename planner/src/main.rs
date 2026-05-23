#![allow(unused)]
use std::{collections::{HashSet, HashMap}, env};

extern crate bit_vec;

mod domain_description;
mod graph_lib;
mod heuristics;
mod relaxation;
mod search;
mod task_network;

use crate::search::fixed_method::heuristic_factory;
use crate::search::{HeuristicType, SearchResult};
use crate::search::htn_andstar::TiebreakerKind;
use domain_description::{read_json_domain, FONDProblem};
use heuristics::{h_add, h_ff, h_max};
use relaxation::RelaxedComposition;
use search::{
    astar::AStarResult,
    goal_checks::{is_goal_strong_od, is_goal_weak_ld},
    search_node::{get_successors_systematic, SearchNode},
};
use task_network::HTN;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("The path to the problem file is not given.");
        return;
    }
    let mut problem = read_json_domain(&args[1]);

    // Parse --threshold argument
    let rho: f64 = args.iter()
        .position(|x| x == "--threshold")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.0);
    problem.rho = rho;
    if rho < 1.0 {
        println!("Using probability threshold (rho): {:.4}", rho);
    }

    // Parse --tiebreak argument (AND* only)
    let tiebreaker = args.iter()
        .position(|x| x == "--tiebreak")
        .and_then(|i| args.get(i + 1))
        .map(|v| match v.as_str() {
            "policy-size" => TiebreakerKind::PolicySize,
            "closure"     => TiebreakerKind::ClosureFirst,
            "combined"    => TiebreakerKind::Combined,
            other => panic!("Unknown tiebreak '{}': use policy-size | closure | combined", other),
        })
        .unwrap_or(TiebreakerKind::NoTiebreak);

    // TODO: Refactor flexible method and fixed method to accept
    // heuristic input of the same type, so we only need one of the
    // following two match expressions

    let heuristic_flexible = match args.get(3) {
        Some(flag) => match flag.as_str() {
            "--add" => {
                println!("Using Add heuristic");
                HeuristicType::HAdd
            },
            "--max" => {
                println!("Using Max heuristic");
                HeuristicType::HMax
            },
            "--ff" => {
                println!("Using FF heuristic");
                HeuristicType::HFF
            },
            "--prob" => {
                println!("Using Prob heuristic");
                HeuristicType::HProb
            },
            _ => panic!("Unknown heuristic")
        },
        None => {
            panic!("Expected heuristic flag")
        }
    };

    match args.get(2) {
        Some(flag) => match flag.as_str() {
            "--fixed" => {
                let heuristic_fixed = match args.get(3) {
                    Some(flag) => match flag.as_str() {
                        "--add" => heuristic_factory::create_function_with_heuristic(h_add),
                        "--max" => heuristic_factory::create_function_with_heuristic(h_max),
                        "--ff"  => heuristic_factory::create_function_with_heuristic(h_ff),
                        _ => panic!("Did not recognise flag {}", flag),
                    },
                    None => panic!("Expected heuristic flag"),
                };
                println!("Running fixed method solver");
                fixed_method(&problem, heuristic_fixed)
            },
            "--flexible" => {
                println!("Running AO* flexible solver");
                ao_star(&problem, heuristic_flexible)
            },
            "--andstar" => {
                println!("Running HTN-AND* solver");
                println!("Tiebreaker: {:?}", tiebreaker);
                htn_andstar(&problem, heuristic_flexible, tiebreaker)
            },
            _ => panic!("Did not recognise flag {}", flag)
        },
        None => ao_star(&problem, heuristic_flexible),
    }
}

fn ao_star(problem: &FONDProblem, h_type: HeuristicType) {
    let (solution, stats) = search::AOStarSearch::run(problem, h_type);
    print!("{}", stats);
    match solution {
        SearchResult::Success(policy) => {
            println!("makespan: {}", policy.makespan);
            println!("policy entries: {}", policy.transitions.len());
            print!("{}", policy);
        }
        SearchResult::NoSolution => {
            println!("Problem has no solution");
        }
    }
}

fn htn_andstar(problem: &FONDProblem, h_type: HeuristicType, tiebreaker: TiebreakerKind) {
    let (solution, stats) = search::htn_andstar::run(problem, h_type, tiebreaker);
    print!("{}", stats);
    match solution {
        SearchResult::Success(x) => {
            println!("makespan: {}", x.makespan);
            println!("policy entries: {}", x.transitions.len());
            print!("{}", x);
        }
        SearchResult::NoSolution => {
            println!("Problem has no solution");
        }
    }
}

fn fixed_method(problem: &FONDProblem, heuristic: heuristic_factory::HeuristicFn) {
    let (solution, stats) = search::fixed_method::astar::a_star_search(
        &problem,
        heuristic,
        get_successors_systematic,
        || 1.0,
        is_goal_strong_od,
    );
    println!("{}", stats);
    if let AStarResult::Strong(policy) = solution {
        println!("Solution was found");
        println!("# of policy entries: {}", policy.transitions.len());
        println!("Success probability: {:.4}", policy.success_probability);
    } else {
        println!("Problem has no solution");
    }
}

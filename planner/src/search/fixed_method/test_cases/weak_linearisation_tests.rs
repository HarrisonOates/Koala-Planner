use super::super::astar::{a_star_search, AStarResult};
use super::super::goal_checks::*;
use super::super::*;
use crate::domain_description::FONDProblem;
use crate::domain_description::Facts;
use search_node::get_successors_systematic;
use std::{
    borrow::BorrowMut,
    collections::{BTreeSet, HashMap, HashSet},
    vec,
};

#[cfg(test)]
#[test]
pub fn weak_ld_problem_1() {
    let problem = FONDProblem::new(
        vec![],
        vec![
            (String::from("prim_a"), vec![], vec![], vec![1.0]),
            (String::from("prim_b"), vec![], vec![], vec![1.0]),
            (String::from("prim_e"), vec![], vec![], vec![1.0]),
            (String::from("prim_x"), vec![], vec![], vec![1.0]),
        ],
        vec![
            (
                String::from("m0"),
                String::from("comp_init"),
                vec![
                    String::from("prim_a"),
                    String::from("comp_c"),
                    String::from("prim_x"),
                ],
                vec![(0, 1), (1, 2)],
            ),
            (
                String::from("m1"),
                String::from("comp_c"),
                vec![String::from("prim_b"), String::from("comp_d")],
                vec![(0, 1)],
            ),
            (
                String::from("m2"),
                String::from("comp_d"),
                vec![String::from("prim_e")],
                vec![],
            ),
        ],
        vec![
            String::from("comp_init"),
            String::from("comp_c"),
            String::from("comp_d"),
        ],
        HashSet::new(),
        String::from("comp_init"),
    );
    let (solution, statistics) = a_star_search(
        &problem,
        |x, y, z, w| 0.0,
        get_successors_systematic,
        || 1.0,
        is_goal_weak_ld,
    );
    println!("\nPLAN\n");
    if let AStarResult::Linear(lin) = solution {
        println!("{}", lin.to_string(&problem));
    } else {
        println!("NO SOLUTION");
    }
    println!(
        "\nSEARCH SPACE explored:{} total:{}\n",
        statistics.space.explored_nodes, statistics.space.total_nodes
    );
    println!("{}", statistics.space.to_string(&problem));
}

#[cfg(test)]
#[test]
pub fn weak_ld_problem_2() {
    let f1: String = String::from("f1");
    let f2: String = String::from("f2");
    let f3: String = String::from("f3");
    let problem = FONDProblem::new(
        vec![f1.clone(), f2.clone(), f3.clone()],
        vec![
            (
                String::from("a"),
                vec![],
                vec![(vec![], vec![f2.clone()]), (vec![], vec![])],
                vec![0.5, 0.5],
            ),
            (
                String::from("b"),
                vec![],
                vec![(vec![f3.clone()], vec![f2.clone()])],
                vec![1.0],
            ),
        ],
        vec![
            (
                String::from("m0"),
                String::from("init"),
                vec![String::from("a"), String::from("b"), String::from("c")],
                vec![(0, 2), (1, 2)],
            ),
            (
                String::from("m1"),
                String::from("c"),
                vec![String::from("a"), String::from("c")],
                vec![(0, 1)],
            ),
            (
                String::from("m2"),
                String::from("c"),
                vec![String::from("a")],
                vec![],
            ),
        ],
        vec![String::from("c"), String::from("init")],
        HashSet::from([f1.clone(), f2.clone()]),
        String::from("init"),
    );
    let (solution, statistics) = a_star_search(
        &problem,
        |x, y, z, w| 0.0,
        get_successors_systematic,
        || 1.0,
        is_goal_weak_ld,
    );
    println!("\nPLAN\n");
    if let AStarResult::Linear(lin) = solution {
        println!("{}", lin.to_string(&problem));
    } else {
        println!("NO SOLUTION");
    }
    println!(
        "\nSEARCH SPACE explored:{} total:{}\n",
        statistics.space.explored_nodes, statistics.space.total_nodes
    );
    println!("{}", statistics.space.to_string(&problem));
}

#[cfg(test)]
#[test]
pub fn weak_ld_problem_3() {
    // facts
    let f1 = String::from("f1");
    // actions
    let a = String::from("a");
    let b = String::from("b");
    // compounds
    let init = String::from("init");
    // methods
    let m1 = String::from("m1");
    let m2 = String::from("m2");
    let m3 = String::from("m3");
    let m4 = String::from("m4");
    let m5 = String::from("m5");
    let m6 = String::from("m6");
    let m7 = String::from("m7");
    // fond problem
    let problem = FONDProblem::new(
        vec![f1.clone()],
        vec![
            (a.clone(), vec![], vec![(vec![], vec![])], vec![1.0]),
            (b.clone(), vec![f1.clone()], vec![(vec![], vec![])], vec![1.0]),
        ],
        vec![
            (m1.clone(), init.clone(), vec![b.clone()], vec![]),
            (m2.clone(), init.clone(), vec![b.clone()], vec![]),
            (m3.clone(), init.clone(), vec![b.clone()], vec![]),
            (m4.clone(), init.clone(), vec![a.clone()], vec![]),
            (m5.clone(), init.clone(), vec![b.clone()], vec![]),
            (m6.clone(), init.clone(), vec![b.clone()], vec![]),
            (m7.clone(), init.clone(), vec![b.clone()], vec![]),
        ],
        vec![init.clone()],
        HashSet::new(),
        init.clone(),
    );
    let (solution, statistics) = a_star_search(
        &problem,
        |x, y, z, w| 0.0,
        get_successors_systematic,
        || 1.0,
        is_goal_weak_ld,
    );
    println!("\nPLAN\n");
    if let AStarResult::Linear(lin) = solution {
        println!("{}", lin.to_string(&problem));
    } else {
        println!("NO SOLUTION");
    }
    println!(
        "\nSEARCH SPACE explored:{} total:{}\n",
        statistics.space.explored_nodes, statistics.space.total_nodes
    );
    println!("{}", statistics.space.to_string(&problem));
}

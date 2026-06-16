use search_node::SearchNode;
use std::{cell::RefCell, collections::BTreeMap, rc::Rc};


use super::*;

/**
 * Floats don't have Ord because of special values like NaN. But since a
 * heuristic value should never be NaN, we may panic in those scenarios.
 * TODO: Consider using the ordered-float crate instead.
 */
#[derive(Clone, Copy)]
struct OrderedFloat(f32);
impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        assert!(
            !self.0.is_nan() && !other.0.is_nan(),
            "NaN values are not allowed"
        );
        self.0.partial_cmp(&other.0).unwrap()
    }
}
impl PartialOrd for OrderedFloat {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        assert!(
            !self.0.is_nan() && !other.0.is_nan(),
            "NaN values are not allowed"
        );
        self.0.partial_cmp(&other.0)
    }
}
impl PartialEq for OrderedFloat {
    fn eq(&self, other: &Self) -> bool {
        assert!(
            !self.0.is_nan() && !other.0.is_nan(),
            "NaN values are not allowed"
        );
        self.0 == other.0
    }
}
impl Eq for OrderedFloat {}

pub struct PriorityQueue {
    // Given a particular f score (key) store all nodes with this f score (vector)
    map: BTreeMap<OrderedFloat, Vec<Rc<RefCell<SearchNode>>>>,
}

impl PriorityQueue {
    pub fn new() -> Self {
        PriorityQueue {
            map: BTreeMap::new(),
        }
    }

    pub fn insert(&mut self, search_node: Rc<RefCell<SearchNode>>) {
        let key = OrderedFloat(search_node.borrow().f_value());
        if let Some(bucket) = self.map.get_mut(&key) {
            bucket.push(search_node.clone());
        } else {
            self.map.insert(key, vec![search_node.clone()]);
        }
    }

    pub fn remove(&mut self, search_node: Rc<RefCell<SearchNode>>) {
        let mut bucket_empty = false;
        let key = OrderedFloat(search_node.borrow().f_value());
        if let Some(bucket) = self.map.get_mut(&key) {
            // Retain elements which are not pointer-equal to this search node
            bucket.retain(|x| !Rc::ptr_eq(x, &search_node));
            bucket_empty = bucket.is_empty();
        }
        // Can't have any empty buckets, since the pop_least function relies on the assumption that
        // all buckets contain at least 1 search node
        if bucket_empty {
            self.map.remove(&key);
        }
    }

    pub fn pop_least(&mut self) -> Option<Rc<RefCell<SearchNode>>> {
        if let Some((&key, bucket)) = self.map.iter_mut().next() {
            // We may assume every bucket is non-empty
            let node = bucket.pop().unwrap();
            // If the bucket becomes empty, remove the whole entry. This is necessary since this
            // function relies on the assumption that all buckets are non-empty
            if bucket.is_empty() {
                self.map.remove(&key);
            }
            Some(node)
        } else {
            None
        }
    }
}

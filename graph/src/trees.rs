use super::*;

use indicatif::ProgressIterator;
use rayon::iter::IntoParallelIterator;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelIterator;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::{collections::HashSet, sync::atomic::AtomicU32};
use vec_rand::xorshift::xorshift as rand_u64;

/// # Implementation of algorithms relative to trees.
///
/// # Definitions
/// - **Self-loops**: Edges with source equal to the destination.
/// - **Singleton**: A node with no incident edges, (self-loops are not considered).
/// - **Spanning Tree**: A set of edges that allows to build a path between every
///     node in the graph. For a graph with n nodes the spanning tree will have n - 1 edges.
/// - **Spanning Arborescence**: is the generalizzation of the spanning tree for graphs
///     with multiple components. Being a tree it trivially contains no self-loops.
///     For a grpah with n nodes and c components the spanning arborescence will have
///     n - c edges.
/// - **Component**: Set of nodes in which any two vertices in it are connected to
///     each other by paths. A singleton is a component and so is a singleton with a
///     self-loop.
impl Graph {
    /// Returns iterator over shuffled edge IDs and node IDs.
    ///
    /// # Arguments
    /// * `random_state`: u64 - The random state to reproduce the given edge sampling.
    fn iter_edges_from_random_state(
        &self,
        random_state: u64,
    ) -> impl Iterator<Item = (EdgeT, NodeT, NodeT)> + '_ {
        let edges_number = self.get_directed_edges_number();
        // We execute two times the xorshift to improve the randomness of the seed.
        let updated_random_state = rand_u64(rand_u64(splitmix64(random_state)));
        (updated_random_state..edges_number + updated_random_state).filter_map(move |i| {
            let edge_id = i % edges_number;
            let (src, dst) = unsafe { self.get_unchecked_node_ids_from_edge_id(edge_id) };
            match src == dst || !self.directed && src > dst {
                true => None,
                false => Some((edge_id, src, dst)),
            }
        })
    }

    /// Returns iterator over shuffled edge IDs and node IDs with preference to given edge types.
    ///
    /// # Arguments
    /// * `random_state`: Option<u64> - The random state to reproduce the given edge sampling.
    /// * `undesired_edge_types`: Option<HashSet<Option<EdgeTypeT>>> - The edge types whose edges are to leave as last.
    /// * `verbose`: Option<bool> - Whether to show a loading bar.
    fn iter_on_edges_with_preference_from_random_state(
        &self,
        random_state: Option<u64>,
        undesired_edge_types: Option<HashSet<Option<EdgeTypeT>>>,
        verbose: Option<bool>,
    ) -> impl Iterator<Item = (NodeT, NodeT)> + '_ {
        let random_state = random_state.unwrap_or(0xbadf00d);
        let verbose = verbose.unwrap_or(false);
        let pb = get_loading_bar(
            verbose,
            format!("Building random spanning tree for {}", self.name).as_ref(),
            self.get_directed_edges_number() as usize,
        );
        let result: Box<dyn Iterator<Item = (NodeT, NodeT)>> =
            if let (Some(uet), _) = (undesired_edge_types, &self.edge_types) {
                // We cannot retrun two different iters that reference data owned by
                // this function, so we clone it. This is fine since it should contains
                // only few values
                let uet_copy = uet.clone();
                Box::new(
                    self.iter_edges_from_random_state(random_state)
                        .filter_map(move |(edge_id, src, dst)| {
                            if uet.contains(&unsafe {
                                self.get_unchecked_edge_type_id_from_edge_id(edge_id)
                            }) {
                                return None;
                            }
                            Some((src, dst))
                        })
                        .chain(self.iter_edges_from_random_state(random_state).filter_map(
                            move |(edge_id, src, dst)| {
                                if !uet_copy.contains(&unsafe {
                                    self.get_unchecked_edge_type_id_from_edge_id(edge_id)
                                }) {
                                    return None;
                                }
                                Some((src, dst))
                            },
                        )),
                )
            } else {
                Box::new(
                    self.iter_edges_from_random_state(random_state)
                        .map(|(_, src, dst)| (src, dst)),
                )
            };

        result.progress_with(pb)
    }

    /// Returns set of edges composing a spanning tree and connected components.
    ///
    /// If the graph is composed of a single node with one or more self-loops,
    /// we consider such a graph as a graph with an empty spanning tree, with
    /// a single component of size one.
    ///
    /// # Arguments
    ///
    /// `edges`: impl Iterator<Item = (NodeT, NodeT)> + 'a - Iterator for the edges to explore. If sorted, computed a minimum spanning tree.
    ///
    /// # Returns
    /// Tuple with:
    ///     - Set of the edges
    ///     - Vector of the nodes components
    ///     - Total components number
    ///     - Minimum component size
    ///     - Maximum component size
    pub(crate) fn kruskal<'a>(
        &self,
        edges: impl Iterator<Item = (NodeT, NodeT)> + 'a,
    ) -> (HashSet<(NodeT, NodeT)>, Vec<NodeT>, NodeT, NodeT, NodeT) {
        // If the graph does not have nodes, we return all
        // results as empty to provide an uniform, though pathological,
        // return value.
        if !self.has_nodes() {
            return (HashSet::new(), Vec::new(), 0, 0, 0);
        }
        // Similarly, when dealing with a graph with no edges, we define
        // the spanning tree as empty and the components as the set of the
        // nodes themselves. Since all nodes are singletons, both the
        // maximum component size and minimum component size equals to one.
        if !self.has_edges() {
            return (
                HashSet::new(),
                self.get_node_ids(),
                self.get_nodes_number(),
                1,
                1,
            );
        }

        let nodes_number = self.get_nodes_number() as usize;
        let mut tree = HashSet::with_capacity(self.get_nodes_number() as usize);
        let mut components = vec![NODE_NOT_PRESENT; nodes_number];
        let mut component_sizes: Vec<NodeT> = Vec::new();
        let mut components_remapping: Vec<NodeT> = Vec::new();
        let mut max_component_size: NodeT = 0;
        let mut min_component_size = NodeT::MAX;

        // When there are singleton nodes with self-loops,
        // which is an arguability weird feature of some graphs,
        // Kruskal fails to identify them because by definition
        // a tree cannot contain self-loop.
        // We call these nodes with one or more self-loops
        // (in the case of a multigraph) `singletons with self-loops` for lack of
        // a better term. These nodes are treated as nodes in their own
        // component and their edges (the self-loops) are not added to the tree.
        if self.has_disconnected_nodes() {
            // When there are singleton nodes, the minimum component size
            // surely becomes one.
            min_component_size = 1;
            // Similarly we need to bump up the max component size, as if
            // this graph is composed of only singleton nodes with self-loops
            // we would not iterate thorugh them in the Kruskal loop
            // since it skips self-loops.
            max_component_size = 1;
            // We iterate through the singleton nodes and the singleton nodes
            // with self-loops.
            self.iter_singleton_node_ids()
                .chain(self.iter_singleton_nodes_with_selfloops_node_ids())
                .enumerate()
                .for_each(|(component_number, node_id)| {
                    components[node_id as usize] = component_number as NodeT;
                });
            // We can re-initialize the component sizes as the vector with
            // all ones bit as the singleton nodes number.
            component_sizes = vec![1; self.get_disconnected_nodes_number() as usize];
            // Similarly, the components remapping can be initialized to a range.
            components_remapping =
                (0..self.get_disconnected_nodes_number()).collect::<Vec<NodeT>>();
        }

        edges.for_each(|(src, dst)| {
            // If this is a self-loop we skip it.
            if src == dst {
                return;
            }
            let src_component = components[src as usize];
            let dst_component = components[dst as usize];
            match (
                src_component == NODE_NOT_PRESENT,
                dst_component == NODE_NOT_PRESENT,
            ) {
                // If neither nodes have a component, they must be inserted
                // both in the components vector and in the tree.
                // The edge must be added to the three.
                (true, true) => {
                    let new_component_id = components_remapping.len() as NodeT;
                    components[src as usize] = new_component_id;
                    components[dst as usize] = new_component_id;
                    components_remapping.push(new_component_id);
                    component_sizes.push(2);
                    max_component_size = max_component_size.max(2);
                }
                // If both nodes have a component, the two components must be merged
                // if they are not the same one.
                // The edge must be added to the three.
                // The components mapping must be updated and afterwards the other nodes
                // must be updated accordingly to this update.
                (false, false) => {
                    if src_component == dst_component {
                        return;
                    }
                    let src_component = components_remapping[src_component as usize];
                    let dst_component = components_remapping[dst_component as usize];
                    if src_component == dst_component {
                        return;
                    }
                    let min_component_id = src_component.min(dst_component);
                    let max_component_id = src_component.max(dst_component);

                    // We merge the two component sizes.
                    component_sizes[min_component_id as usize] +=
                        component_sizes[max_component_id as usize];

                    // We check if we have a new component size record
                    max_component_size =
                        max_component_size.max(component_sizes[min_component_id as usize]);

                    // If the component we have removed and integrated with
                    // the component with the smaller has a size greater than
                    // one, we may need to remap some element of the component
                    // to this new component.
                    // Since the components within this loops start from edges
                    // they cannot be smaller in cardinality than
                    // two nodes.
                    components_remapping
                        .iter_mut()
                        .zip(component_sizes.iter_mut())
                        .for_each(|(component_id, component_size)| {
                            // If one of other components is being remapped to
                            // the maximum component, we need to update its value
                            // to the new value this component is being remapped to.
                            if *component_id == max_component_id {
                                *component_id = min_component_id;
                                // We need to invalidate the size of the component
                                // we have remapped because otherwise we may count it
                                // when computing the minimum component size.
                                *component_size = 0;
                            }
                        });
                }
                // If only one node has a component, the second node must be added.
                _ => {
                    let (component_id, not_inserted_node) = match src_component == NODE_NOT_PRESENT
                    {
                        true => (components_remapping[dst_component as usize], src),
                        false => (components_remapping[src_component as usize], dst),
                    };
                    component_sizes[component_id as usize] += 1;
                    max_component_size =
                        max_component_size.max(component_sizes[component_id as usize]);
                    components[not_inserted_node as usize] = component_id as NodeT;
                }
            };
            tree.insert((src, dst));
        });

        // Remapping components to a dense remapping
        let mut components_number = 0;
        for i in 0..components_remapping.len() {
            if components_remapping[i] >= components_number {
                components_remapping[i] = components_number;
                components_number += 1;
            } else {
                components_remapping[i] = components_remapping[components_remapping[i] as usize];
            }
        }

        components.par_iter_mut().for_each(|remapped| {
            *remapped = components_remapping[*remapped as usize];
        });

        // If the minimum component size is still bigger than one
        // that is, we do not know alredy that there is a singleton
        // we need to compute it.
        if min_component_size > 1 {
            min_component_size = match components_number {
                1 => max_component_size,
                2 => self.get_nodes_number() - max_component_size,
                _ => component_sizes
                    .into_par_iter()
                    .filter(|val| *val > 0)
                    .min()
                    .unwrap(),
            };
        }

        (
            tree,
            components,
            components_number as NodeT,
            min_component_size,
            max_component_size,
        )
    }

    /// Returns set of edges composing a spanning tree and connected components.
    ///
    /// The spanning tree is NOT minimal.
    /// The given random_state is NOT the root of the tree.
    ///
    /// This method, additionally, allows for undesired edge types to be
    /// used to build the spanning tree only in extremis when it is utterly
    /// necessary in order to complete the spanning arborescence.
    ///
    /// The quintuple returned contains:
    /// - Set of the edges used in order to build the spanning arborescence.
    /// - Vector of the connected component of each node.
    /// - Number of connected components.
    /// - Minimum component size.
    /// - Maximum component size.
    ///
    /// # Arguments
    ///
    /// * `random_state`: Option<EdgeT> - The random_state to use for the holdout,
    /// * `undesired_edge_types`: &Option<HashSet<Option<EdgeTypeT>>> - Which edge types id to try to avoid.
    /// * `verbose`: Option<bool> - Whether to show a loading bar or not.
    ///
    /// # Example
    /// To compute a random spanning arborescence using Kruskal you can use the following:
    /// ```rust
    /// # let graph = graph::test_utilities::load_ppi(true, true, true, true, false, false);
    /// let (
    ///     spanning_arborescence_set,
    ///     connected_components_number,
    ///     number_of_connected_components,
    ///     minimum_component_size,
    ///     maximum_component_size
    /// ) = graph.random_spanning_arborescence_kruskal(
    ///     Some(42),
    ///     None,
    ///     Some(false)
    /// );
    /// assert_eq!(connected_components_number.len(), graph.get_nodes_number() as usize);
    /// assert!(minimum_component_size <= maximum_component_size);
    /// assert!(maximum_component_size <= graph.get_nodes_number());
    /// ```
    pub fn random_spanning_arborescence_kruskal(
        &self,
        random_state: Option<EdgeT>,
        undesired_edge_types: Option<HashSet<Option<EdgeTypeT>>>,
        verbose: Option<bool>,
    ) -> (HashSet<(NodeT, NodeT)>, Vec<NodeT>, NodeT, NodeT, NodeT) {
        self.kruskal(self.iter_on_edges_with_preference_from_random_state(
            random_state,
            undesired_edge_types,
            verbose,
        ))
    }

    /// Returns consistent spanning arborescence using Kruskal.
    ///
    /// The spanning tree is NOT minimal.
    ///
    /// The quintuple returned contains:
    /// - Set of the edges used in order to build the spanning arborescence.
    /// - Vector of the connected component of each node.
    /// - Number of connected components.
    /// - Minimum component size.
    /// - Maximum component size.
    ///
    /// # Arguments
    /// * `verbose`: Option<bool> - Whether to show a loading bar or not.
    ///
    /// # Example
    /// To compute a spanning arborescence using Kruskal you can use the following:
    /// ```rust
    /// # let graph = graph::test_utilities::load_ppi(true, true, true, true, false, false);
    /// let (
    ///     spanning_arborescence_set,
    ///     connected_components_number,
    ///     number_of_connected_components,
    ///     minimum_component_size,
    ///     maximum_component_size
    /// ) = graph.spanning_arborescence_kruskal(None);
    /// assert_eq!(connected_components_number.len(), graph.get_nodes_number() as usize);
    /// assert!(minimum_component_size <= maximum_component_size);
    /// assert!(maximum_component_size <= graph.get_nodes_number());
    /// ```
    pub fn spanning_arborescence_kruskal(
        &self,
        verbose: Option<bool>,
    ) -> (HashSet<(NodeT, NodeT)>, Vec<NodeT>, NodeT, NodeT, NodeT) {
        let verbose = verbose.unwrap_or(false);
        let pb = get_loading_bar(
            verbose,
            &format!(
                "Computing spanning arborescence with Kruskal for {}",
                self.get_name()
            ),
            self.get_unique_edges_number() as usize,
        );
        self.kruskal(
            self.iter_unique_edge_node_ids(self.directed)
                .progress_with(pb),
        )
    }

    #[manual_binding]
    /// Returns set of edges composing a spanning arborescence.
    ///
    /// This is the implementaiton of [A Fast, Parallel Spanning Tree Algorithm for Symmetric Multiprocessors (SMPs)](https://smartech.gatech.edu/bitstream/handle/1853/14355/GT-CSE-06-01.pdf)
    /// by David A. Bader and Guojing Cong.
    ///
    /// The returned tuple contains:
    /// - The number edges required in order to build the spanning arborescence.
    /// - Iterator over the edges used in order to build the spanning arborescence.
    ///
    /// # Arguments
    /// * `verbose`: Option<bool> - Whether to show a loading bar or not.
    ///
    /// # Raises
    /// * If the system configuration does not allow for the creation of the thread pool.
    pub fn spanning_arborescence(
        &self,
        verbose: Option<bool>,
    ) -> Result<(usize, impl Iterator<Item = (NodeT, NodeT)> + '_)> {
        self.must_be_undirected()?;
        let verbose = verbose.unwrap_or(false);
        let nodes_number = self.get_nodes_number() as usize;
        let mut parents = vec![NODE_NOT_PRESENT; nodes_number];
        let (cpu_number, pool) = get_thread_pool()?;
        let shared_stacks: Arc<Vec<Mutex<Vec<NodeT>>>> = Arc::from(
            (0..std::cmp::max(cpu_number - 1, 1))
                .map(|_| Mutex::from(Vec::new()))
                .collect::<Vec<Mutex<Vec<NodeT>>>>(),
        );
        let active_nodes_number = AtomicUsize::new(0);
        let completed = AtomicBool::new(false);
        let total_inserted_edges = AtomicUsize::new(0);
        let thread_safe_parents = ThreadDataRaceAware {
            value: std::cell::UnsafeCell::new(&mut parents),
        };

        pool.scope(|s| {
            // for each leaf of the previous stub tree start a DFS keeping track
            // of which nodes we visited and updating accordingly the parents vector.
            // the nice trick here is that, since all the leafs are part of the same tree,
            // if two processes find the same node, we don't care which one of the two take
            // it so we can proceed in a lockless fashion (and maybe even without atomics
            // if we manage to remove the colors vecotr and only keep the parents one)
            s.spawn(|_| {
                let pb = get_loading_bar(
                    verbose,
                    format!("Computing spanning tree of graph {}", self.get_name()).as_ref(),
                    nodes_number,
                );
                let parents = thread_safe_parents.value.get();
                (0..nodes_number).progress_with(pb).for_each(|src| unsafe {
                    // If the node has already been explored we skip ahead.
                    if (*parents)[src] != NODE_NOT_PRESENT {
                        return;
                    }

                    // find the first not explored node (this is guardanteed to be in a new component)
                    if self.is_unchecked_singleton_from_node_id(src as NodeT) {
                        // We set singletons as self-loops for now.
                        (*parents)[src] = src as NodeT;
                        return;
                    }
                    loop {
                        if (*parents)[src] != NODE_NOT_PRESENT {
                            break;
                        }
                        if active_nodes_number.load(Ordering::SeqCst) == 0 {
                            if (*parents)[src] != NODE_NOT_PRESENT {
                                break;
                            }
                            (*parents)[src] = src as NodeT;

                            shared_stacks[0].lock().expect("The lock is poisoned from the panic of another thread")
                                .push(src as NodeT);
                            active_nodes_number.fetch_add(1, Ordering::SeqCst);
                            break;
                        }
                    }
                });
                completed.store(true, Ordering::SeqCst);
            });
            (0..shared_stacks.len()).for_each(|_| {
                s.spawn(|_| 'outer: loop {
                    let thread_id = rayon::current_thread_index().expect("current_thread_id not called from a rayon thread. This should not be possible because this is in a Rayon Thread Pool.");
                    let src = 'inner: loop {
                        {
                            for mut stack in (thread_id..(shared_stacks.len() + thread_id))
                                .map(|id| shared_stacks[id % shared_stacks.len()].lock().expect("The lock is poisoned from the panic of another thread"))
                            {
                                if let Some(src) = stack.pop() {
                                    break 'inner src;
                                }
                            }

                            if completed.load(Ordering::SeqCst) {
                                break 'outer;
                            }
                        }
                    };
                    let parents = thread_safe_parents.value.get();
                    unsafe{self.iter_unchecked_neighbour_node_ids_from_source_node_id(src)}
                        .for_each(|dst| unsafe {
                            if (*parents)[dst as usize] == NODE_NOT_PRESENT {
                                (*parents)[dst as usize] = src;
                                total_inserted_edges.fetch_add(1, Ordering::SeqCst);
                                active_nodes_number.fetch_add(1, Ordering::SeqCst);
                                shared_stacks[rand_u64(dst as u64) as usize % shared_stacks.len()]
                                    .lock()
                                    .expect("The lock is poisoned from the panic of another thread")
                                    .push(dst);
                            }
                        });
                    active_nodes_number.fetch_sub(1, Ordering::SeqCst);
                });
            });
        });

        // convert the now completed parents vector to a list of tuples representing the edges
        // of the spanning arborescense.
        Ok((
            // Number of edges inserted
            total_inserted_edges.load(Ordering::SeqCst),
            // Return an iterator over all the edges in the spanning arborescence
            parents.into_iter().enumerate().filter_map(|(src, dst)| {
                if src as NodeT == dst {
                    return None;
                }
                Some((src as NodeT, dst))
            }),
        ))
    }

    /// Compute the connected components building in parallel a spanning tree using [bader's algorithm](https://www.sciencedirect.com/science/article/abs/pii/S0743731505000882).
    ///
    /// **This works only for undirected graphs.**
    ///
    /// This method is **not thread save and not deterministic** but by design of the algorithm this
    /// shouldn't matter but if we will encounter non-detemristic bugs here is where we want to look.
    ///
    /// The returned quadruple contains:
    /// - Vector of the connected component for each node.
    /// - Number of connected components.
    /// - Minimum connected component size.
    /// - Maximum connected component size.
    ///
    /// # Arguments
    ///
    /// * `verbose`: Option<bool> - Whether to show a loading bar or not.
    ///
    /// # Raises
    /// * If the given graph is directed.
    /// * If the system configuration does not allow for the creation of the thread pool.
    pub fn connected_components(
        &self,
        verbose: Option<bool>,
    ) -> Result<(Vec<NodeT>, NodeT, NodeT, NodeT)> {
        // TODO! refactor atomics
        self.must_be_undirected()?;
        if !self.has_nodes() {
            return Ok((Vec::new(), 0, 0, 0));
        }
        if self.get_edges_number() == 0 {
            return Ok((
                self.iter_node_ids().collect(),
                self.get_nodes_number(),
                1,
                1,
            ));
        }
        let verbose = verbose.unwrap_or(false);

        let components = self
            .iter_node_ids()
            .map(|_| AtomicU32::new(NODE_NOT_PRESENT))
            .collect::<Vec<_>>();
        let mut min_component_size: NodeT = NodeT::MAX;
        let mut max_component_size: NodeT = 0;
        let mut components_number: NodeT = 0;
        let (cpu_number, pool) = get_thread_pool()?;
        let shared_stacks: Arc<Vec<Mutex<Vec<NodeT>>>> = Arc::from(
            (0..std::cmp::max(cpu_number - 1, 1))
                .map(|_| Mutex::from(Vec::new()))
                .collect::<Vec<Mutex<Vec<NodeT>>>>(),
        );
        let active_nodes_number = AtomicUsize::new(0);
        let current_component_size = AtomicU32::new(0);
        let completed = AtomicBool::new(false);
        let thread_safe_min_component_size = ThreadDataRaceAware {
            value: std::cell::UnsafeCell::new(&mut min_component_size),
        };
        let thread_safe_max_component_size = ThreadDataRaceAware {
            value: std::cell::UnsafeCell::new(&mut max_component_size),
        };
        let thread_safe_components_number = ThreadDataRaceAware {
            value: std::cell::UnsafeCell::new(&mut components_number),
        };

        // since we were able to build a stub tree with cpu.len() leafs,
        // we spawn the treads and make anyone of them build the sub-trees.
        pool.scope(|s| {
            // for each leaf of the previous stub tree start a DFS keeping track
            // of which nodes we visited and updating accordingly the components vector.
            // the nice trick here is that, since all the leafs are part of the same tree,
            // if two processes find the same node, we don't care which one of the two take
            // it so we can proceed in a lockless fashion (and maybe even without atomics
            // if we manage to remove the colors vecotr and only keep the components one)
            s.spawn(|_| {
                let pb = get_loading_bar(
                    verbose,
                    format!(
                        "Computing connected components of graph {}",
                        self.get_name()
                    )
                    .as_ref(),
                    self.get_nodes_number() as usize,
                );
                let min_component_size = thread_safe_min_component_size.value.get();
                let max_component_size = thread_safe_max_component_size.value.get();
                let components_number = thread_safe_components_number.value.get();
                self.iter_node_ids()
                    .progress_with(pb)
                    .for_each(|src| {
                        // If the node has already been explored we skip ahead.
                        if components[src as usize].load(Ordering::Relaxed) != NODE_NOT_PRESENT {
                            return;
                        }

                        // find the first not explored node (this is guardanteed to be in a new component)
                        if self.has_disconnected_nodes()
                            && (unsafe{self.is_unchecked_singleton_from_node_id(src)
                                || self.is_unchecked_singleton_with_selfloops_from_node_id(src)})
                        {
                            // We set singletons as self-loops for now.
                            unsafe {
                                components[src as usize]
                                    .store(**components_number, Ordering::Relaxed);
                                **components_number += 1;
                                **min_component_size = 1;
                                **max_component_size = (**max_component_size).max(1);
                            }
                            return;
                        }

                        loop {
                            // if the node has been now mapped to a component by anyone of the
                            // parallel threads, move on to the next node.
                            if components[src as usize].load(Ordering::Relaxed) != NODE_NOT_PRESENT {
                                break;
                            }
                            // Otherwise, Check if the parallel threads are finished
                            // and are all waiting for a new node to explore.
                            // In that case add the currently not explored node to the
                            // work stack of the first thread.
                            if active_nodes_number.load(Ordering::Relaxed) == 0 {
                                // The check here might seems redundant but its' needed
                                // to prevent data races.
                                //
                                // If the last parallel thread finishes its stack between the
                                // presence check above and the active nodes numbers check
                                // the src node will never increase the component size and thus
                                // leading to wrong results.
                                if components[src as usize].load(Ordering::Relaxed) != NODE_NOT_PRESENT {
                                    break;
                                }
                                let ccs =
                                    current_component_size.swap(1, Ordering::Relaxed) as NodeT;
                                unsafe {
                                    **max_component_size = (**max_component_size).max(ccs);
                                    if ccs > 1 {
                                        **min_component_size = (**min_component_size).min(ccs);
                                    }
                                    components[src as usize]
                                        .store(**components_number, Ordering::Relaxed);
                                    **components_number += 1;
                                }
                                active_nodes_number.fetch_add(1, Ordering::Relaxed);
                                shared_stacks[0].lock().expect("The lock is poisoned from the panic of another thread").push(src);
                                break;
                            }
                            // Otherwise, Loop until the parallel threads are finished.
                        }
                    });
                completed.store(true, Ordering::Relaxed);
            });

            // Spawn the parallel threads that handle the components mapping,
            // these threads use work-stealing, meaning that if their stack is empty,
            // they will steal nodes from the stack of another random thread.
            (0..shared_stacks.len()).for_each(|_| {
                s.spawn(|_| {
                    let thread_id = rayon::current_thread_index().expect("current_thread_id not called from a rayon thread. This should not be possible because this is in a Rayon Thread Pool.");
                    'outer: loop {
                    // get the id, we use this as an idex for the stacks vector.

                    let src = 'inner: loop {
                        {
                            for mut stack in (thread_id..(shared_stacks.len() + thread_id))
                                .map(|id| shared_stacks[id % shared_stacks.len()].lock().expect("The lock is poisoned from the panic of another thread"))
                            {
                                if let Some(src) = stack.pop() {
                                    break 'inner src;
                                }
                            }

                            if completed.load(Ordering::Relaxed) {
                                break 'outer;
                            }
                        }
                    };

                    let src_component = components[src as usize].load(Ordering::Relaxed);
                    unsafe{self.iter_unchecked_neighbour_node_ids_from_source_node_id(src)}
                        .for_each(|dst| {
                            if components[dst as usize].swap(src_component, Ordering::SeqCst)
                                == NODE_NOT_PRESENT
                            {
                                active_nodes_number.fetch_add(1, Ordering::SeqCst);
                                current_component_size.fetch_add(1, Ordering::SeqCst);
                                shared_stacks[rand_u64(dst as u64) as usize % shared_stacks.len()]
                                    .lock()
                                    .expect("The lock is poisoned from the panic of another thread")
                                    .push(dst);
                            }
                        });
                    active_nodes_number.fetch_sub(1, Ordering::SeqCst);
                }});
            });
        });

        let ccs = current_component_size.load(Ordering::SeqCst);
        max_component_size = max_component_size.max(ccs);
        if ccs > 1 {
            min_component_size = min_component_size.min(ccs);
        }

        Ok((
            unsafe { std::mem::transmute::<Vec<AtomicU32>, Vec<NodeT>>(components) },
            components_number,
            min_component_size,
            max_component_size,
        ))
    }
}

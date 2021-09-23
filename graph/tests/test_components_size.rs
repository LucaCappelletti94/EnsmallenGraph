extern crate graph;
use graph::{utils::get_loading_bar, EdgeFileReader, Graph};
use std::collections::HashMap;

#[test]
/// Test that the components number is reasonable, this raised the wierd case in which:
/// singletons: false selfloops: false smallest: 1 biggest: 3, edges: [(0, 1), (2, 3), (4, 5)]
fn test_components_size() {
    let graph_name = "ComponentSizeTest".to_owned();
    let edges_reader = EdgeFileReader::new("tests/data/test_components.csv")
        .unwrap()
        .set_header(Some(false))
        .unwrap()
        .set_separator(Some(",".to_string()))
        .unwrap()
        .set_verbose(Some(false))
        .set_numeric_node_ids(Some(true));

    let g = Graph::from_file_readers(
        Some(edges_reader),
        None,
        None,
        None,
        true,
        true,
        false,
        graph_name.clone(),
    )
    .unwrap();

    // THIS IS NOT DETERMINISTIC
    let n = 10_000;
    let pb = get_loading_bar(true, "Executing connected components test", n);
    for _ in 0..n {
        pb.inc(1);
        let (components, _components_number, smallest, biggest) =
            g.get_connected_components(None).unwrap();
        assert!(
            biggest >= smallest,
            "smallest: {} biggest: {}",
            smallest,
            biggest
        );

        if g.has_disconnected_nodes() {
            assert!(smallest == 1);
        }
        if smallest == 1 {
            assert!(
                g.has_disconnected_nodes(),
                concat!(
                    "For the minimum connected component to have a single node, the graph ",
                    "must contain disconnected nodes.\n",
                    "The node degrees of this graph is {:?}.\n",
                    "The directed flag of this graph is {:?}.\n",
                    "The edge list of this graph is {:?}.\n",
                    "The component node components are: {:?}.\n"
                ),
                g.get_node_degrees(),
                g.is_directed(),
                g.get_edge_node_ids(true),
                components
            );
        }
    }

    let (components, number_of_components, smallest, biggest) =
        g.get_connected_components(None).unwrap();

    assert_eq!(components, [0, 0, 1, 1, 2, 2].to_vec());
    assert_eq!(number_of_components, 3);
    assert_eq!(smallest, 2); // the size of the smallest component
    assert_eq!(biggest, 2); // the size of the biggest component

    let (number_of_components2, smallest2, biggest2) = g.get_connected_components_number(None);
    assert_eq!(number_of_components, number_of_components2, "There is a difference between the number of components returned by the connected_components method and the connected_components_number.");
    assert_eq!(smallest, smallest2, "There is a difference between the smallest returned by the connected_components method and the connected_components_number.");
    assert_eq!(biggest, biggest2, "There is a difference between the biggest returned by the connected_components method and the connected_components_number.");

    let mut components_size = HashMap::new();
    for component_index in &components {
        let counter = components_size.entry(*component_index).or_insert(0);
        *counter += 1;
    }

    assert_eq!(
        number_of_components as usize,
        components_size.len(),
        "The number of components is wrong!"
    );
    assert_eq!(
        smallest,
        *components_size.values().min().unwrap(),
        "The smallest is wrong!"
    );
    assert_eq!(
        biggest,
        *components_size.values().max().unwrap(),
        "The biggest is wrong!"
    );
}

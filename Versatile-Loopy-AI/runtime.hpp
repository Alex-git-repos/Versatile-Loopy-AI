#pragma once
#include "data_structures_containers.hpp"

class runtime {
public:
	void collect_input_data(model& ai, input_data& in_dat, id_map& ident_map);
	void collect_training_data(model& ai, training_data& input, id_map& ident_map);
	void recursive_search_adjust(node_id current_node, id_map& id_mapping, node_stack& connection_mapping, net_info& model_data, input_data& in_dat, error_in_acts& error);
	void train_model(model& ai, id_map& ident_map, lid_map& lident_map, node_stack& conn_map, net_info& model_data);
	void run_model(id_map& ident_map, lid_map& lident_map, node_stack& conn_map, net_info& model_data, input_data& in_dat);
};

#include "running.ipp"
#include "training.ipp"
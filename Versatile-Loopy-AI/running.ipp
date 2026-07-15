#include "data_structures_containers.hpp"
#include <numeric>
#include <iostream>
#include <algorithm>
using namespace std;

void runtime::collect_input_data(model& ai, input_data& in_dat, id_map& ident_map) {
	boost::json::object json_main_obj{ ai.main_obj };
	float usr_input;
	in_dat.in_map.clear();

	cout << "\n----------\n\n";
	for (auto& node : json_main_obj) {
		if (separate_functions::check_is_node(node) && strcmp(node.value().as_object().at("flag").as_string().c_str(), "input") == 0) {
			cout << "Please enter input for " << node.key_c_str() << ": ";
			cin >> usr_input;
			in_dat.in_map.emplace(ident_map.node_id_mapping.at(node.key_c_str()), usr_input);
		}
	}

	return;
}

void runtime::run_model(id_map& ident_map, lid_map& lident_map, node_stack& conn_map, net_info& model_data, input_data& in_dat) {
	map<node_id, float> inputs_map{ in_dat.in_map };
	map<node_id, float> outputs_map;
	node_id node_name;
	in_dat.activations_map = in_dat.in_map;

	for (long long lid_counter{ 0 }; lid_counter < model_data.max_layer_id.layer_id; ++lid_counter) {
		for (auto lident_map_it{ lident_map.reverse_lid_mapping.lower_bound(lid{ lid_counter }) }; lident_map_it != lident_map.reverse_lid_mapping.upper_bound(lid{ lid_counter }); ++lident_map_it) {
			node_name = ident_map.node_id_mapping.at(lident_map_it->second);
			for (auto conn_map_it{ conn_map.connection_mapping.lower_bound(node_name) }; conn_map_it != conn_map.connection_mapping.upper_bound(node_name); ++conn_map_it) {
				conn_map.stack_copy = {
					node_name,
					conn_map_it->second
				};
				cout << "Stack_copy set to " << conn_map.stack_copy.first.id << ", " << conn_map.stack_copy.second.id << "\n";

				vector<float> weights_copy;
				long long x_counter{ 0 };
				while (true) {
					matrix_id current_matrix_value{ .p = conn_map.stack_copy.second, .c = conn_map.stack_copy.first, .x = x_counter++ };

					if (model_data.weights.find(current_matrix_value) != model_data.weights.end()) {
						weights_copy.push_back(model_data.weights.at(current_matrix_value));
					}
					else break;
				}
				cout << "Weights_copy set\n";

				vector<float> outputs_vec;
				float current_input{ inputs_map.at(conn_map.stack_copy.first) };

				for (auto& weight : weights_copy) {
					// Multiply current weight by input, then add the distributed bias
					outputs_vec.push_back((current_input * weight) + model_data.biases.at(conn_map.stack_copy.second));
				}
				cout << "Outputs_vec solved\n\n";

				// Sum outputs and average
				float final_contribution{ accumulate(outputs_vec.begin(), outputs_vec.end(), float { 0 }) / static_cast<float>(outputs_vec.size()) };
				in_dat.logit_collection_map.emplace(conn_map.stack_copy.first, final_contribution);
				while (true) {
					if (outputs_map.find(conn_map.stack_copy.second) == outputs_map.end()) {
						outputs_map.emplace(conn_map.stack_copy.second, 0);
					}
					else {
						outputs_map.at(conn_map.stack_copy.second) += final_contribution;
						break;
					}
				}
			}
		}

		map<size_t, node_id> vector_mapping;
		vector<float> logits;
		
		inputs_map.clear();

		for (auto& logit : outputs_map) {
			vector_mapping.emplace(logits.size(), logit.first);
			logits.push_back(logit.second);
		}
		float max_logit{ ranges::max(logits) };

		// Softsign for activation
		float current_activation{ 0 };
		for (size_t it{ 0 }; it < logits.size(); ++it) {
			current_activation = logits.at(it) / (1 + abs(logits.at(it)));
			inputs_map.emplace(vector_mapping.at(it), current_activation);
			in_dat.activations_map.emplace(vector_mapping.at(it), current_activation);
		}

		outputs_map.clear();
	}

	for (auto& elem : inputs_map) {
		for (auto& entry : ident_map.node_id_mapping) {
			if (elem.first == entry.second) {
				cout << "Node: " << entry.first;
			}
		}
		cout << ", Value: " << elem.second << "\n";
	}
	in_dat.in_map = inputs_map;

	return;
}
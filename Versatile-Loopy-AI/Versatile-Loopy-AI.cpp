#include "data_structures_containers.hpp"
#include "runtime.hpp"
#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>
#include <boost/json.hpp>

using namespace std;

/*
	BEWARE! Logits only have one bias. A vec. of logits sum with the bias is a "distribution": logits[1, 2, 3] + bias[0.5] = biased_logits[1 + 0.5, 2 + 0.5, 3 + 0.5].
	BEWARE! NIDs, LIDs, and MIDs with values of -2 refer to blank, unset values.
	FUTURE IMPROVEMENTS:
*/

namespace parse_funcs {
	void find_max_lid(model& ai, net_info& input) {
		input.max_layer_id = { .layer_id = ai.main_obj.at("max_lid").as_int64() };
		return;
	}

	void parse_into_id_mapping(model& ai, id_map& input) {
		// Heads up: this function is chill
		boost::json::object json_main_obj{ ai.main_obj };
		string node_name{ "\0" };
		long long node_identification{ 0 };

		for (auto& node : json_main_obj) {
			if (separate_functions::check_is_node(node)) {
				node_name = node.key_c_str();
				node_identification = node.value().as_object().at("id").as_int64();
				input.node_id_mapping.emplace(node_name, node_id{ node_identification });
				input.reverse_id_mapping.emplace(node_id{ node_identification }, node_name);
			}
		}
		return;
	}

	void parse_into_lid_mapping(model& ai, lid_map& input) {
		// Heads up: this function is also chill
		boost::json::object json_main_obj{ ai.main_obj };
		string node_name{ "\0" };
		long long node_lident{ 0 };

		for (auto& node : json_main_obj) {
			if (separate_functions::check_is_node(node)) {
				node_name = node.key_c_str();
				node_lident = node.value().as_object().at("lid").as_int64();
				input.lid_mapping.emplace(node_name, lid{ node_lident });
				input.reverse_lid_mapping.emplace(lid{ node_lident }, node_name);
			}
		}
		return;
	}

	void parse_into_conn_map(model& ai, id_map& ident_map, node_stack& input) {
		boost::json::object json_main_obj{ ai.main_obj };

		for (auto& node : json_main_obj) {
			if (separate_functions::check_is_node(node) && (node.value().as_object().at("lid").as_int64() != 0)) {
				node_id parent_id{ ident_map.node_id_mapping.at(node.key_c_str()) };
				boost::json::array connections_array{ node.value().as_object().at("connections").as_array() };

				for (auto& connection : connections_array) {
					node_id child_id{ ident_map.node_id_mapping.at(connection.as_string().c_str())};
					input.connection_mapping.emplace(child_id, parent_id);
					input.reverse_connection_mapping.emplace(parent_id, child_id);
				}
			}
		}
	}

	void parse_net_info(model& ai, id_map& ident_map, net_info& input) {
		// This function was a nightmare. It has been reworked.
		boost::json::object json_main_obj{ ai.main_obj };

		for (auto& node : json_main_obj) {
			if (separate_functions::check_is_node(node) && (node.value().as_object().at("lid").as_int64() != 0)) {
				boost::json::object weights{ node.value().as_object().at("weights").as_object() };
				node_id node_nid{ ident_map.node_id_mapping.at(node.key_c_str()) };

				for (auto& weights_entry : weights) {
					node_id entry_nid{ ident_map.node_id_mapping.at(weights_entry.key_c_str()) };
					boost::json::array weights_entry_arr{ weights_entry.value().as_array() };
					long long x_value{ 0 };

					for (auto& weight_value : weights_entry_arr) {
						matrix_id current_mid{ .p = node_nid, .c = entry_nid, .x = x_value++ };
						input.weights.emplace(current_mid, weight_value.to_number<float>());
					}
				}

				// Okie, now we can be calmer, for we are parsing the biases...
				float current_bias{ node.value().as_object().at("bias").to_number<float>() };
				input.biases.emplace(ident_map.node_id_mapping.at(node.key_c_str()), current_bias);
			}
		}
		return;
	}
}

int main() {
	model ai_model_in_ram;
	node_stack conn_maps;
	id_map idf_mapping;
	lid_map lidf_mapping;
	net_info model_data;
	input_data in_dat;
	runtime executer;

	cout << "Initialized all containers...\n";
	parse_funcs::parse_into_id_mapping(ai_model_in_ram, idf_mapping);
	cout << "Parsed NID mappings\n";
	parse_funcs::parse_into_lid_mapping(ai_model_in_ram, lidf_mapping);
	cout << "Parsed LID mappings\n";
	parse_funcs::parse_into_conn_map(ai_model_in_ram, idf_mapping, conn_maps);
	cout << "Parsed node connections\n";
	parse_funcs::parse_net_info(ai_model_in_ram, idf_mapping, model_data);
	cout << "Parsed NN data\n";
	parse_funcs::find_max_lid(ai_model_in_ram, model_data);

	cout << "\n----------\n\n";
	cout << "Connections:\n";
	for (auto& elem : conn_maps.connection_mapping) {
		cout << "First node: " << elem.first.id << ", Second node: " << elem.second.id << "\n";
	}
	cout << "Weights:\n";
	for (auto& elem : model_data.weights) {
		cout << "Parent NID: " << elem.first.p.id << ", Child NID: " << elem.first.c.id << ", X value: " << elem.first.x << ", Weight: " << elem.second << "\n";
	}
	
	char option_selected{ '\0' };
	bool will_prompt{ true };

	while (true) {
		if (will_prompt) {
			cout << "\n----------\n\nWould you like to train or run the model (t/r)? ";
			cin >> option_selected;
		}
		else ignore;

		switch (option_selected) {
		case 't':
		case 'T':
			executer.train_model(ai_model_in_ram, idf_mapping, lidf_mapping, conn_maps, model_data);
			will_prompt = true;
			break;
		case 'r':
		case 'R':
			executer.collect_input_data(ai_model_in_ram, in_dat, idf_mapping);
			executer.run_model(idf_mapping, lidf_mapping, conn_maps, model_data, in_dat);
			will_prompt = true;
			break;
		default:
			cout << "Invalid option. Please choose again (t/r): ";
			cin >> option_selected;
			will_prompt = false;
			break;
		}
	}

	return 0;
}
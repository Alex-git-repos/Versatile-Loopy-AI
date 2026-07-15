#include "data_structures_containers.hpp"
#include <xutility>
using namespace std;

void runtime::collect_training_data(model& ai, training_data& input, id_map& ident_map) {
	boost::json::object json_main_obj{ ai.main_obj };
	boost::json::object tdat{ json_main_obj.at("train_dat").as_object() };
	boost::json::value ramount{ json_main_obj.at("correct_amount_requirement") };
	boost::json::value tolerance{ json_main_obj.at("tolerance") };

	unsigned long long iteration{ 0 };
	for (auto& entry : tdat) {
		input.train_dat_in.push_back({});
		for (auto& element : entry.value().as_object().at("in").as_object()) {
			input.train_dat_in.at(iteration).emplace(ident_map.node_id_mapping.at(element.key_c_str()), element.value().to_number<float>());
		}
		input.train_dat_out.push_back({});
		for (auto& element : entry.value().as_object().at("out").as_object()) {
			input.train_dat_out.at(iteration++).emplace(ident_map.node_id_mapping.at(element.key_c_str()), element.value().to_number<float>());
		}
	}
	input.correct_amount_requirement = ramount.to_number<unsigned long long>();
	input.tolerance = tolerance.to_number<float>();
	return;
}

void runtime::recursive_search_adjust(node_id current_node, id_map& id_mapping, node_stack& connection_mapping, net_info& model_data, input_data& in_dat, error_in_acts& error) {
	cout << "\n\nBranching into " << current_node.id << "\n";

	// First find n, the number of child nodes.
	size_t n{ connection_mapping.reverse_connection_mapping.count(current_node) };
	
	cout << "Child node count: " << n << "\n";

	// Then adjust the child weights
	if (connection_mapping.reverse_connection_mapping.contains(current_node)) {
		vector<float> zsum_vec;
		for (auto sum_it{ connection_mapping.reverse_connection_mapping.lower_bound(current_node) }; sum_it != connection_mapping.reverse_connection_mapping.upper_bound(current_node); ++sum_it) {
			zsum_vec.push_back(abs(in_dat.logit_collection_map.at(sum_it->second)));
		}

		for (auto conn_it{ connection_mapping.reverse_connection_mapping.lower_bound(current_node) }; conn_it != connection_mapping.reverse_connection_mapping.upper_bound(current_node); ++conn_it) {
			cout << "-> Next iteration\n";
			node_id child_node{ conn_it->second };
			cout << "Found child node of " << child_node.id << "\n";

			long long x_value{ 0 };
			map<matrix_id, float> weights_copy;
			while (true) {
				matrix_id partial_matrix{ .p = current_node, .c = child_node, .x = x_value++ };
				if (model_data.weights.contains(partial_matrix)) {
					cout << "Found weight of " << partial_matrix.c.id << " -> " << partial_matrix.p.id << " @x=" << partial_matrix.x << " with value " << model_data.weights.at(partial_matrix) << "\n";
					weights_copy.emplace(partial_matrix, model_data.weights.at(partial_matrix));
				}
				else break;
			}

			float zsum{ accumulate(zsum_vec.begin(), zsum_vec.end(), float { 0 }) };
			float a_over_asum{ abs(in_dat.logit_collection_map.at(child_node)) / zsum };
			float lcount_to_nfirst{ float { 1 } / model_data.max_layer_id.layer_id };
			float deltaA{ error.errors.at(error.overarching_node) * a_over_asum * lcount_to_nfirst };
			float deltaZ{ deltaA / (1 - abs(deltaA)) }; // Inverse Softsign

			for (auto& weight : weights_copy) {
				float deltaW{ 0 };
				if (in_dat.activations_map.at(child_node) != 0) {
					deltaW = deltaZ / in_dat.activations_map.at(child_node);
					cout << "DeltaW set to " << deltaW << "\n";
				}
				else {
					deltaW = deltaZ;
					cout << "DeltaW set to " << deltaW << "\n";
				}
					
				model_data.weights.at(weight.first) += deltaW;
			}
			recursive_search_adjust(child_node, id_mapping, connection_mapping, model_data, in_dat, error);
		}
	}

	cout << "Returning\n\n";
	return;
}

void runtime::train_model(model& ai, id_map& ident_map, lid_map& lident_map, node_stack& conn_map, net_info& model_data) {
	cout << "\nStarted training\n";

	training_data train_dat;
	collect_training_data(ai, train_dat, ident_map);

	unsigned long long amount_correct{ 0 };
	unsigned long long iteration{ 0 };
	bool will_quit{ false };
	while (!will_quit) {
		input_data in_dat(train_dat.train_dat_in.at(iteration));
		run_model(ident_map, lident_map, conn_map, model_data, in_dat);
		cout << "Iteration is " << iteration << "\n";

		map<node_id, float> errors_map{ in_dat.in_map };
		vector<float> maxfinder_vec;
		for (auto& error : errors_map) {
			errors_map.at(error.first) = train_dat.train_dat_out.at(iteration).at(error.first) - error.second;
			maxfinder_vec.push_back(abs(errors_map.at(error.first)));
		}

		if (ranges::max(maxfinder_vec) <= train_dat.tolerance) {
			++amount_correct;
			if (amount_correct >= train_dat.correct_amount_requirement) will_quit = true;
		}
		else amount_correct = 0;

		if (!will_quit) {
			for (auto lident_it{ lident_map.reverse_lid_mapping.lower_bound(model_data.max_layer_id) }; lident_it != lident_map.reverse_lid_mapping.upper_bound(model_data.max_layer_id); ++lident_it) {
				error_in_acts error(ident_map.node_id_mapping.at(lident_it->second));
				error.errors = errors_map;
				recursive_search_adjust(error.overarching_node, ident_map, conn_map, model_data, in_dat, error);
			}
		}

		++iteration;
		if (train_dat.train_dat_in.size() == iteration) iteration = 0;
	}

	for (auto& weight : model_data.weights) {
		cout << "Node " << ident_map.reverse_id_mapping.at(weight.first.c) << " -> " << ident_map.reverse_id_mapping.at(weight.first.p) << " @x=" << weight.first.x << " has new weight of " << weight.second << "\n";
	}
	cout << "Training complete\n";
	return;
}
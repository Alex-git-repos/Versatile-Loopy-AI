#pragma once
using namespace std;
#include <string>
#include <vector>
#include <stdexcept>
#include <unordered_map>
#include <compare>
#include <map>
#include <boost/json.hpp>
#include <sstream>
#include <fstream>

struct node_id {
	long long id{ 0 };
	auto operator<=>(const node_id&) const = default;
};

struct lid {
	long long layer_id{ 0 };
	auto operator<=>(const lid&) const = default;
};

struct matrix_id {
	// NID
	node_id p{ -2 };
	// NID
	node_id c{ -2 };
	// ENTRY
	long long x{ -2 };
	auto operator<=>(const matrix_id&) const = default;
};

class id_map {
public:
	map<string, node_id> node_id_mapping;
	map<node_id, string> reverse_id_mapping;
};

class lid_map {
public:
	map<string, lid> lid_mapping;
	multimap<lid, string> reverse_lid_mapping;
};

class node_stack {
public:
	multimap<node_id, node_id> connection_mapping;
	multimap<node_id, node_id> reverse_connection_mapping;
	pair<node_id, node_id> stack_copy;
};

class net_info {
public:
	map<matrix_id, float> weights;
	map<node_id, float> biases;
	lid max_layer_id;
};

class model {
public:
	boost::json::object main_obj;

	model(string path = { "./" }, string name = { "template.json" }) {
		string formatted_path{ path + name };
		ifstream model_json(formatted_path);
		if (!model_json.is_open()) {
			model_json.open(formatted_path, ios::in);
		}

		stringstream buf_access;
		buf_access << model_json.rdbuf();

		boost::json::value model_val{ boost::json::parse(buf_access.str()) };
		main_obj = model_val.as_object();
		model_json.close();
		return;
	}
};

class training_data {
public:
	unsigned long long correct_amount_requirement{ 0 };
	float tolerance{ 0 };
	vector<map<node_id, float>> train_dat_in;
	vector<map<node_id, float>> train_dat_out;
};

namespace separate_functions {
	bool check_is_node(boost::json::object::value_type node) {
		return (strcmp(node.key_c_str(), "train_dat") != 0) && (strcmp(node.key_c_str(), "correct_amount_requirement") != 0) && (strcmp(node.key_c_str(), "max_lid") != 0) && (strcmp(node.key_c_str(), "tolerance") != 0);
	}
	bool check_is_tdat(boost::json::object::value_type node) {
		return strcmp(node.key_c_str(), "train_dat") == 0;
	}
	bool check_is_ramount(boost::json::object::value_type node) {
		return strcmp(node.key_c_str(), "correct_amount_requirement") == 0;
	}
}

class input_data {
public:
	map<node_id, float> in_map;
	map<node_id, float> activations_map;
	map<node_id, float> logit_collection_map;

	input_data(map<node_id, float> initializer = {}) : in_map{ initializer } {
		return;
	}
};

class error_in_acts {
public:
	map<node_id, float> errors;
	node_id overarching_node;
	vector<float> hits;

	error_in_acts(node_id initializer = { .id = -2 }) : overarching_node{ initializer } {
		return;
	}
};
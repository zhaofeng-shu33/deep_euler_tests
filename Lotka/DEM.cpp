/*Deep euler implementation*/

#include <iostream>
#include <fstream>
#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include <regex>

#include <boost/program_options.hpp>
#include <torch/script.h>

using namespace std;

//const int N = 16;
//const int N_z = N / 2 - 1;
const int nn_inputs = 4;
const int nn_outputs = 2; // output dimension
c10::TensorOptions global_tensor_op;

string file_name = "../lotka_dem.txt";
string file_name_normal_euler = "../lotka_euler.txt";
string time_counting_file_name = "../clock.txt";
string model_file = "../../training/traced_model_e20_2021_11_04.pt";
string range_model_file = "../../training/traced_range_model_e20_2021_11_04.pt";
string embedded_model_file = "../../training/traced_range_embedded_model_e22_2021_11_10.pt";
static bool use_embedded = false;
static bool use_generalized = false;

typedef double value_type;
typedef vector<value_type> state_type;




//ode function of bubble dynamic
class lotka {
//	std::ofstream outputs_out;
public:
	torch::jit::script::Module model;
	torch::Tensor inputs; //reused tensor of inputs

	lotka(std::vector<double> inital_values) {

		//metamodel initializations
		int _size = inital_values.size();
		inputs = torch::ones({ 1, _size }, global_tensor_op);
		//model inputs: dt x1 x2 x3 z... x1now x2now x3now
		for (int i = 0; i < _size; i++) inputs[0][i] = inital_values[i];
		//outputs: z... grad_z(wall)
		try {
			if (use_embedded) {
				model = torch::jit::load(embedded_model_file);
			}
			else if(use_generalized) {
				model = torch::jit::load(range_model_file);
			}
			else {
				model = torch::jit::load(model_file);
			}
			std::vector<torch::jit::IValue> inp;
			inp.push_back(torch::ones({ 1, _size }, global_tensor_op));
			std::cout << inp << endl;
			// Execute the model and turn its output into a tensor.
			at::Tensor output = model.forward(inp).toTensor().detach();
			std::cout << output << endl;
		}
		catch (const c10::Error& e) {
			std::cerr << "Error loading the model: " << e.what() << endl;
			//	exit(-1);
		}
	}


	/*Rewrites the errors array with the predicted local truncation errors*/
	void local_error(double t, double t_next, const double* x, double* errors) {
		//updating inputs
		inputs[0][0] = t_next;
		inputs[0][1] = t;
		for (int i = 0; i < nn_inputs - 2; i++) {
			inputs[0][i + 2] = x[i];
		}
		std::vector<torch::jit::IValue> inps;
		inps.push_back(inputs);
		//evaluating
		torch::Tensor loc_trun_err = model.forward(inps).toTensor().detach();

		for (int i = 0; i < nn_outputs; i++) {
			errors[i] = loc_trun_err[0][i].item<double>();
		}
	}

	/*ODE function*/
	inline void operator()(double t, double* x, double* dx) {
		dx[0] = x[0] - x[0] * x[1];
		dx[1] = -x[1] + x[0] * x[1];
	}
};

class ODESolver
{
public:
	// order is the number of the output
	ODESolver(int order): order(order) {
	// initialize necessary data structures
		derivative = (double*)malloc(sizeof(double) * order);
		local_error = (double*)malloc(sizeof(double) * order);
	};

	bool setInitialCondition(double* conds, double at) {
		begin_t = at;
		init_conds = (double*)malloc(sizeof(double) * order);
		for (int u = 0; u < order; u++) {
			init_conds[u] = conds[u];
		}
		return true;
	}

	void setTimeStep(double dt) {
		delta_t = dt;
	}

	bool setStepNumber(int steps) {
		max_l = steps;
		sol_t = (double*)malloc(sizeof(double) * (max_l + 1));
		sol = new double* [max_l + 1];
		for (int i = 0; i < max_l + 1; i++)
			sol[i] = new double[order];
		return true;
	}
	
	void solve(lotka& lot) {

		for (int u = 0; u < order; u++) {
			sol[0][u] = init_conds[u];
		}

		double t = begin_t;
		sol_t[0] = t;
		int l = 0;
		while (l < max_l) {
			lot.local_error(t, t + delta_t, sol[l], local_error); // fill the local_error
			lot(t, sol[l], derivative); // fill the derivative
			for (int j = 0; j < order; j++) {
				sol[l + 1][j] = sol[l][j] + delta_t * derivative[j] + delta_t * delta_t * local_error[j];
			}
			l++;
			t += delta_t;
			sol_t[l] = t;
		}
	}

	// disable residue approximation
	void solve_normal(lotka& lot) {

		for (int u = 0; u < order; u++) {
			sol[0][u] = init_conds[u];
		}

		double t = begin_t;
		sol_t[0] = t;
		int l = 0;
		while (l < max_l) {
			lot(t, sol[l], derivative); // fill the derivative
			for (int j = 0; j < order; j++) {
				sol[l + 1][j] = sol[l][j] + delta_t * derivative[j];
			}
			l++;
			t += delta_t;
			sol_t[l] = t;
		}
	}

	// output sol_t and sol
	void output(ofstream& out) {
		for (int i = 0; i <= max_l; i++) {
			out << sol_t[i];
			for (int j = 0; j < order; j++) {
				out << ' ' << sol[i][j];
			}
			out << '\n';
		}
	}

	~ODESolver(){
		free(init_conds);
		free(derivative);
		free(sol_t);
		for (int indx = 0; indx <= max_l; ++indx)
		{
			delete sol[indx];
		}
		delete[] sol;
	}
private:
	int order = 1;
	double* local_error;
	double* init_conds;
	double* sol_t;
	double** sol;
	double* derivative;
	double begin_t = 0;
	double delta_t = 0.1;
	int max_l = 10;
};


void setup_ofstream(ofstream& ofs) {
	if(!ofs.is_open())exit(-1);
	ofs.precision(17);
	ofs.flags(ios::scientific);
}

int main(int argc, const char* argv[]) {
	boost::program_options::options_description desc;
	desc.add_options()
		("help,h", "Show this help screen")
		("embedded", boost::program_options::value<bool>()->implicit_value(true)->default_value(false), "whether to use embedded model")
		("generalized", boost::program_options::value<bool>()->implicit_value(true)->default_value(false), "whether to use embedded model");

	boost::program_options::variables_map vm;
	boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
	boost::program_options::notify(vm);
	if (vm.count("help")) {
		std::cout << desc << '\n';
		return 0;
	}
	use_embedded = vm["embedded"].as<bool>();
	use_generalized = vm["generalized"].as<bool>();
	global_tensor_op = torch::TensorOptions().dtype(torch::kFloat64);
	std::cout << "Lotka Volterra with meta-model started\n" << setprecision(17) << endl;

	double* x = new double[nn_outputs]{ 2.0, 1.0 };

	double dem_step_list[] = {0.1, 0.05, 0.01, 0.005};
	double euler_step_list[] = { 0.002, 0.001, 0.0005, 0.0002};
	int step_list_length = (sizeof(dem_step_list) / sizeof(*dem_step_list));
	std::regex _regex(".txt");
	std::string replace_base;
	if (use_embedded) {
		replace_base = "_embedded.txt";
	}
	else if (use_generalized) {
		replace_base = "_generalized.txt";
	}
	else {
		replace_base = ".txt";
	}
	std::string time_counting_file_name_special = std::regex_replace(time_counting_file_name, _regex, replace_base);
	ofstream clock_of(time_counting_file_name_special);
	double t_stop = 15.0;
	//initial conditions
	std::vector<double> initial_inputs;
	if (use_generalized) {
		initial_inputs = { 0.0, 0.1, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0 };
	}
	else {
		initial_inputs = { 1e-5, 0.0, 2.0, 1.0 };
	}
	for (int i = 0; i < step_list_length; i++) {

		std::string file_name_i = std::regex_replace(file_name, _regex, std::to_string(i) + replace_base);
		std::string file_name_normal_euler_i = std::regex_replace(file_name_normal_euler, _regex, std::to_string(i) + ".txt");

		ofstream ofs(file_name_i);
		setup_ofstream(ofs);
		std::cout << "Writing file: " << file_name_i << endl;
		ofstream ofs_2(file_name_normal_euler_i);
		setup_ofstream(ofs_2);

		
		
		double t_start = 0.0;
		lotka bubi(initial_inputs);
		
		ODESolver solver(nn_outputs), solver_2(nn_outputs);
		solver.setInitialCondition(x, 0.0);
		solver.setTimeStep(dem_step_list[i]);
		solver.setStepNumber(int(t_stop/ dem_step_list[i]));

		std::cout << "Solving..." << endl;
		auto t1 = chrono::high_resolution_clock::now();
		solver.solve(bubi);
		auto t2 = chrono::high_resolution_clock::now();
		// std::cout << "DEM Time (ms):" << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << endl;
		clock_of << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
		solver.output(ofs);

		solver_2.setInitialCondition(x, 0.0);
		solver_2.setTimeStep(euler_step_list[i]);
		solver_2.setStepNumber(int(t_stop / euler_step_list[i]));
		t1 = chrono::high_resolution_clock::now();
		solver_2.solve_normal(bubi);
		t2 = chrono::high_resolution_clock::now();
		// std::cout << "EM Time (ms):" << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << endl;
		clock_of << ' ' << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << endl;
		solver_2.output(ofs_2);
		ofs.flush();
		ofs.close();
		ofs_2.flush();
		ofs_2.close();
	}
	clock_of.close();
	std::cout << "Ready"<< endl;
	return 0;
}
#include <boost/numeric/odeint.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/program_options.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#define _USE_MATH_DEFINES
#include <math.h>
#include "step_adjuster.hpp"
#include "runge_kutta_bs3.hpp"

using namespace boost::numeric::odeint;
typedef std::vector< double > state_type;
typedef custom_controlled_runge_kutta< runge_kutta_dopri5< state_type >, custom_error_checker< double, range_algebra, default_operations >, custom_step_adjuster<double, double>> RK45;
typedef custom_controlled_runge_kutta< runge_kutta_bs3< state_type >, custom_error_checker< double, range_algebra, default_operations >, custom_step_adjuster<double, double>> RK23;

/* The rhs of x' = f(x) */
void spiral_problem(const state_type& x, state_type& dxdt, const double t)
{
    dxdt[0] = std::cos(t) - x[1];
    dxdt[1] = std::sin(t) + x[0];
}

void lotka_volterra_problem(const state_type& x, state_type& dxdt, const double t)
{
    dxdt[0] = x[0] * (1 - x[1]);
    dxdt[1] = -x[1] * (1 - x[0]);
}

void brusselator_problem(const state_type& x, state_type& dxdt, const double t)
{
    dxdt[0] = 1 + x[0] * x[0] * x[1] - 4 * x[0];
    dxdt[1] = 3 * x[0] - x[0] * x[0] * x[1];
}

struct push_back_state_and_time
{
    std::vector< state_type >& m_states;
    std::vector< double >& m_times;

    push_back_state_and_time(std::vector< state_type >& states, std::vector< double >& times)
        : m_states(states), m_times(times) { }

    void operator()(const state_type& x, double t)
    {
        m_states.push_back(x);
        m_times.push_back(t);
    }
};


int main(int argc, const char* argv[]) {
    boost::program_options::options_description desc;
    desc.add_options()
        ("help,h", "Show this help screen")
        ("model_file_name", boost::program_options::value<std::string>()->default_value(""), "NN controller file name, leave empty if not used")
        ("method", boost::program_options::value<std::string>()->default_value("DP5"), "ode method in use, support DP5 or BS3")
        ("problem", boost::program_options::value<std::string>()->default_value("Spiral"), "problem to solve, support spiral or lotka_volterra")
        ("is_fixed", "using fixed method")
        ("atol", boost::program_options::value<double>()->default_value(1.0e-6), "absolute tolerance or stepsize when is_fixed is specified");

    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
    boost::program_options::notify(vm);
    if (vm.count("help")) {
        std::cout << desc << '\n';
        return 0;
    }
    std::string model_file_name = vm["model_file_name"].as<std::string>();
    double abs_err = vm["atol"].as<double>();
    bool is_fixed = vm.count("is_fixed") > 0;
    std::string method_name = vm["method"].as<std::string>();
    std::string problem_name = vm["problem"].as<std::string>();
    state_type y(2);

    double rel_err = 0.0, a_x = 1.0, a_dxdt = 0.0, max_dt = 100.0;
    double t_start = 0.0, t_end, y0_0, y1_0;
    std::vector<state_type> x_vec;
    std::vector<double> times;
    RK45 rk45_solver(
        custom_error_checker< double, range_algebra, default_operations >(abs_err, rel_err, a_x, a_dxdt),
        custom_step_adjuster<double, double>(max_dt),
        RK45::stepper_type(),
        model_file_name, is_fixed);
    RK23 rk23_solver(
        custom_error_checker< double, range_algebra, default_operations >(abs_err, rel_err, a_x, a_dxdt),
        custom_step_adjuster<double, double>(max_dt),
        RK23::stepper_type(),
        model_file_name, is_fixed);
    double initial_step;
    size_t repeat_time = 1000;
    long int_ns = 0;
    void (*problem)(const state_type&, state_type&, const double);
    if (problem_name == "Spiral") {
        problem = &spiral_problem;
        t_end = 2 * M_PI;
        y0_0 = 0.0; // initial value
        y1_0 = 0.0;
    }
    else if (problem_name == "LotkaVolterra") {
        problem = &lotka_volterra_problem;
        t_end = 15.0;
        y0_0 = 2.0; // initial value
        y1_0 = 1.0;
    }
    else {
        problem = &brusselator_problem;
        t_end = 20.0;
        y0_0 = 1.5; // initial value
        y1_0 = 3.0;
    }
    y[0] = y0_0;
    y[1] = y1_0;
    if (method_name == "DP5") {
        if (is_fixed) {
            initial_step = abs_err;
        }
        else {
            initial_step = select_initial_step(problem, t_start, y, rk45_solver.stepper().error_order(), rel_err, abs_err);
        }

        size_t steps = integrate_adaptive(rk45_solver, problem,
            y, t_start, t_end, initial_step, push_back_state_and_time(x_vec, times));

        for (int i = 0; i < repeat_time; i++) {
            y[0] = y0_0; // reset initial value
            y[1] = y1_0;
            auto t1 = std::chrono::high_resolution_clock::now();
            steps = integrate_adaptive(rk45_solver, problem,
                y, t_start, t_end, initial_step);
            auto t2 = std::chrono::high_resolution_clock::now();
            int_ns += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        }
    }
    else { // BS3 currently

        if (is_fixed) {
            initial_step = abs_err;
        }
        else {
            initial_step = select_initial_step(problem, t_start, y, rk23_solver.stepper().error_order(), rel_err, abs_err);
        }
        size_t steps = integrate_adaptive(rk23_solver, problem,
            y, t_start, t_end, initial_step, push_back_state_and_time(x_vec, times));

        for (int i = 0; i < repeat_time; i++) {
            y[0] = y0_0; // reset initial value
            y[1] = y1_0;
            auto t1 = std::chrono::high_resolution_clock::now();
            steps = integrate_adaptive(rk23_solver, problem,
                y, t_start, t_end, initial_step);
            auto t2 = std::chrono::high_resolution_clock::now();
            int_ns += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        }
    }
    double average_time = int_ns * 1.0 / repeat_time;
    /* output */
    /*for (size_t i = 0; i <= steps; i++)
    {
        std::cout << std::setprecision(7) << times[i] << '\t' << x_vec[i][0] << '\t' << x_vec[i][1] << '\n';
    }
    */
    std::cout << average_time << std::endl;
}
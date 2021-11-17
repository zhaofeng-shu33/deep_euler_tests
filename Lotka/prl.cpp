#include <boost/numeric/odeint.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <iostream>
#include <vector>
#define _USE_MATH_DEFINES
#include <math.h>
#include "step_adjuster.hpp"

using namespace boost::numeric::odeint;
typedef std::vector< double > state_type;
typedef runge_kutta_dopri5< state_type > error_stepper_type;
typedef custom_controlled_runge_kutta< error_stepper_type, custom_error_checker< double, range_algebra, default_operations >, custom_step_adjuster<double, double>> controlled_stepper_type;

/* The rhs of x' = f(x) */
void spiral_problem(const state_type& x, state_type& dxdt, const double t)
{
    dxdt[0] = std::cos(t) - x[1];
    dxdt[1] = std::sin(t) + x[0];
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


int main() {
    const double y1_0 = 0.0;
    const double y0_0 = 0.0;
    state_type y(2);
    y[0] = y0_0; // initial value
    y[1] = y1_0;
    double abs_err = 1.0e-6, rel_err = 0.0, a_x = 1.0, a_dxdt = 0.0, max_dt = 100.0;
    double t_start = 0.0, t_end = 2 * M_PI;
    std::vector<state_type> x_vec;
    std::vector<double> times;

    controlled_stepper_type controlled_stepper(
        custom_error_checker< double, range_algebra, default_operations >(abs_err, rel_err, a_x, a_dxdt),
        custom_step_adjuster<double, double>(max_dt),
        controlled_stepper_type::stepper_type(),
        true);
    double initial_step = select_initial_step(spiral_problem, t_start, y, controlled_stepper.stepper().error_order(), rel_err, abs_err);
    size_t steps = integrate_adaptive(controlled_stepper, spiral_problem,
        y, t_start, t_end, initial_step, push_back_state_and_time(x_vec, times));
    /* output */
    for (size_t i = 0; i <= steps; i++)
    {
        std::cout << std::setprecision(7) << times[i] << '\t' << x_vec[i][0] << '\t' << x_vec[i][1] << '\n';
    }
}
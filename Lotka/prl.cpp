#include <boost/numeric/odeint.hpp>
#include <iostream>
#include <vector>

#include "step_adjuster.hpp"

using namespace boost::numeric::odeint;
typedef std::vector< double > state_type;
typedef runge_kutta_dopri5< state_type > error_stepper_type;
typedef controlled_runge_kutta< error_stepper_type, default_error_checker< double, range_algebra, default_operations >, custom_step_adjuster<double, double>> controlled_stepper_type;
const double gam = 0.15;

/* The rhs of x' = f(x) */
void harmonic_oscillator(const state_type& x, state_type& dxdt, const double /* t */)
{
    dxdt[0] = x[1];
    dxdt[1] = -x[0] - gam * x[1];
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
    state_type x(2);
    x[0] = 1.0; // start at x=1.0, p=0.0
    x[1] = 0.0;
    double abs_err = 1.0e-10, rel_err = 1.0e-6, a_x = 1.0, a_dxdt = 0.0, max_dt = 0.5;
    std::vector<state_type> x_vec;
    std::vector<double> times;

    controlled_stepper_type controlled_stepper(
        default_error_checker< double, range_algebra, default_operations >(abs_err, rel_err, a_x, a_dxdt),
        custom_step_adjuster<double, double>(max_dt));
    size_t steps = integrate_adaptive(controlled_stepper, harmonic_oscillator,
        x, 0.0, 10.0, 0.1, push_back_state_and_time(x_vec, times));
    std::cout << steps << std::endl;
}
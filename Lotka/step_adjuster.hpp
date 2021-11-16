#pragma once
#include <cmath>



#include <boost/numeric/odeint/stepper/controlled_runge_kutta.hpp>

namespace boost {
namespace numeric {
namespace odeint {



template< typename Value, typename Time >
class custom_step_adjuster
{
public:
    typedef Time time_type;
    typedef Value value_type;

    custom_step_adjuster(const time_type max_dt=static_cast<time_type>(0))
            : m_max_dt(max_dt)
    {}


    time_type decrease_step(time_type dt, const value_type error, const int error_order) const
    {
        // returns the decreased time step
        BOOST_USING_STD_MIN();
        BOOST_USING_STD_MAX();
        using std::pow;

        dt *= max
        BOOST_PREVENT_MACRO_SUBSTITUTION(
                static_cast<value_type>( static_cast<value_type>(9) / static_cast<value_type>(10) *
                                         pow(error, static_cast<value_type>(-1) / (error_order))),
                static_cast<value_type>( static_cast<value_type>(1) / static_cast<value_type> (5)));
        if(m_max_dt != static_cast<time_type >(0))
            // limit to maximal stepsize even when decreasing
            dt = detail::min_abs(dt, m_max_dt);
        return dt;
    }

    time_type increase_step(time_type dt, value_type error, const int stepper_order) const
    {
        // returns the increased time step
        BOOST_USING_STD_MIN();
        BOOST_USING_STD_MAX();
        using std::pow;

        dt *= max
            BOOST_PREVENT_MACRO_SUBSTITUTION(
                static_cast<value_type>(static_cast<value_type>(9) / static_cast<value_type>(10) *
                    pow(error, static_cast<value_type>(-1) / (stepper_order))),
                static_cast<value_type>(static_cast<value_type>(1) / static_cast<value_type> (5)));
        if (m_max_dt != static_cast<time_type>(0))
            // limit to maximal stepsize even when decreasing
            dt = detail::min_abs(dt, m_max_dt);
        return dt;
    }

    bool check_step_size_limit(const time_type dt)
    {
        if(m_max_dt != static_cast<time_type >(0))
            return detail::less_eq_with_sign(dt, m_max_dt, dt);
        return true;
    }

    time_type get_max_dt() { return m_max_dt; }

protected:
    time_type m_max_dt;
};
} // odeint
} // numeric
} // boost
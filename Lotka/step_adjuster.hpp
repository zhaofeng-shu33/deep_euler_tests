#pragma once
#include <cmath>



#include <boost/numeric/odeint/stepper/controlled_runge_kutta.hpp>

namespace boost {
namespace numeric {
namespace odeint {
    double scale_norm(const std::vector<double>& y, const std::vector<double>& scale) {
        size_t n = boost::size(y);
        double _norm = 0.0;
        for (int i = 0; i < n; i++) {
            _norm += pow(y[i] / scale[i], 2.0);
        }
        _norm = sqrt(_norm / n);
        return _norm;
    }
    /*
    template<class value_type, class State>
    value_type scale_norm(State& y, State& scale) {
        value_type _norm = 0.0;
        detail::for_each2(boost::begin(y), boost::end(y),
            boost::begin(scale), [](auto& y_i, auto& scale_i) { _norm += y_i / scale_i });
    }
    */
    template< class Fac1 = double >
    struct custom_scale_sum
    {
        const Fac1 m_alpha1, m_alpha2;

        custom_scale_sum(Fac1 alpha1, Fac1 alpha2) : m_alpha1(alpha1), m_alpha2(alpha2) { }

        template< class T1, class T2 >
        void operator()(T1& t1, const T2& t2) const
        {
            t1 = m_alpha1 * std::abs(t2) + m_alpha2;
        }

        typedef void result_type;
    };

    template<class value_type, class System, class State>
    value_type select_initial_step(System fun, value_type t0, State& y0, size_t order, value_type rtol, value_type atol) {
        // create local State to hold values
        state_wrapper< State > f0, f1, scale, tmp_err;
        range_algebra algebra;
        // resize it
        adjust_size_by_resizeability(scale, y0, typename is_resizeable<State>::type());
        adjust_size_by_resizeability(tmp_err, y0, typename is_resizeable<State>::type());
        adjust_size_by_resizeability(f0, y0, typename is_resizeable<State>::type());
        adjust_size_by_resizeability(f1, y0, typename is_resizeable<State>::type());
        // compute f0
        fun(y0, f0.m_v, t0);
        // using for_each to compute the scale vector
        algebra.for_each2(scale.m_v, y0,
            custom_scale_sum< value_type >(rtol, atol));
        // compute d0, d1
        value_type d0 = scale_norm(y0, scale.m_v);
        value_type d1 = scale_norm(f0.m_v, scale.m_v);
        // h0
        value_type h0 = 0.01 * d0 / d1;
        if (d0 < 1e-5 || d1 < 1e-5) {
            h0 = 1e-6;
        }
        algebra.for_each3(tmp_err.m_v, y0, f0.m_v,
            default_operations::scale_sum2< value_type >(1.0, h0));
        // tmp_err.m_v becomes y1 now
        State& y1 = tmp_err.m_v;
        // compute f1
        fun(y1, f1.m_v, t0 + h0);
        algebra.for_each3(tmp_err.m_v, f1.m_v, f0.m_v,
            default_operations::scale_sum2< value_type >(1.0, -1.0));
        value_type d2 = scale_norm(tmp_err.m_v, scale.m_v) / h0;
        value_type h1;
        if (d1 <= 1e-15 || d2 <= 1e-15) {
            h1 = std::max(1e-6, h0 * 1e-3);
        }
        else {
            h1 = pow(0.01 / std::max(d1, d2), 1.0 / (order + 1));
        }
        return std::min(100 * h0, h1);
    }


template< typename Value, typename Time >
class custom_step_adjuster
{
public:
    typedef Time time_type;
    typedef Value value_type;

    custom_step_adjuster(const time_type max_dt=static_cast<time_type>(0))
            : m_max_dt(max_dt)
    {}


    time_type decrease_step(time_type dt, const value_type error, const int stepper_order) const
    {
        // returns the decreased time step
        BOOST_USING_STD_MIN();
        BOOST_USING_STD_MAX();
        using std::pow;

        dt *= max
        BOOST_PREVENT_MACRO_SUBSTITUTION(
                static_cast<value_type>( static_cast<value_type>(9) / static_cast<value_type>(10) *
                                         pow(error, static_cast<value_type>(-1) / (stepper_order))),
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

// only support FSAL
template<
    class ErrorStepper,
    class ErrorChecker = default_error_checker< typename ErrorStepper::value_type,
    typename ErrorStepper::algebra_type,
    typename ErrorStepper::operations_type >,
    class StepAdjuster = default_step_adjuster< typename ErrorStepper::value_type,
    typename ErrorStepper::time_type >,
    class Resizer = typename ErrorStepper::resizer_type,
    class ErrorStepperCategory = typename ErrorStepper::stepper_category
>
class custom_controlled_runge_kutta;

template<
    class ErrorStepper,
    class ErrorChecker,
    class StepAdjuster,
    class Resizer
>
class custom_controlled_runge_kutta< ErrorStepper, ErrorChecker, StepAdjuster, Resizer, explicit_error_stepper_fsal_tag >
{

public:

    typedef ErrorStepper stepper_type;
    typedef typename stepper_type::state_type state_type;
    typedef typename stepper_type::value_type value_type;
    typedef typename stepper_type::deriv_type deriv_type;
    typedef typename stepper_type::time_type time_type;
    typedef typename stepper_type::algebra_type algebra_type;
    typedef typename stepper_type::operations_type operations_type;
    typedef Resizer resizer_type;
    typedef ErrorChecker error_checker_type;
    typedef StepAdjuster step_adjuster_type;
    typedef explicit_controlled_stepper_fsal_tag stepper_category;

#ifndef DOXYGEN_SKIP
    typedef typename stepper_type::wrapped_state_type wrapped_state_type;
    typedef typename stepper_type::wrapped_deriv_type wrapped_deriv_type;

    typedef custom_controlled_runge_kutta< ErrorStepper, ErrorChecker, StepAdjuster, Resizer, explicit_error_stepper_tag > controlled_stepper_type;
#endif // DOXYGEN_SKIP

    /**
     * \brief Constructs the controlled Runge-Kutta stepper.
     * \param error_checker An instance of the error checker.
     * \param stepper An instance of the underlying stepper.
     */
    custom_controlled_runge_kutta(
        const error_checker_type& error_checker = error_checker_type(),
        const step_adjuster_type& step_adjuster = step_adjuster_type(),
        const stepper_type& stepper = stepper_type()
    )
        : m_stepper(stepper), m_error_checker(error_checker), m_step_adjuster(step_adjuster),
        m_first_call(true)
    { }

    /*
     * Version 1 : try_step( sys , x , t , dt )
     *
     * The two overloads are needed in order to solve the forwarding problem
     */
     /**
      * \brief Tries to perform one step.
      *
      * This method tries to do one step with step size dt. If the error estimate
      * is to large, the step is rejected and the method returns fail and the
      * step size dt is reduced. If the error estimate is acceptably small, the
      * step is performed, success is returned and dt might be increased to make
      * the steps as large as possible. This method also updates t if a step is
      * performed.
      *
      * \param system The system function to solve, hence the r.h.s. of the ODE. It must fulfill the
      *               Simple System concept.
      * \param x The state of the ODE which should be solved. Overwritten if
      * the step is successful.
      * \param t The value of the time. Updated if the step is successful.
      * \param dt The step size. Updated.
      * \return success if the step was accepted, fail otherwise.
      */
    template< class System, class StateInOut >
    controlled_step_result try_step(System system, StateInOut& x, time_type& t, time_type& dt)
    {
        return try_step_v1(system, x, t, dt);
    }


    /**
     * \brief Tries to perform one step. Solves the forwarding problem and
     * allows for using boost range as state_type.
     *
     * This method tries to do one step with step size dt. If the error estimate
     * is to large, the step is rejected and the method returns fail and the
     * step size dt is reduced. If the error estimate is acceptably small, the
     * step is performed, success is returned and dt might be increased to make
     * the steps as large as possible. This method also updates t if a step is
     * performed.
     *
     * \param system The system function to solve, hence the r.h.s. of the ODE. It must fulfill the
     *               Simple System concept.
     * \param x The state of the ODE which should be solved. Overwritten if
     * the step is successful. Can be a boost range.
     * \param t The value of the time. Updated if the step is successful.
     * \param dt The step size. Updated.
     * \return success if the step was accepted, fail otherwise.
     */
    template< class System, class StateInOut >
    controlled_step_result try_step(System system, const StateInOut& x, time_type& t, time_type& dt)
    {
        return try_step_v1(system, x, t, dt);
    }



    /*
     * Version 2 : try_step( sys , in , t , out , dt );
     *
     * This version does not solve the forwarding problem, boost::range can not be used.
     *
     * The disabler is needed to solve ambiguous overloads
     */
     /**
      * \brief Tries to perform one step.
      *
      * \note This method is disabled if state_type=time_type to avoid ambiguity.
      *
      * This method tries to do one step with step size dt. If the error estimate
      * is to large, the step is rejected and the method returns fail and the
      * step size dt is reduced. If the error estimate is acceptably small, the
      * step is performed, success is returned and dt might be increased to make
      * the steps as large as possible. This method also updates t if a step is
      * performed.
      *
      * \param system The system function to solve, hence the r.h.s. of the ODE. It must fulfill the
      *               Simple System concept.
      * \param in The state of the ODE which should be solved.
      * \param t The value of the time. Updated if the step is successful.
      * \param out Used to store the result of the step.
      * \param dt The step size. Updated.
      * \return success if the step was accepted, fail otherwise.
      */
    template< class System, class StateIn, class StateOut >
    typename boost::disable_if< boost::is_same< StateIn, time_type >, controlled_step_result >::type
        try_step(System system, const StateIn& in, time_type& t, StateOut& out, time_type& dt)
    {
        if (m_dxdt_resizer.adjust_size(in, detail::bind(&custom_controlled_runge_kutta::template resize_m_dxdt_impl< StateIn >, detail::ref(*this), detail::_1)) || m_first_call)
        {
            initialize(system, in, t);
        }
        return try_step(system, in, m_dxdt.m_v, t, out, dt);
    }


    /*
     * Version 3 : try_step( sys , x , dxdt , t , dt )
     *
     * This version does not solve the forwarding problem, boost::range can not be used.
     */
     /**
      * \brief Tries to perform one step.
      *
      * This method tries to do one step with step size dt. If the error estimate
      * is to large, the step is rejected and the method returns fail and the
      * step size dt is reduced. If the error estimate is acceptably small, the
      * step is performed, success is returned and dt might be increased to make
      * the steps as large as possible. This method also updates t if a step is
      * performed.
      *
      * \param system The system function to solve, hence the r.h.s. of the ODE. It must fulfill the
      *               Simple System concept.
      * \param x The state of the ODE which should be solved. Overwritten if
      * the step is successful.
      * \param dxdt The derivative of state.
      * \param t The value of the time. Updated if the step is successful.
      * \param dt The step size. Updated.
      * \return success if the step was accepted, fail otherwise.
      */
    template< class System, class StateInOut, class DerivInOut >
    controlled_step_result try_step(System system, StateInOut& x, DerivInOut& dxdt, time_type& t, time_type& dt)
    {
        m_xnew_resizer.adjust_size(x, detail::bind(&custom_controlled_runge_kutta::template resize_m_xnew_impl< StateInOut >, detail::ref(*this), detail::_1));
        m_dxdt_new_resizer.adjust_size(x, detail::bind(&custom_controlled_runge_kutta::template resize_m_dxdt_new_impl< StateInOut >, detail::ref(*this), detail::_1));
        controlled_step_result res = try_step(system, x, dxdt, t, m_xnew.m_v, m_dxdtnew.m_v, dt);
        if (res == success)
        {
            boost::numeric::odeint::copy(m_xnew.m_v, x);
            boost::numeric::odeint::copy(m_dxdtnew.m_v, dxdt);
        }
        return res;
    }


    /*
     * Version 4 : try_step( sys , in , dxdt_in , t , out , dxdt_out , dt )
     *
     * This version does not solve the forwarding problem, boost::range can not be used.
     */
     /**
      * \brief Tries to perform one step.
      *
      * This method tries to do one step with step size dt. If the error estimate
      * is to large, the step is rejected and the method returns fail and the
      * step size dt is reduced. If the error estimate is acceptably small, the
      * step is performed, success is returned and dt might be increased to make
      * the steps as large as possible. This method also updates t if a step is
      * performed.
      *
      * \param system The system function to solve, hence the r.h.s. of the ODE. It must fulfill the
      *               Simple System concept.
      * \param in The state of the ODE which should be solved.
      * \param dxdt The derivative of state.
      * \param t The value of the time. Updated if the step is successful.
      * \param out Used to store the result of the step.
      * \param dt The step size. Updated.
      * \return success if the step was accepted, fail otherwise.
      */
    template< class System, class StateIn, class DerivIn, class StateOut, class DerivOut >
    controlled_step_result try_step(System system, const StateIn& in, const DerivIn& dxdt_in, time_type& t,
        StateOut& out, DerivOut& dxdt_out, time_type& dt)
    {
        unwrapped_step_adjuster& step_adjuster = m_step_adjuster;
        if (!step_adjuster.check_step_size_limit(dt))
        {
            // given dt was above step size limit - adjust and return fail;
            dt = step_adjuster.get_max_dt();
            return fail;
        }

        m_xerr_resizer.adjust_size(in, detail::bind(&custom_controlled_runge_kutta::template resize_m_xerr_impl< StateIn >, detail::ref(*this), detail::_1));

        //fsal: m_stepper.get_dxdt( dxdt );
        //fsal: m_stepper.do_step( sys , x , dxdt , t , dt , m_x_err );
        m_stepper.do_step(system, in, dxdt_in, t, out, dxdt_out, dt, m_xerr.m_v);

        // this potentially overwrites m_x_err! (standard_error_checker does, at least)
        value_type max_rel_err = m_error_checker.error(m_stepper.algebra(), in, dxdt_in, m_xerr.m_v, dt);

        if (max_rel_err > 1.0)
        {
            // error too big, decrease step size and reject this step
            dt = step_adjuster.decrease_step(dt, max_rel_err, m_stepper.stepper_order());
            return fail;
        }
        // otherwise, increase step size and accept
        t += dt;
        dt = step_adjuster.increase_step(dt, max_rel_err, m_stepper.stepper_order());
        return success;
    }


    /**
     * \brief Resets the internal state of the underlying FSAL stepper.
     */
    void reset(void)
    {
        m_first_call = true;
    }

    /**
     * \brief Initializes the internal state storing an internal copy of the derivative.
     *
     * \param deriv The initial derivative of the ODE.
     */
    template< class DerivIn >
    void initialize(const DerivIn& deriv)
    {
        boost::numeric::odeint::copy(deriv, m_dxdt.m_v);
        m_first_call = false;
    }

    /**
     * \brief Initializes the internal state storing an internal copy of the derivative.
     *
     * \param system The system function to solve, hence the r.h.s. of the ODE. It must fulfill the
     *               Simple System concept.
     * \param x The initial state of the ODE which should be solved.
     * \param t The initial time.
     */
    template< class System, class StateIn >
    void initialize(System system, const StateIn& x, time_type t)
    {
        typename odeint::unwrap_reference< System >::type& sys = system;
        sys(x, m_dxdt.m_v, t);
        m_first_call = false;
    }

    /**
     * \brief Returns true if the stepper has been initialized, false otherwise.
     *
     * \return true, if the stepper has been initialized, false otherwise.
     */
    bool is_initialized(void) const
    {
        return !m_first_call;
    }


    /**
     * \brief Adjust the size of all temporaries in the stepper manually.
     * \param x A state from which the size of the temporaries to be resized is deduced.
     */
    template< class StateType >
    void adjust_size(const StateType& x)
    {
        resize_m_xerr_impl(x);
        resize_m_dxdt_impl(x);
        resize_m_dxdt_new_impl(x);
        resize_m_xnew_impl(x);
    }


    /**
     * \brief Returns the instance of the underlying stepper.
     * \returns The instance of the underlying stepper.
     */
    stepper_type& stepper(void)
    {
        return m_stepper;
    }

    /**
     * \brief Returns the instance of the underlying stepper.
     * \returns The instance of the underlying stepper.
     */
    const stepper_type& stepper(void) const
    {
        return m_stepper;
    }



private:


    template< class StateIn >
    bool resize_m_xerr_impl(const StateIn& x)
    {
        return adjust_size_by_resizeability(m_xerr, x, typename is_resizeable<state_type>::type());
    }

    template< class StateIn >
    bool resize_m_dxdt_impl(const StateIn& x)
    {
        return adjust_size_by_resizeability(m_dxdt, x, typename is_resizeable<deriv_type>::type());
    }

    template< class StateIn >
    bool resize_m_dxdt_new_impl(const StateIn& x)
    {
        return adjust_size_by_resizeability(m_dxdtnew, x, typename is_resizeable<deriv_type>::type());
    }

    template< class StateIn >
    bool resize_m_xnew_impl(const StateIn& x)
    {
        return adjust_size_by_resizeability(m_xnew, x, typename is_resizeable<state_type>::type());
    }


    template< class System, class StateInOut >
    controlled_step_result try_step_v1(System system, StateInOut& x, time_type& t, time_type& dt)
    {
        if (m_dxdt_resizer.adjust_size(x, detail::bind(&custom_controlled_runge_kutta::template resize_m_dxdt_impl< StateInOut >, detail::ref(*this), detail::_1)) || m_first_call)
        {
            initialize(system, x, t);
        }
        return try_step(system, x, m_dxdt.m_v, t, dt);
    }


    stepper_type m_stepper;
    error_checker_type m_error_checker;
    step_adjuster_type m_step_adjuster;
    typedef typename unwrap_reference< step_adjuster_type >::type unwrapped_step_adjuster;

    resizer_type m_dxdt_resizer;
    resizer_type m_xerr_resizer;
    resizer_type m_xnew_resizer;
    resizer_type m_dxdt_new_resizer;

    wrapped_deriv_type m_dxdt;
    wrapped_state_type m_xerr;
    wrapped_state_type m_xnew;
    wrapped_deriv_type m_dxdtnew;
    bool m_first_call;
};

} // odeint
} // numeric
} // boost
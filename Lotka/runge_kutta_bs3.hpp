/*
 [auto_generated]
 boost/numeric/odeint/stepper/runge_kutta_bs3.hpp

 [begin_description]
 Implementation of the Dormand-Prince 5(4) method. This stepper can also be used with the dense-output controlled stepper.
 [end_description]

 Copyright 2010-2013 Karsten Ahnert
 Copyright 2010-2013 Mario Mulansky
 Copyright 2012 Christoph Koke

 Distributed under the Boost Software License, Version 1.0.
 (See accompanying file LICENSE_1_0.txt or
 copy at http://www.boost.org/LICENSE_1_0.txt)
 */


#ifndef BOOST_NUMERIC_ODEINT_STEPPER_runge_kutta_bs3_HPP_INCLUDED
#define BOOST_NUMERIC_ODEINT_STEPPER_runge_kutta_bs3_HPP_INCLUDED


#include <boost/numeric/odeint/util/bind.hpp>

#include <boost/numeric/odeint/stepper/base/explicit_error_stepper_fsal_base.hpp>
#include <boost/numeric/odeint/algebra/range_algebra.hpp>
#include <boost/numeric/odeint/algebra/default_operations.hpp>
#include <boost/numeric/odeint/algebra/algebra_dispatcher.hpp>
#include <boost/numeric/odeint/algebra/operations_dispatcher.hpp>
#include <boost/numeric/odeint/stepper/stepper_categories.hpp>

#include <boost/numeric/odeint/util/state_wrapper.hpp>
#include <boost/numeric/odeint/util/is_resizeable.hpp>
#include <boost/numeric/odeint/util/resizer.hpp>
#include <boost/numeric/odeint/util/same_instance.hpp>

namespace boost {
namespace numeric {
namespace odeint {



template<
class State ,
class Value = double ,
class Deriv = State ,
class Time = Value ,
class Algebra = typename algebra_dispatcher< State >::algebra_type ,
class Operations = typename operations_dispatcher< State >::operations_type ,
class Resizer = initially_resizer
>
class runge_kutta_bs3
#ifndef DOXYGEN_SKIP
: public explicit_error_stepper_fsal_base<
  runge_kutta_bs3< State , Value , Deriv , Time , Algebra , Operations , Resizer > ,
  3, 3, 2 , State , Value , Deriv , Time , Algebra , Operations , Resizer >
#else
: public explicit_error_stepper_fsal_base
#endif
{

public :

    #ifndef DOXYGEN_SKIP
    typedef explicit_error_stepper_fsal_base<
    runge_kutta_bs3< State , Value , Deriv , Time , Algebra , Operations , Resizer > ,
    3, 3, 2 , State , Value , Deriv , Time , Algebra , Operations , Resizer > stepper_base_type;
    #else
    typedef explicit_error_stepper_fsal_base< runge_kutta_bs3< ... > , ... > stepper_base_type;
    #endif
    
    typedef typename stepper_base_type::state_type state_type;
    typedef typename stepper_base_type::value_type value_type;
    typedef typename stepper_base_type::deriv_type deriv_type;
    typedef typename stepper_base_type::time_type time_type;
    typedef typename stepper_base_type::algebra_type algebra_type;
    typedef typename stepper_base_type::operations_type operations_type;
    typedef typename stepper_base_type::resizer_type resizer_type;

    #ifndef DOXYGEN_SKIP
    typedef typename stepper_base_type::stepper_type stepper_type;
    typedef typename stepper_base_type::wrapped_state_type wrapped_state_type;
    typedef typename stepper_base_type::wrapped_deriv_type wrapped_deriv_type;
    #endif // DOXYGEN_SKIP


    runge_kutta_bs3( const algebra_type &algebra = algebra_type() ) : stepper_base_type( algebra )
    { }


    template< class System , class StateIn , class DerivIn , class StateOut , class DerivOut >
    void do_step_impl( System system , const StateIn &in , const DerivIn &dxdt_in , time_type t ,
            StateOut &out , DerivOut &dxdt_out , time_type dt )
    {
        const value_type a2 = static_cast<value_type> ( 1 ) / static_cast<value_type>( 2 );
        const value_type a3 = static_cast<value_type> ( 3 ) / static_cast<value_type> ( 4 );
        const value_type a4 = static_cast<value_type> ( 1 ) / static_cast<value_type> ( 1 );

        const value_type b21 = static_cast<value_type> ( 1 ) / static_cast<value_type> ( 2 );

        const value_type b32 = static_cast<value_type> ( 3 ) / static_cast<value_type>( 4 );

        const value_type b41 = static_cast<value_type> ( 2 ) / static_cast<value_type> ( 9 );
        const value_type b42 = static_cast<value_type> ( 1 ) / static_cast<value_type> ( 3 );
        const value_type b43 = static_cast<value_type> ( 4 ) / static_cast<value_type> ( 9 );

        const value_type c1 = static_cast<value_type> (2) / static_cast<value_type>(9);
        const value_type c2 = static_cast<value_type> (1) / static_cast<value_type>(3);
        const value_type c3 = static_cast<value_type> (4) / static_cast<value_type>(9);

        typename odeint::unwrap_reference< System >::type &sys = system;

        m_k_x_tmp_resizer.adjust_size( in , detail::bind( &stepper_type::template resize_k_x_tmp_impl<StateIn> , detail::ref( *this ) , detail::_1 ) );

        //m_x_tmp = x + dt*b21*dxdt
        stepper_base_type::m_algebra.for_each3( m_x_tmp.m_v , in , dxdt_in ,
                typename operations_type::template scale_sum2< value_type , time_type >( 1.0 , dt*b21 ) );

        sys( m_x_tmp.m_v , m_k2.m_v , t + dt*a2 );
        // m_x_tmp = x + dt*b32*m_k2
        stepper_base_type::m_algebra.for_each3( m_x_tmp.m_v , in, m_k2.m_v ,
                typename operations_type::template scale_sum2< value_type , time_type >( 1.0 ,  dt*b32 ));

        sys( m_x_tmp.m_v , m_k3.m_v , t + dt*a3 );
        // m_x_tmp = x + dt * (b41*dxdt + b42*m_k2 + b43*m_k3)
        stepper_base_type::m_algebra.for_each5( m_x_tmp.m_v , in , dxdt_in , m_k2.m_v , m_k3.m_v ,
                typename operations_type::template scale_sum4< value_type , time_type , time_type , time_type >( 1.0 , dt*b41 , dt*b42 , dt*b43 ));


        // the new derivative
        sys( out , dxdt_out , t + dt );
    }



    template< class System , class StateIn , class DerivIn , class StateOut , class DerivOut , class Err >
    void do_step_impl( System system , const StateIn &in , const DerivIn &dxdt_in , time_type t ,
            StateOut &out , DerivOut &dxdt_out , time_type dt , Err &xerr )
    {
        const value_type c1 = static_cast<value_type> ( 2 ) / static_cast<value_type>( 9 );
        const value_type c2 = static_cast<value_type> ( 1 ) / static_cast<value_type>( 3 );
        const value_type c3 = static_cast<value_type> ( 4 ) / static_cast<value_type>( 9 );

        const value_type dc1 = c1 - static_cast<value_type> ( 7 ) / static_cast<value_type>( 24 );
        const value_type dc2 = c2 - static_cast<value_type> ( 1 ) / static_cast<value_type>( 4 );
        const value_type dc3 = c3 - static_cast<value_type> ( 1 ) / static_cast<value_type>( 3 );
        const value_type dc4 = static_cast<value_type>( -1 ) / static_cast<value_type> ( 8 );

        /* ToDo: copy only if &dxdt_in == &dxdt_out ? */
        if( same_instance( dxdt_in , dxdt_out ) )
        {
            m_dxdt_tmp_resizer.adjust_size( in , detail::bind( &stepper_type::template resize_dxdt_tmp_impl<StateIn> , detail::ref( *this ) , detail::_1 ) );
            boost::numeric::odeint::copy( dxdt_in , m_dxdt_tmp.m_v );
            do_step_impl( system , in , dxdt_in , t , out , dxdt_out , dt );
            //error estimate
            stepper_base_type::m_algebra.for_each5( xerr , m_dxdt_tmp.m_v , m_k2.m_v , m_k3.m_v, dxdt_out ,
                                                    typename operations_type::template scale_sum4< time_type , time_type , time_type , time_type>( dt*dc1 , dt*dc2, dt*dc3 , dt*dc4) );

        }
        else
        {
            do_step_impl( system , in , dxdt_in , t , out , dxdt_out , dt );
            //error estimate
            stepper_base_type::m_algebra.for_each5( xerr , dxdt_in , m_k2.m_v , m_k3.m_v , dxdt_out ,
                                                    typename operations_type::template scale_sum4< time_type , time_type , time_type , time_type>(dt * dc1, dt * dc2, dt * dc3, dt * dc4) );

        }

    }


    template< class StateOut , class StateIn1 , class DerivIn1 , class StateIn2 , class DerivIn2 >
    void calc_state( time_type t , StateOut &x ,
                     const StateIn1 &x_old , const DerivIn1 &deriv_old , time_type t_old ,
                     const StateIn2 & /* x_new */ , const DerivIn2 &deriv_new , time_type t_new ) const
    {
    }


    template< class StateIn >
    void adjust_size( const StateIn &x )
    {
        resize_k_x_tmp_impl( x );
        resize_dxdt_tmp_impl( x );
        stepper_base_type::adjust_size( x );
    }
    

private:

    template< class StateIn >
    bool resize_k_x_tmp_impl( const StateIn &x )
    {
        bool resized = false;
        resized |= adjust_size_by_resizeability( m_x_tmp , x , typename is_resizeable<state_type>::type() );
        resized |= adjust_size_by_resizeability( m_k2 , x , typename is_resizeable<deriv_type>::type() );
        resized |= adjust_size_by_resizeability( m_k3 , x , typename is_resizeable<deriv_type>::type() );
        return resized;
    }

    template< class StateIn >
    bool resize_dxdt_tmp_impl( const StateIn &x )
    {
        return adjust_size_by_resizeability( m_dxdt_tmp , x , typename is_resizeable<deriv_type>::type() );
    }
        


    wrapped_state_type m_x_tmp;
    wrapped_deriv_type m_k2 , m_k3;
    wrapped_deriv_type m_dxdt_tmp;
    resizer_type m_k_x_tmp_resizer;
    resizer_type m_dxdt_tmp_resizer;
};



/************* DOXYGEN ************/
/**
 * \class runge_kutta_bs3
 * \brief The Bogacki¨CShampine method.
 *
 * The Bogacki¨CShampine method is a very popular method for solving ODEs, see
 * <a href=""></a>.
 * The method is explicit and fulfills the Error Stepper concept. Step size control
 * is provided but continuous output is available which make this method favourable for many applications. 
 * 
 * This class derives from explicit_error_stepper_fsal_base and inherits its interface via CRTP (current recurring
 * template pattern). The method possesses the FSAL (first-same-as-last) property. See
 * explicit_error_stepper_fsal_base for more details.
 *
 * \tparam State The state type.
 * \tparam Value The value type.
 * \tparam Deriv The type representing the time derivative of the state.
 * \tparam Time The time representing the independent variable - the time.
 * \tparam Algebra The algebra type.
 * \tparam Operations The operations type.
 * \tparam Resizer The resizer policy type.
 */


    /**
     * \fn runge_kutta_bs3::runge_kutta_bs3( const algebra_type &algebra )
     * \brief Constructs the runge_kutta_bs3 class. This constructor can be used as a default
     * constructor if the algebra has a default constructor.
     * \param algebra A copy of algebra is made and stored inside explicit_stepper_base.
     */

    /**
     * \fn runge_kutta_bs3::do_step_impl( System system , const StateIn &in , const DerivIn &dxdt_in , time_type t , StateOut &out , DerivOut &dxdt_out , time_type dt )
     * \brief This method performs one step. The derivative `dxdt_in` of `in` at the time `t` is passed to the
     * method. The result is updated out-of-place, hence the input is in `in` and the output in `out`. Furthermore,
     * the derivative is update out-of-place, hence the input is assumed to be in `dxdt_in` and the output in
     * `dxdt_out`. 
     * Access to this step functionality is provided by explicit_error_stepper_fsal_base and 
     * `do_step_impl` should not be called directly.
     *
     * \param system The system function to solve, hence the r.h.s. of the ODE. It must fulfill the
     *               Simple System concept.
     * \param in The state of the ODE which should be solved. in is not modified in this method
     * \param dxdt_in The derivative of x at t. dxdt_in is not modified by this method
     * \param t The value of the time, at which the step should be performed.
     * \param out The result of the step is written in out.
     * \param dxdt_out The result of the new derivative at time t+dt.
     * \param dt The step size.
     */

    /**
     * \fn runge_kutta_bs3::do_step_impl( System system , const StateIn &in , const DerivIn &dxdt_in , time_type t , StateOut &out , DerivOut &dxdt_out , time_type dt , Err &xerr )
     * \brief This method performs one step. The derivative `dxdt_in` of `in` at the time `t` is passed to the
     * method. The result is updated out-of-place, hence the input is in `in` and the output in `out`. Furthermore,
     * the derivative is update out-of-place, hence the input is assumed to be in `dxdt_in` and the output in
     * `dxdt_out`. 
     * Access to this step functionality is provided by explicit_error_stepper_fsal_base and 
     * `do_step_impl` should not be called directly.
     * An estimation of the error is calculated.
     *
     * \param system The system function to solve, hence the r.h.s. of the ODE. It must fulfill the
     *               Simple System concept.
     * \param in The state of the ODE which should be solved. in is not modified in this method
     * \param dxdt_in The derivative of x at t. dxdt_in is not modified by this method
     * \param t The value of the time, at which the step should be performed.
     * \param out The result of the step is written in out.
     * \param dxdt_out The result of the new derivative at time t+dt.
     * \param dt The step size.
     * \param xerr An estimation of the error.
     */

    /**
     * \fn runge_kutta_bs3::calc_state( time_type t , StateOut &x , const StateIn1 &x_old , const DerivIn1 &deriv_old , time_type t_old , const StateIn2 &  , const DerivIn2 &deriv_new , time_type t_new ) const
     * \brief This method is used for continuous output and it calculates the state `x` at a time `t` from the 
     * knowledge of two states `old_state` and `current_state` at time points `t_old` and `t_new`. It also uses
     * internal variables to calculate the result. Hence this method must be called after two successful `do_step`
     * calls.
     */

    /**
     * \fn runge_kutta_bs3::adjust_size( const StateIn &x )
     * \brief Adjust the size of all temporaries in the stepper manually.
     * \param x A state from which the size of the temporaries to be resized is deduced.
     */

} // odeint
} // numeric
} // boost


#endif // BOOST_NUMERIC_ODEINT_STEPPER_runge_kutta_bs3_HPP_INCLUDED

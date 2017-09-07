#ifndef TICTOC_H
#define TICTOC_H

#include <iostream>				// std::cout, std::endl
#include <iomanip>				// std::setw, std::setfill, std::setprecision, std::left, std::right
#include <type_traits>			// std::is_void_v, std::invoke_result_t, std::enable_if_t, std::is_invocable_v, std::is_default_constructible_v, std::is_same_v
#include <utility>				// std::forward
#include <chrono>				// std::chrono::steady_clock, std::chrono::duration
#include <vector>				// std::vector

#define TICTOC_DEFAULT_UNITS std::milli

template <bool print = true>
inline void tic();

template <bool print = true, class units = TICTOC_DEFAULT_UNITS>
inline double toc(bool cumulative = false);

namespace tictoc_ns {
	using namespace std;
	using namespace chrono;
	vector<steady_clock::time_point> tics;
	char save_fill = ' ';
	steady_clock::time_point initial_reference;
	struct init {
		init() {
			tics.reserve(10);
			tic<0>();
			toc<0>();
		}
	} run;

	template <class unit>
	constexpr std::string unit_name() {
		if (is_same_v<unit, ratio<1,1>>)
			return "sec";
		else if (is_same_v<unit, milli>)
			return "ms";
		else if (is_same_v<unit, micro>)
			return "μs";
		else if (is_same_v<unit, nano>)
			return "ns";
	}
}


template <bool verbose = true>
inline void tic()
{
	using namespace std;
	using namespace chrono;
	using namespace tictoc_ns;
	if constexpr (verbose) {
		save_fill = cout.fill();
		cout
		<< "\n"
		<< right
		<< setfill('-')
		<< setw(30 + 12*tics.size())
		<< "tic" << endl;
	}
	tics.push_back(steady_clock::now());
	if (tics.size() == 1)
		initial_reference = tics.front();
}

template
<bool verbose = true,
class units = TICTOC_DEFAULT_UNITS>
inline double toc(bool cumulative)
{
	using namespace std;
	using namespace chrono;
	using namespace tictoc_ns;
	steady_clock::time_point end = steady_clock::now();
	
	steady_clock::time_point start;
	if (tics.empty() || cumulative) {
		start = initial_reference;
	}
	else
	{
		start = tics.back();
		tics.pop_back();
	}

	duration<double, units> elapsed = end - start;

	if constexpr (verbose)
	{
		std::cout
		<< std::right
		<< setfill('-')
		<< std::setw(30 + 12*tics.size())
		<< "toc" << "\n"
		<< setprecision(9)
		<< std::setw(27 + 12*tics.size()) << elapsed.count() << " " << unit_name<units>() << "\n" << std::endl;
		std::cout << std::setfill(save_fill);
	}

	return elapsed.count();
}


template
<auto* func,
class units = TICTOC_DEFAULT_UNITS,
class ...ArgTypes,
class R = std::enable_if_t<	std::is_invocable_v<decltype(*func), ArgTypes...>,
							std::invoke_result_t<decltype(*func), ArgTypes...>>>
R tictoc(ArgTypes&&... args)			// by pointer (e.g. for function or functor pointers)
{
	using namespace std;
	tic();
	
	if constexpr(is_void_v<R>)
	{
		(*func) (forward<ArgTypes>(args)...);
		toc<true, units>();
	}
	else
	{
		auto&& result = (*func) (forward<ArgTypes>(args)...);
		toc<true, units>();
		return result;
	}
}


template
<class Functor,
class units = TICTOC_DEFAULT_UNITS,
class ...ArgTypes,
class R = std::enable_if_t<	std::is_default_constructible_v<Functor> &&
							std::is_invocable_v<Functor, ArgTypes...>,
							std::invoke_result_t<Functor, ArgTypes...>>>
R tictoc(ArgTypes&&... args)			// by type (e.g. for functors)
{
	using namespace std;
	tic();
	
	if constexpr(is_void_v<R>)
	{
		Functor() (forward<ArgTypes>(args)...);
		toc<true, units>();
	}
	else
	{
		auto&& result = Functor() (forward<ArgTypes>(args)...);
		toc<true, units>();
		return result;
	}
}


template
<class L,
class units = TICTOC_DEFAULT_UNITS,
class ...ArgTypes,
class R = std::enable_if_t<!std::is_default_constructible_v<L> &&
							std::is_invocable_v<L, ArgTypes...>,
							std::invoke_result_t<L, ArgTypes...>>>
R tictoc(L&& lambda, ArgTypes&&... args)			// by value (e.g. for lambdas)
{
	using namespace std;
	tic();
	
	if constexpr(is_void_v<R>)
	{
		lambda(forward<ArgTypes>(args)...);
		toc<true, units>();
	}
	else
	{
		auto&& result = lambda(forward<ArgTypes>(args)...);
		toc<true, units>();
		return result;
	}
}

#undef TICTOC_DEFAULT_UNITS
#endif /* TICTOC_H */


/* Test tictoc() */
#if __INCLUDE_LEVEL__ == 0 && defined __INCLUDE_LEVEL__
	#include <iostream>			// std::cout, std::endl
	#include <thread>			// std::this_thread::sleep_for
	#include <chrono>			// literal seconds operator
	#include <string>			// std::string
	#include <iomanip>			// std::setprecision
	#include <type_traits>		// std::invoke_result_t
	#include "type_stuff.h"		// type_name()

using namespace std;
using namespace string_literals;
using namespace this_thread;

// From here: http://www.cplusplus.com/reference/thread/this_thread/sleep_for/
string sleeping(int n) 
{
	cout << "Countdown:\n";
	for (int i=n; i>0; --i) {
		cout << i << endl;
		sleep_for (1s);
	}
	cout << "Lift off!\n";

	return "Yippee!";
}


void hello()
{ cout << "hello world" << endl; }


struct functor
{
	string operator() (double d) {
		cout << "lookey here: " << d << endl;
		return "Got as input: " + std::to_string(d);
	}
} test_functor;


int main(int argc, char* argv[])
{
	cout << argv[0] << " Starting...\n" << endl;

	cout << "Testing function passed by pointer..." << tictoc<sleeping>(3) << endl << endl;
	cout << "\n\n" << endl;
	
	cout << "Testing function passed by pointer (no args or return value)..." << endl;
	tictoc<hello>();
	cout << "\n\n" << endl;


	auto hi = [a = argv[0]] (int n)
	{
		for (int i=0; i<n; i++)
			cout << "hi there!" << endl;

		cout << "from within:\n\t" << a << endl;
		return "i'm the lambda return value!"s;
	};
	cout << "Testing lambda passed by value..." << endl;
	cout << tictoc(hi, 3) << endl;
	cout << "return value type: " << type_name< invoke_result_t<decltype(hi), int> >() << endl;
	cout << "\n\n" << endl;

	int test_var = 0;
	auto hi2 = [&,a = argv[0]] (int n) -> int&
	{
		for (int i=0; i<n; i++)
			cout << "hi there!" << endl;

		cout << "from within:\n\t" << a << endl;
		return test_var;
	};
	decltype(auto) tv = tictoc(hi2,2);
	tv = -1;
	cout << "Returning by reference...\ttest_var=" << test_var << " vs. tv=" << tv << endl;

	cout << "Testing functor passed by pointer..." << endl;
	cout << setprecision(2) << tictoc<&test_functor>(5.0) << endl;
	cout << "\n\n" << endl;


	cout << "Testing functor passed by class type (and mixing/nesting tictoc call types)..." << endl;
	tic();
	cout << setprecision(2) << tictoc<functor>(5.0) << endl;
	toc();
	cout << "\n" << endl;


	cout << "Testing cumulative timing..." << endl;
	tic();
	cout << "tic #1" << endl;
	toc(true);
	cout << "tic #2" << endl;
	toc(true);
	cout << "tic #3" << endl;
	toc(true);
	cout << "tic #4" << endl;
	toc(true);
	cout << "tic #5" << endl;
	toc();

	return 0;
}
#endif
/* Test tictoc() */
#ifndef TICTOC_H
#define TICTOC_H

#include <iostream>				// std::cout, std::endl
#include <iomanip>				// std::setw, std::setfill, std::setprecision, std::left, std::right
#include <type_traits>			// std::is_void_v, std::invoke_result_t, std::enable_if_t, std::is_invocable_v, std::is_default_constructible_v, std::is_same_v
#include <utility>				// std::forward
#include <chrono>				// std::chrono::steady_clock, std::chrono::duration
#include <vector>				// std::vector

#define DEFAULT_UNITS std::milli

template <bool PRINT = true>
inline void tic();

template <bool PRINT = true, class UNITS = DEFAULT_UNITS>
inline double toc();

namespace tictoc_detail {
	using namespace std;
	using namespace chrono;
	using Point = steady_clock::time_point;

	vector<Point> Tics;
	Point start;
	struct init {
		init() {
			Tics.reserve(10);
			tic<0>();
			toc<0>();
		}
	} run;
	char save_fill = ' ';

	template <class UNITS>
	constexpr std::string unit_name() {
		if (is_same_v<UNITS, ratio<1,1>>) 	return "sec";
		else if (is_same_v<UNITS, milli>) 	return "ms";
		else if (is_same_v<UNITS, micro>) 	return "μs";
		else if (is_same_v<UNITS, nano>) 	return "ns";
	}
}


template <bool PRINT = true>
inline void tic()
{
	using namespace std;
	using namespace chrono;
	using namespace tictoc_detail;

	if constexpr (PRINT)
		save_fill = cout.fill();

	Tics.push_back(steady_clock::now());
	start = Tics.back();
}

template
<bool PRINT = true,
class UNITS = DEFAULT_UNITS>
inline double toc()
{
	using namespace std;
	using namespace chrono;
	using namespace tictoc_detail;

	if (!Tics.empty())
		Tics.pop_back();

	Point end = steady_clock::now();
	duration<double, UNITS> elapsed = end - start;

	if constexpr (PRINT)
	{
		std::cout
		<< std::right
		<< setfill('.')
		<< setw(30 + 22*Tics.size())
		<< "elapsed:"
		<< setfill(' ')
		<< setw(10)
		<< setprecision(9)
		<< elapsed.count()
		<< setw(3)
		<< unit_name<UNITS>()
		<< "\n" << std::endl;
		std::cout << std::setfill(save_fill);
	}

	return elapsed.count();
}


template
<
size_t TIMES = 1,
class UNITS = DEFAULT_UNITS,
class Callable,
class ...ArgTypes,
class R = std::invoke_result_t<Callable, ArgTypes...>,
std::enable_if_t<TIMES == 1, int> =0
>
R tictoc(Callable&& f, ArgTypes&&... args)
{
	using namespace std;

	tic();
	if constexpr (is_void_v<R>)
	{
		f(forward<ArgTypes>(args)...);
		toc<true, UNITS>();
	}
	else
	{
		auto&& result = f(forward<ArgTypes>(args)...);
		toc<true, UNITS>();
		return (R) result;
	}
}


template
<
size_t TIMES,
class UNITS = DEFAULT_UNITS,
class Callable,
class ...ArgTypes,
class R = std::invoke_result_t<Callable, ArgTypes...>,
std::enable_if_t<(TIMES > 1), int> =0
>
R tictoc(Callable&& f, ArgTypes&&... args)
{
	using namespace std;

	tic();
	if constexpr (is_void_v<R>)
	{
		f(forward<ArgTypes>(args)...);
		toc<true, UNITS>();
	}
	else
	{
		auto&& result = f(forward<ArgTypes>(args)...);
		toc<true, UNITS>();
		return result;
	}
}

#undef DEFAULT_UNITS
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

	cout << "Testing function passed by pointer..." << tictoc(sleeping, 3) << endl << endl;
	cout << "\n\n" << endl;

	cout << "Testing function passed by pointer (no args or return value)..." << endl;
	tictoc(hello);
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
	decltype(auto) tv = tictoc(hi2, 2);
	tv = -1;
	cout << "Returning by reference...\ttest_var=" << test_var << " vs. tv=" << tv << endl;

	cout << "Testing functor passed by pointer..." << endl;
	cout << setprecision(2) << tictoc(test_functor, 5.0) << endl;
	cout << "\n\n" << endl;


	cout << "Testing functor passed by class type (and mixing/nesting tictoc call types)..." << endl;
	tic();
	cout << setprecision(2) << tictoc(functor{}, 5.0) << endl;
	toc();
	cout << "\n" << endl;


	cout << "Testing cumulative timing..." << endl;
	tic();
	cout << "tic #1" << endl;
	toc();
	cout << "tic #2" << endl;
	toc();
	cout << "tic #3" << endl;
	toc();
	cout << "tic #4" << endl;
	toc();
	cout << "tic #5" << endl;
	toc();

	return 0;
}
#endif
/* Test tictoc() */
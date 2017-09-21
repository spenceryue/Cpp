#ifndef RANGE_H
#define RANGE_H

#include <type_traits>		// std::enable_if_t, is_same_v, common_type_t, conditional_t, is_arithmetic_v
#include <iterator>			// std::input_iterator_tag
#include "report_errors.h"
#include "type_stuff.h"
#include "faces.h"

#include <tuple>
#include "print_tuple.h"
#include <iostream>


template <bool is_unused = false, class T = char>
class range
{
	T current;
	const T stop;
	const T step;
	constexpr static struct null_sentinel {} sentinel{};
	struct unused {~unused() {}};

public:
	using iterator_category = std::input_iterator_tag;
	using value_type = std::conditional_t<is_unused, unused, T>;
	using difference = void;
	using pointer = T*;
	using reference = T&;

	auto operator* () const
	{
		if constexpr (is_unused)
			return unused{};
		else
			return current;
	}

	auto& operator++ ()
	{
		current += step;
		return *this;
	}

	auto operator++ (int)
	{
		range copy{*this};
		current += step;
		return copy;
	}

	template <class Any>
	bool operator!= (const Any&) const
	{
		return current < stop;
	}

	template <class Any>
	bool operator== (const Any&) const
	{
		return current >= stop;
	}

	template <bool U = false, class A = int, class B = A, class C = A, std::enable_if_t<std::is_arithmetic_v<C>, int> =0>
	explicit range (A start, B stop, C step = 1) : // Explicit prevents candidate interpretation as copy constructor during class template resolution
	current(start), stop(stop), step(step)
	{
		if (step <= 0)
			throw_err("Oh no! The step argument for range() should be positive.");
	}

	template <bool U = false, class A = int, std::enable_if_t<std::is_arithmetic_v<A>, int> =0>
	explicit range (A stop = 0) :
	range(0, stop)
	{}

	range (const range& copy) :
	current(copy.current), stop(copy.stop), step(copy.step)
	{}

	range (range&& other) noexcept :
	current(other.current), stop(other.stop), step(other.step)
	{}

	auto& begin ()
	{
		return *this;
	}

	auto& end () const
	{
		return sentinel;
	}

	/* Pretty print */
	friend std::ostream& operator<< (std::ostream& output, const range& R)
	{
		return output << type_name<range>() << std::tuple{R.current, R.stop, R.step};
	}
};

/* Deduction guides */
template <bool U = false, class A = int, class B = A, class C = A, std::enable_if_t<std::is_arithmetic_v<C>, int> =0>
explicit range (A start, B stop, C step = 1) -> range<U, std::common_type_t<A, B, C>>;

template <bool U = false, class A = int, std::enable_if_t<std::is_arithmetic_v<A>, int> =0>
explicit range (A stop = 0) -> range<U, A>;

#endif /* RANGE_H */



/* Test range */
#if __INCLUDE_LEVEL__ == 0 && defined __INCLUDE_LEVEL__
	#include <iostream> 		// std::cout, std::endl
	#include <iomanip> 			// std::setw, std::left
	#include <numeric>			// std::accumulate
	#include <vector>			// std::vector
	#include <unordered_map>	// std::unordered_map
	#include <string>			// std::string
	#include <climits>
	#include <algorithm>		// std::for_each
	#include <utility>			// std::make_pair
	#include <exception>		// std::exception
	#include "tictoc.h"
	#include "argmax.h"
	#include "faces.h"
	#include "type_stuff.h"
	#include "basename.h"

using namespace std;
int main(int argc, char* argv[])
{
	cout << pikachu << "\n" << endl;

	startup_msg(argv[0]);

	cout << range(10) << endl;
	cout << type_name<decltype(range(10).end())>() << endl;

	try
	{
		cerr << "(Purposeful screw up)" << endl;
		range(0,0,-1);
	}
	catch (const std::exception& e)
	{}


	using T = unsigned long long;
	constexpr T N = (INT_MAX-1);
	cout << (void*)N << endl;
	cout << type_name<decltype(range(0,N,5))>() << "\n\n" <<endl;

	tic();
	for (auto i : range<1,int>(0,25,3))
		volatile auto ii = i;
	toc();

	tic();
	for (auto i : range(0,25,3))
		cout << "hi " << i << endl;
	toc();

	cout << endl << endl;
	cout << left << setw(20) << 'a' << range('a') << endl;
	cout << left << setw(20) << INT_MAX << range(INT_MAX) << endl;
	cout << left << setw(20) << LONG_MAX << range(LONG_MAX) << endl;
	cout << left << setw(20) << ULONG_MAX << range(ULONG_MAX) << endl;
	cout << left << setw(20) << LLONG_MAX << range(LLONG_MAX) << endl;
	cout << left << setw(20) << 111.5 << range(0,111.5) << endl;
	cout << left << setw(20) << 111.5f << range(0,LLONG_MAX,111.5f) << endl;
	cout << endl << endl;

	auto ranged = [=]
	{
		T result = 0;
		for (auto i : range(0,N,5))
		{
			result += (i%2) ? i : -i;
		}
		return result;
	};

	auto traditional = [=]
	{
		T result = 0;
		for (T i=0; i<N; i+=5)
			result += (i%2) ? i : -i;
		return result;
	};

	auto stl = [=]
	{
		auto r = range(0,N,5);
		return accumulate(r, r, T(0), [] (T result, T i)
			{
				return (i%2) ? result + i : result - i;
			});
	};

	std::unordered_map<int, string> names {{0, "Ranged"}, {1, "Traditional"}, {2, "STL Accumulate"}};
	vector<double> times(3);
	vector<double> score(3,0);
	for (auto i : range<1>(3))
	{
		{
			tic<0>();
			T total = stl();
			double a = toc<0>();
			times[2] = a;
			cout << setw(14 + 4) << right << "stl accumulate: " << setw(7 + 4) << times[2] << "\t(odr-using output..." << (void*)(total) <<")" << endl;
		}

		{
			tic<0>();
			T total = ranged();
			double a = toc<0>();
			times[0] = a;
			cout << setw(14 + 4) << right << "Ranged: " << setw(7 + 4) << times[0] << "\t(odr-using output..." << (void*)(total) <<")" << endl;
		}

		{
			tic<0>();
			T total = traditional();
			double a = toc<0>();
			times[1] = a;
			cout << setw(14 + 4) << right << "traditional: " << setw(7 + 4) << times[1] << "\t(odr-using output..." << (void*)(total) <<")" << endl;
		}

		cout << setw(14 + 4) << left << "best: " << setw(7 + 4) << right << names[argmin(times)] << endl;

		++score[argmin(times)];

		cout << endl;
	}

	cout << endl;
	cout << "Scores:" << endl;
	int i=0;
	for (auto e: score)
		cout << setw(14 + 3) << names[i++] + ":" << setw(7 + 4) << e << endl;

	return 0;
}
#endif
/* Test range */
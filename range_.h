#ifndef RANGE_H
#define RANGE_H

#include <type_traits>		// std::enable_if_t, common_type_t, conditional_t, is_arithmetic_v
#include <iterator>			// std::input_iterator_tag
#include "report_errors.h"
#include "type_stuff.h"
#include "faces.h"

#include <tuple>			// tuple
#include "print_tuple.h"
#include <iostream>			// basic_ostream


namespace range_detail
{
	struct Unused{~Unused() {}} unused;


	template
	<
		bool IS_UNUSED = false,
		class T = char
	>
	struct range_iterator
	{
		using iterator_category = std::input_iterator_tag;
		using value_type = std::conditional_t<IS_UNUSED, Unused, T>;
		using difference = void;
		using pointer = value_type*;
		using reference = value_type&;


		T current;
		const T stop;
		const T step;
		const bool INCREMENT;


		range_iterator (T current, T stop, T step) :
		current {current},
		stop {stop},
		step {step},
		INCREMENT {step > 0}
		{}


		value_type operator* () const
		{
			if constexpr (IS_UNUSED)	return unused;
			else						return current;
		}


		value_type& operator++ ()
		{
			current += step;
			if constexpr (IS_UNUSED)	return unused;
			else						return current;
		}


		value_type operator++ (int)
		{
			T copy {current};
			current += step;
			if constexpr (IS_UNUSED)	return unused;
			else						return copy;
		}


		template
		<
			class Any
		>
		bool operator== (const Any&) const
		{
			if (INCREMENT)		return current >= stop;
			else				return current <= stop;
		}


		template
		<
			class Any
		>
		bool operator!= (const Any&) const
		{
			if (INCREMENT)		return current < stop;
			else				return current > stop;
		}
	};
}


template
<
	bool IS_UNUSED = false,
	class T = char
>
struct range : public range_detail::range_iterator<IS_UNUSED, T>
{
	using range_iterator = range_detail::range_iterator<IS_UNUSED, T>;

	template
	<
		class A,
		class B,
		class C = B,
		class Common = std::common_type_t<A,B,C>,
		std::enable_if_t<std::is_arithmetic_v<Common>, int> =0
	>
	explicit range (A start, B stop, C step = 1) :
	range_iterator (start, stop, step)
	{
		if (this->step == 0)									throw_err(type_name<range>() + ": invalid argument error: step == 0");
		else if (this->step > 0 && this->current > this->stop)	throw_err(type_name<range>() + ": invalid argument error: step > 0 and start > stop");
		else if (this->step < 0 && this->current < this->stop)	throw_err(type_name<range>() + ": invalid argument error: step < 0 and start < stop");
	}


	template
	<
		std::enable_if_t<std::is_arithmetic_v<T>, int> =0
	>
	explicit range (T stop = 0) :
	range { (T)0, stop }
	{}


	auto begin () const
	{
		return *this;
	}


	auto end () const
	{
		return range_detail::unused;
	}


	/* Pretty print */
	template
	<
		class Ch,
		class Tr
	>
	friend std::basic_ostream<Ch, Tr>& operator<< (std::basic_ostream<Ch, Tr>& out, const range& r)
	{
		return out << type_name<range>() << std::tuple{r.current, r.stop, r.step};
	}
};


/* Deduction guide */
template
<
	bool IS_UNUSED = false,
	class T = char,
	class A,
	class B,
	class C = B,
	class Common = std::common_type_t<A,B,C>,
	std::enable_if_t<std::is_arithmetic_v<Common>, int> =0
>
explicit range (A start, B stop, C step = 1) -> range<IS_UNUSED, Common>;


#endif /* RANGE_H */



/* Test RANGE */
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
	#include <cmath>			// HUGE_VALL


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
		range(0,1,-1);
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

	[[maybe_unused]] auto ranged = [=]
	{
		T result = 0;
		for (auto i : range(0,N,5))
		{
			result += (i%2) ? i : -i;
		}
		return result;
	};

	[[maybe_unused]] auto traditional = [=]
	{
		T result = 0;
		for (T i=0; i<N; i+=5)
			result += (i%2) ? i : -i;
		return result;
	};

	[[maybe_unused]] auto stl = [=]
	{
		auto r = range(0,N,5);
		return accumulate(r, r, T(0), [] (T result, T i)
			{
				return (i%2) ? result + i : result - i;
			});
	};

	[[maybe_unused]] auto for_e = [=]
	{
		auto r = range(0,N,5);
		T result = 0;
		for_each(r, r, [&] (T i)
			{
				result += (i%2) ? i : -i;
			});
		return result;
	};

	std::unordered_map<int, string> names {{0, "Ranged"}, {1, "Traditional"}, {2, "STL Accumulate"}, {3, "STL For Each"}};
	vector<double> times(names.size());
	vector<double> score(names.size(),0);
	for (auto i : range<1>(12))
	{

		{
			tic<0>();
			T total = traditional();
			double a = toc<0>();
			times[1] = a;
			cout << setw(14 + 4) << right << "traditional: " << setw(7 + 4) << times[1] << "\t(odr-using output..." << (void*)(total) <<")" << endl;
		}

		{
			tic<0>();
			T total = for_e();
			double a = toc<0>();
			times[3] = a;
			cout << setw(14 + 4) << right << "stl for_each: " << setw(7 + 4) << times[3] << "\t(odr-using output..." << (void*)(total) <<")" << endl;
		}

		/*{
			tic<0>();
			T total = stl();
			double a = toc<0>();
			times[2] = a;
			cout << setw(14 + 4) << right << "stl accumulate: " << setw(7 + 4) << times[2] << "\t(odr-using output..." << (void*)(total) <<")" << endl;
		}*/
		times[2] = HUGE_VALL;

		{
			tic<0>();
			T total = ranged();
			double a = toc<0>();
			times[0] = a;
			cout << setw(14 + 4) << right << "Ranged: " << setw(7 + 4) << times[0] << "\t(odr-using output..." << (void*)(total) <<")" << endl;
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
/* Test RANGE */
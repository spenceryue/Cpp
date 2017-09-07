#ifndef GET_RANDOM_H
#define GET_RANDOM_H
#include <random> 				// std::random_device, seed_seq, default_random_engine, mt19937, uniform_real_distribution, uniform_int_distribution
#include <type_traits>			// is_floating_point_v
#include <utility>				// std::make_index_sequence, index_sequence
#include "tictoc.h"

namespace get_random_ns {
	template <class T,
	size_t...i,
	class R = std::conditional_t< std::is_floating_point_v<T>,
								  std::uniform_real_distribution<T>,
								  std::uniform_int_distribution<T>>>
	auto get_random (T min, T max, std::index_sequence<i...>)
	{
		using namespace std;
		// static random_device rd;
		static auto rd = [] {auto e = toc<0, nano>(); /*cout << setprecision(20) << e << endl;*/ return (unsigned int)(e)%2 ? (unsigned int)(e*3 + 1) : (unsigned int)(e/2);};

		seed_seq s = { (void(i), rd())... };
		// default_random_engine gen {s};
		std::mt19937 gen {s};

		return [=] () mutable {return R(min,max)(gen);};
	}
}


template <
class A = int,
size_t NUM_SEEDS = 6,
class B = A,
class C = std::common_type_t<A,B>,
class Indices = std::make_index_sequence<NUM_SEEDS>>
auto get_random (A min, B max)
{
	return get_random_ns::get_random<C>(min, max, Indices{});
}

template <
size_t NUM_SEEDS,
class A = int,
class B = A,
class C = std::common_type_t<A,B>,
class Indices = std::make_index_sequence<NUM_SEEDS>>
auto get_random (A min, B max)
{
	return get_random_ns::get_random<C>(min, max, Indices{});
}

template <
class T = int,
size_t NUM_SEEDS = 6,
bool value = std::is_floating_point_v<T>,
class Indices = std::make_index_sequence<NUM_SEEDS>>
auto get_random (T max = value ? T(1.0) : T(100))
{
	return get_random_ns::get_random<T>(0, max, Indices{});
}

template <
size_t NUM_SEEDS,
class T = int,
bool value = std::is_floating_point_v<T>,
class Indices = std::make_index_sequence<NUM_SEEDS>>
auto get_random (T max = value ? T(1.0) : T(100))
{
	return get_random_ns::get_random<T>(0, max, Indices{});
}

#endif /* GET_RANDOM_H */



/* Test get_random */
#if __INCLUDE_LEVEL__ == 0 && defined __INCLUDE_LEVEL__
	#include <iostream> 		// std::cout, std::endl
	#include <string> 			// std::string
	#include <algorithm> 		// std::for_each, generate
	#include "range.h"
	#include "zip.h"
	#include "enumerate.h"
	#include "type_stuff.h"

using namespace std;
int main(int argc, char* argv[])
{
	cout << "Hello World!" << endl;
	cout << endl << endl;


	auto rand = get_random();
	cout << type_name<decltype(rand)>() << endl;
	cout << endl << endl;


	auto r = range(10);
	for_each (begin(r), begin(r), [&](auto i)
	{
		cout << "A random number was born: " << rand() << endl;
	});
	cout << endl << endl;


	auto rand2 = get_random(5.0);
	for (auto i : range<1>(10))
		cout << "Range loop producing random number: " << rand2() << endl;
	cout << endl << endl;


	auto rand3 = get_random<char>('a','z');
	string a(10,' ');
	generate(a.begin(), a.end(), rand3);
	string b(10,' ');
	generate(b.begin(), b.end(), rand3);

	// for (auto [i, c] : enumerate<unsigned int>(zip(a,b))) {
	for (auto [i, c] : enumerate<unsigned int>(zip(range<1>(1),range(2)))) {
		cout << '[' << i << "]: (" << get<1>(c) << "," << get<1>(c) << ")";
		if (i < a.size()-1)
			cout << ", ";
		else
			cout << endl;
	}
	cout << endl << endl;


	auto rand4 = get_random<2, float>();
	auto rand5 = get_random<float, 3>();
	cout << type_name<decltype(rand4)>() << endl;
	cout << type_name<decltype(rand5)>() << endl;
	cout << endl << endl;


	return 0;
}
#endif
/* Test get_random */
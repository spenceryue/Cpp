#ifndef GET_RANDOM_H
#define GET_RANDOM_H
#include <random> 				// std::random_device, seed_seq, default_random_engine, mt19937, uniform_real_distribution, uniform_int_distribution
#include <type_traits>			// is_floating_point_v
#include <utility>				// std::make_index_sequence, index_sequence
#include "tictoc.h"

namespace random_detail {
	using namespace std;

	template
	<
		class T,
		size_t...index
	>
	auto random (T min, T max, index_sequence<index...>)
	{
		using Real = uniform_real_distribution<T>;
		using Integer = uniform_int_distribution<T>;
		using Distribution = conditional_t<is_floating_point_v<T>, Real, Integer>;


		static auto random_source = []
		{
			unsigned int e = toc<0, nano>();
			if (e % 2)			return e*3 + 1;
			else				return e/2;
		};


		seed_seq seed =
		{
			((void) index, random_source())
			...
		};
		mt19937 generator {seed};


		return [=] () mutable
		{
			return Distribution(min, max)(generator);
		};
	}
}


template
<
	class A = int,
	size_t SEEDS = 6,
	class B = A
>
auto random (A min, B max)
{
	using C = std::common_type_t<A,B>;
	using Indices = std::make_index_sequence<SEEDS>;

	return random_detail::random<C>(min, max, Indices{});
}


template
<
	class T = int,
	size_t SEEDS = 6,
	bool IS_FLOAT = std::is_floating_point_v<T>
>
auto random (T max = IS_FLOAT ? T(1.0) : T(100))
{
	using Indices = std::make_index_sequence<SEEDS>;

	return random_detail::random<T>(0, max, Indices{});
}

#endif /* GET_RANDOM_H */



/* Test random */
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


	auto rand = random();
	cout << type_name<decltype(rand)>() << endl;
	cout << endl << endl;


	auto r = range(10);
	for_each (begin(r), begin(r), [&](auto i)
	{
		cout << "A random number was born: " << rand() << endl;
	});
	cout << endl << endl;


	auto rand2 = random(5.0);
	for (auto i : range<1>(10))
		cout << "Range loop producing random number: " << rand2() << endl;
	cout << endl << endl;


	auto rand3 = random<char>('a','z');
	string a(10,' ');
	generate(a.begin(), a.end(), rand3);
	string b(10,' ');
	generate(b.begin(), b.end(), rand3);

	// for (auto [i, c] : enumerate<unsigned int>(zip(a,b))) {
	/*for (auto [i, c] : enumerate<unsigned int>(zip(range<1>(1),range(2)))) {
		cout << '[' << i << "]: (" << get<1>(c) << "," << get<1>(c) << ")";
		if (i < a.size()-1)
			cout << ", ";
		else
			cout << endl;
	}
	cout << endl << endl;*/


	auto rand4 = random<float, 3>();
	cout << type_name<decltype(rand4)>() << endl;
	cout << endl << endl;


	return 0;
}
#endif
/* Test random */
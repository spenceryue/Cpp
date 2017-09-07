#ifndef USE_COMMAS_H
#define USE_COMMAS_H

#include <type_traits>		// std::is_arithmetic_v, std::is_floating_point_v
#include <utility>			// std::move
#include <sstream>			// std::stringstream
#include <string>			// std::string
#include <iterator>			// std::ostream_iterator
#include <algorithm>		// std::for_each, std::copy
#include <cmath>			// std::pow
#include "faces.h"


template<unsigned int precision = 6, class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
std::string use_commas(T x)
{
	using namespace std;

	stringstream result;

	// Determine sign
	bool negative = x < 0;
	if (negative) {
		result.put('-');
		x *= -1;
	}
	
	// Separate integer and decimal components (at precision specified)
	int integer = x;
	int decimal = (x - integer) * pow(10, precision);

	// Insert commas in integer component
	string support{ to_string(integer) };
	for_each(cbegin(support), cend(support), [&, count = (3 - support.size()%3) % 3] (char c) mutable
	{
		if (count && count%3 == 0)
			result.put(',');

		result.put(c);
		count++;
	});

	if constexpr(is_floating_point_v<T>)
	{
		// Trim decimal trailing zeros
		while (decimal % 10 == 0) {
			decimal /= 10;
		}

		// Append fractional component	
		result.put('.');
		support = move(to_string(decimal));
		copy(cbegin(support), cend(support), ostream_iterator<char>(result, ""));
	}

	return result.str();
}
#endif /* USE_COMMAS_H */



/* Test use_commas() */
#if __INCLUDE_LEVEL__ == 0 && defined __INCLUDE_LEVEL__
	#include <iostream>
	#include <ostream>

using namespace std;

int main(int argc, char ** argv) {
	double a = -3453.120;

	cout << use_commas(235678.546) << endl;
	cout << use_commas(0.305) << endl;
	cout << std::to_string(0.305) << endl;
	cout << use_commas(24624305) << endl;
	cout << use_commas(a) << endl;
	cout << use_commas(static_cast<int>(a)) << endl;

	/*
	// error: no matching function for call to 'use_commas(const char [9])'
	cout << use_commas("23579835") << endl;
	*/
}
#endif
/* Test use_commas() */
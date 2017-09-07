#ifndef ARGMAX_H
#define ARGMAX_H
#include <algorithm>		// std::max_element, min_element
#include <type_traits>		// std::enable_if_t, std::void_t, std::is_same_v
#include <utility>			// std::declval
#include <iterator>			// std::declval
#include "is_valid.h"

namespace argmax_ns {
	template <class T>
	using has_difference_operator = decltype(std::declval<T>() - std::declval<T>());

	template <class T>
	using is_iterable = decltype(std::begin(std::declval<T>()), std::end(std::declval<T>()));
}


template <class ForwardIt, std::enable_if_t< !is_valid_v<argmax_ns::has_difference_operator, ForwardIt>,int> =0>
constexpr size_t argmax (ForwardIt first, ForwardIt last)
{
	return std::max_element(first, last) - first;
}

template <class ForwardIt, std::enable_if_t< is_valid_v<argmax_ns::has_difference_operator, ForwardIt>,int> =0>
size_t argmax (ForwardIt first, ForwardIt last)
{
	int index = 0;
	auto max = *(first++);
	for (int i=1; first != last; ++i, ++first)
	{
		if (max < *first)
		{
			max = *first;
			index = i;
		}
	}

	return index;
}

template <class T, std::enable_if_t< is_valid_v<argmax_ns::is_iterable, T>,int> =0>
constexpr size_t argmax (T& container) {
	return argmax(std::begin(container), std::end(container));
}

template <class ForwardIt, class Compare, std::enable_if_t< is_valid_v<argmax_ns::has_difference_operator, ForwardIt>,int> =0>
constexpr size_t argmax(ForwardIt first, ForwardIt last, Compare cmp)
{
	return std::max_element(first, last, cmp) - first;
}


/* argmin */
template <class ForwardIt, std::enable_if_t< !is_valid_v<argmax_ns::has_difference_operator, ForwardIt>,int> =0>
constexpr size_t argmin (ForwardIt first, ForwardIt last)
{
	return std::min_element(first, last) - first;
}

template <class ForwardIt, std::enable_if_t< is_valid_v<argmax_ns::has_difference_operator, ForwardIt>,int> =0>
size_t argmin (ForwardIt first, ForwardIt last)
{
	int index = 0;
	auto min = *(first++);
	for (int i=1; first != last; ++i, ++first)
	{
		if (*first < min)
		{
			min = *first;
			index = i;
		}
	}

	return index;
}

template <class T, std::enable_if_t< is_valid_v<argmax_ns::is_iterable, T>,int> =0>
constexpr size_t argmin (T& container) {
	return argmin(std::begin(container), std::end(container));
}

template <class ForwardIt, class Compare, std::enable_if_t< is_valid_v<argmax_ns::has_difference_operator, ForwardIt>,int> =0>
constexpr size_t argmin(ForwardIt first, ForwardIt last, Compare cmp)
{
	return std::min_element(first, last, cmp) - first;
}

#endif /* ARGMAX_H */


/* Test argmax */
#if __INCLUDE_LEVEL__ == 0 && defined __INCLUDE_LEVEL__
	#include <iostream> 		// std::cout, std::endl
	#include <vector> 			// std::vector
	#include <cmath> 			// std::abs
	#include "type_stuff.h"
	#include "faces.h"

using namespace std;
int main(int argc, char* argv[])
{
	cout << "Hello World!" << "\n" << endl;

	vector<int> a 		{124, 15, 11661, 6626, 357357, 33535, 545544, 11555, 752, 4666886, 0, -12};
	constexpr int b[]	{124, 15, 11661, 6626, 357357, 33535, 545544, 11555, 752, 4666886, 0, -12};

	constexpr auto compare = [] (int p, int q) {return (p%10 < q%10);};
	{
		auto s = begin(a);
		auto f = end(a);
		for (int i=0; s!=f; ++i, ++s) {
			if (i)
				cout << "  ";
			cout << "[" << i << "]:" << *s/10 << "(" << (abs(*s)%10) << ")";
		}
		cout << "\n" << endl;
	}
	

	cout << "size: " << a.size() << endl;
	cout << "argmax(a): " << argmax(a) << endl;
	cout << "argmax(a.begin(), a.end()): " << argmax(a.begin(), a.end()) << endl;
	cout << "argmax(a.begin(), a.end(), compare): " << argmax(a.begin(), a.end(), compare) << endl;
	cout << endl;


	cout << "argmax(b): " << argmax(b) << endl;
	cout << "argmax(begin(b), end(b)): " << argmax(begin(b), end(b)) << endl;
	cout << "argmax(begin(b), end(b), compare): " << argmax(begin(b), end(b), compare) << endl;
	cout << "argmax(begin(b), end(b)): " << argmax(begin(b), end(b)) << endl;
	cout << endl;


	cout << "size: " << a.size() << endl;
	cout << "argmin(a): " << argmin(a) << endl;
	cout << "argmin(a.begin(), a.end()): " << argmin(a.begin(), a.end()) << endl;
	cout << "argmin(a.begin(), a.end(), compare): " << argmin(a.begin(), a.end(), compare) << endl;
	cout << endl;


	cout << "argmin(b): " << argmin(b) << endl;
	cout << "argmin(begin(b), end(b)): " << argmin(begin(b), end(b)) << endl;
	cout << "argmin(begin(b), end(b), compare): " << argmin(begin(b), end(b), compare) << endl;
	cout << "argmin(begin(b), end(b)): " << argmin(begin(b), end(b)) << endl;

	return 0;
}
#endif
/* Test argmax */
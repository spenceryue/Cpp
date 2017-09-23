#ifndef PRINT_TUPLE_H
#define PRINT_TUPLE_H
#include <iostream>				// std::basic_ostream
#include <tuple>				// std::tuple
#include <utility>				// std::index_sequence, index_sequence_for
#include "wrap_references.h"

/* Source: http://en.cppreference.com/w/cpp/utility/integer_sequence */
namespace print_tuple_detail {
	std::string SEP = ", ";

	template<class Ch, class Tr, class Tuple, size_t... I>
	inline void print_tuple (std::basic_ostream<Ch,Tr>& output, const Tuple& tup, std::index_sequence<I...>)
	{
	    ((output << (I == 0? "" : SEP) << unwrap_references(std::get<I>(tup))), ...); // unwrap tuple elements which are of type std::reference_wrapper, forward all others
	}
}

template<class Ch, class Tr, class... Args>
std::basic_ostream<Ch,Tr>& operator<< (std::basic_ostream<Ch,Tr>& output, const std::tuple<Args...>& tup)
{
    output << "(";
    print_tuple_detail::print_tuple(output, tup, std::index_sequence_for<Args...>{});
    return output << ")";
}
#endif /* PRINT_TUPLE_H */



/* Test print_tuple */
#if __INCLUDE_LEVEL__ == 0 && defined __INCLUDE_LEVEL__
	#include <iostream> 		// std::cout, std::endl
	#include <string> 			// std::string
	#include <tuple> 			// std::tuple
	#include "faces.h"
	#include "basename.h"
	#include "type_stuff.h"

using namespace std;
using namespace string_literals;
int main(int argc, char* argv[])
{
	cout << pikachu << "\n" << endl;
	startup_msg(argv[0]);

	auto a = "yay"s;
	auto b = tuple<string, const char*, char, string&, int, float, void*>{"hello"s, "world", ' ', a, 5, 5.2f, (void*) LLONG_MAX};
	auto c = tuple{"hello"s, "world", ' ', a, 5, 5.2f, LLONG_MAX};

	cout << type_name<decltype(b)>() << "\n\t" << b << "\n" << endl;
	cout << type_name<decltype(c)>() << "\n\t" << c << "\n" << endl;

	return 0;
}
#endif
/* Test print_tuple */
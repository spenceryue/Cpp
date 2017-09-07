#ifndef TYPES_H
#define TYPES_H

#include <type_traits>			// std::is_void_v, std::invoke_result_t, std::enable_if_t
#include <string>				// std::string
#include "check_signature.h"	// (package this along)


template <class T>
constexpr
std::string type_name()
{
	using std::string;
#ifdef __clang__
	string p = __PRETTY_FUNCTION__;
	return string(p.data() + 34, p.size() - 34 - 1);
#elif defined(__GNUC__)
	string p = __PRETTY_FUNCTION__;
	#if __cplusplus < 201402
		return string(p.data() + 36, p.size() - 36 - 1);
	#else
		return string(p.data() + 53, p.size() - 53 - 58); /* Tuned as of 8.23.2017 */
		// return string(p.data(), p.size());
	#endif
#elif defined(_MSC_VER)
	string p = __FUNCSIG__;
	return string(p.data() + 38, p.size() - 38 - 7);
#endif
}
/* Source: https://stackoverflow.com/a/20170989/3624264 */

/* My extension of the above ( ͡° ͜ʖ ͡°) */
template <class T, class ...Ts>
constexpr
std::string type_names(std::string sep = ", ")
{
	return (type_name<T>() + ... + (sep + type_name<Ts>()) );
}

template <class ...Ts, std::enable_if_t<sizeof...(Ts) == 0, int> =0>
constexpr
std::string type_names(std::string sep = ", ")
{
	return "";
}

template <auto* func, class ...ArgTypes>	// for free functions
constexpr inline bool has_void_return() {
	return std::is_void_v< std::invoke_result_t< decltype(*func), ArgTypes... >>;
}

template <class F, class ...ArgTypes>	// for lambdas and functors
constexpr inline bool has_void_return() {
	return std::is_void_v< std::invoke_result_t< F, ArgTypes... >>;
}
#endif /* TYPES_H */


/* Test types */
#if __INCLUDE_LEVEL__ == 0 && defined __INCLUDE_LEVEL__
	#include <iostream>			// std::cout, std::endl, std::boolalpha
	#include <iomanip>			// std::setw, std::left
	#include <string>			// std::string

using namespace std;

void hello()
{
	cout << "hello world" << endl;
}

struct callme {
	string operator() (double d) {
		cout << "lookey here: " << d << endl;
		return std::to_string(d);
	}
	using signature = decltype(&callme::operator());	// technique needed for printing type_name
};


int main(int argc, char* argv[])
{
	cout << argv[0] << " Starting..." << endl;
	cout << boolalpha;
	cout << left;

	cout << setw(50) << "Check out type_names()! Pretty sexy: "		<< setw(12) << '\t' << type_names<double,int*>() << endl;
	cout << setw(50) << "Check out type_names()! empty overload: "	<< setw(12) << '\t' << "\"" << type_names<>() << "\"" << endl;
	cout << setw(50) << "Check out type_names()! one type: "		<< setw(12) << '\t' << type_names<std::string>() << endl;

	cout << setw(50) << "(free function) hello -> void: " 			<< setw(8) << has_void_return<hello>()
									  								<< setw(4) << '\t' << type_name<decltype(hello)>() << endl;
	
	cout << setw(50) << "(free function) main -> void: " 			<< setw(8) << has_void_return<main,int,char**>()
									  								<< setw(4) << '\t' << type_name<decltype(main)>() << endl;

	auto hi = [] (int a, string b)
	{
		cout << "hi there" << a << b << endl;
	};
	cout << setw(50) << "(lambda) hi -> void: "						<< setw(8) << has_void_return<decltype(hi), int, string>()
									  								<< setw(4) << '\t' << type_name<decltype(hi)>() << endl;
	
	cout << setw(50) << "(functor) callme -> void: " 				<< setw(8) << has_void_return<callme,double>()
									  								<< setw(4) << '\t' << type_name<callme::signature>() << endl;

	return 0;
}
#endif
/* Test types */
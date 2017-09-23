#ifndef IS_VALID_H
#define IS_VALID_H
#include <iostream> 			// std::cout, std::endl
#include <type_traits> 			// std::true_type, false_type, invoke_result_t

namespace is_valid_detail {
	template <template<class...> class, class = void, class...>
	struct valid : std::false_type
	{
		static constexpr bool value = false;
	};

	template <template<class...> class Expression, class...T>
	struct valid <Expression, std::void_t<Expression<T...>>, T...> : std::true_type
	{
		static constexpr bool value = true;
	};
}

template <template<class...> class Expression, class...T>
constexpr auto is_valid_v = is_valid_detail::valid<Expression, void, T...>::value;

template <class ...T, class Lambda>
constexpr bool is_valid (Lambda c)
{
	using namespace is_valid_detail;
	return is_valid_v<std::invoke_result_t, Lambda, T...>;
}

#define make_expr(expression...)		[] (auto&& ...x) -> decltype(((expression),...)) {}
#endif /* IS_VALID_H */



/* Test is_valid */
#if __INCLUDE_LEVEL__ == 0 && defined __INCLUDE_LEVEL__
	#include <iostream> 		// std::cout, std::endl
	#include <iomanip>	 		// std::setw, std::left, std::setfill
	#include <string> 			// std::string
	#include <vector> 			// std::vector
	#include <utility>			// std::declval
	#include <type_traits>		// std::invoke_result_t
	#include "type_stuff.h"
	#include "faces.h"

struct test
{
	void foo() {};
};

int main(int argc, char* argv[])
{
	using namespace std;
	using namespace is_valid_detail;

	auto hi = [] (auto&& ...x) -> decltype(((x.data()), ...)) {};
	auto hi2 = [] (auto&& ...x) -> decltype(((x.begin()),...)) {};

	cout << "Hello World!" << "\n" << endl;

	cout << left << /*boolalpha <<*/ setw(70) << setfill('.') << "accessing member function" << type_name<decltype(test::foo)>() << "\n" << endl;
	cout << left << /*boolalpha <<*/ setw(70) << setfill('.') << "lambda_t<>" << is_valid_v<invoke_result_t, decltype(hi), string, vector<int>> << "\n" << endl;
	cout << left << /*boolalpha <<*/ setw(70) << setfill('.') << "is_valid<string>(hi)" << is_valid<string>(hi) << "\n" << endl;
	cout << left << /*boolalpha <<*/ setw(70) << setfill('.') << "is_valid<int>(hi2)" << is_valid<int>(hi2) << "\n" << endl;
	cout << left << /*boolalpha <<*/ setw(70) << setfill('.') << "is_valid<string>(make_expr(x.end()))" << is_valid<string>(make_expr(x.end())) << "\n" << endl;
	cout << left << /*boolalpha <<*/ setw(70) << setfill('.') << "is_valid<string, vector<int>>(make_expr(x.end(), x.begin()))" << is_valid<string, vector<int>>(make_expr(x.end(), x.begin())) << "\n" << endl;

	cout << "\n" << type_name<invoke_result_t<decltype(hi), vector<int>, string>>() << endl;
	cout << "\n" << boolalpha << is_valid_v<invoke_result_t, decltype(hi2), vector<int>, string> << endl;
	return 0;
}
#endif
/* Test is_valid */
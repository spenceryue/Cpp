#ifndef WRAP_REFERENCES_H
#define WRAP_REFERENCES_H
#include <functional>			// std::reference_wrapper, ref
#include <type_traits> 			// std::enable_if_t, remove_reference_t, is_rvalue_reference_v
#include "is_valid.h"


namespace wrap_references_ns {
	template <class T, class U = std::remove_reference_t<T>>
	using has_type = typename U::type;
}

template <class T, std::enable_if_t<!std::is_rvalue_reference_v<T&&>, int> =0>
std::reference_wrapper<std::remove_reference_t<T>> wrap_references(T&& lvalue_ref) {
	return std::ref(lvalue_ref); // create a reference_wrapper
}

template <class T, std::enable_if_t<std::is_rvalue_reference_v<T&&>, int> =0>
T&& wrap_references(T&& rvalue) {
	return std::forward<T>(rvalue); // forward rvalue along
}

template <class T, std::enable_if_t<is_valid_v< wrap_references_ns::has_type, T> &&
												std::is_same_v<std::remove_reference_t<T>, std::reference_wrapper<wrap_references_ns::has_type<T>>>, int> =0>
typename T::type& unwrap_references(T wrapper) {
	return wrapper; // unwrap reference_wrapper
}

template <class T, std::enable_if_t<!is_valid_v< wrap_references_ns::has_type, T>, int> =0>
T&& unwrap_references(T&& value) {
	return std::forward<T>(value); // forward value along
}

#endif /* WRAP_REFERENCES_H */



/* Test wrap_references */
#if __INCLUDE_LEVEL__ == 0 && defined __INCLUDE_LEVEL__
	#include <iostream> 		// std::cout, std::endl
	#include <string> 			// std::string
	#include "faces.h"
	#include "basename.h"
	#include "type_stuff.h"

using namespace std;
using namespace string_literals;
int main(int argc, char* argv[])
{
	cout << pikachu << endl;
	cout << endl << endl;

	startup_msg(argv[0]);

	int a = 5;
	auto b = wrap_references(5);
	cout << type_name<decltype(b)>() << endl;
	auto c = wrap_references(a);
	cout << type_name<decltype(c)>() << endl;

	cout << (unwrap_references(b), "") << type_name<decltype(unwrap_references(b))>() << endl;
	cout << (unwrap_references(c), "") << type_name<decltype(unwrap_references(c))>() << endl;

	return 0;
}
#endif
/* Test wrap_references */
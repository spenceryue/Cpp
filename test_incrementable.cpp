#include <iostream>
#include <type_traits>

template <class ...>
using void_t = void;

template <class T, class = void>
struct is_incrementable : public std::false_type { };
template <class T>
struct is_incrementable<T, void_t<decltype(++(std::declval<T&>()))>> : public std::true_type { };
#include "type_stuff.h"
int main()
{
    std::cout << std::boolalpha;
    std::cout << is_incrementable<int>::value << std::endl;
    std::cout << type_name<void_t<decltype(++(std::declval<int&>()))>>() << std::endl;
    return 0;
}

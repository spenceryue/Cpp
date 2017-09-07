#ifndef ENUMERATE_H
#define ENUMERATE_H
#include <limits>				// std::numeric_limits
#include <type_traits>			// std::is_rvalue_reference_v
#include <utility>				// std::forward
#include "range.h"
#include "zip.h"


template <class T = int, class Iterable>
auto enumerate(Iterable&& I)
{
	return zip{range<0,T>(std::numeric_limits<T>::max()), std::forward<Iterable>(I)};
}

#endif /* ENUMERATE_H */



/* Test enumerate */
#if __INCLUDE_LEVEL__ == 0 && defined __INCLUDE_LEVEL__
	#include <iostream> 		// std::cout, std::endl
	#include <iomanip>	 		// std::setw
	#include <string> 			// std::string
	#include "faces.h"
	#include "basename.h"
	#include "type_stuff.h"

using namespace std;
using namespace string_literals;
int main(int argc, char* argv[])
{
	cout << pikachu << "\n" << endl;
	startup_msg(argv[0]);

	string s = "Hello World";
	enumerate(s);
	enumerate("Hello World"s);
	cout << type_name<decltype(enumerate(s))>() << endl;
	cout << type_name<decltype(enumerate("Hello World"s))>() << endl;

	for (auto [i,x] : enumerate("Heelloo Worrld"s))
	// for (auto [i,x] : enumerate(s))
		cout << "[" << i << "]: " << x << ((i==13) ? "\n" : ", ");
	cout << endl << endl;

	return 0;
}
#endif
/* Test enumerate */
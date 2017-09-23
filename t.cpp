#include <iostream>
#include <algorithm>		// std::for_each
#include <vector>			// std::vector
#include "range.h"
#include <tuple>
#include "print_tuple.h"
#include "print_vector.h"
#include "type_stuff.h"
#include "basename.h"
using namespace std;


template <class ...ArgTypes>
void test(ArgTypes&&... args)
{
	(cout << ... << (cout << args, " ")) << endl;
}

struct A {
	using type = int;
} t;

int main(int argc, char* argv[]) {
	startup_msg(argv[0]);

	test("i", "am", "a", "putzeroo", "!");

	tuple<vector<string>> a;

	{
		a = tuple<vector<string>>{vector<string>{"hello", "world"}};
	}

	void(tuple<vector<string>>{vector<string>{"helasdflo", "woadfrladd"}});

	cout << a << endl;

	cout << type_name<typename A::type>() << endl;

	cout << ((-6)  % (5 ))<< endl;
	cout << ((-6)  % (-5)) << endl;
}

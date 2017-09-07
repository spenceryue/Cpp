#include <iostream>
using std::cout;
using std::endl;


class Test
{
    int my_count;
    static int count;
public:
	Test() : my_count(++count)
	{
		cout << "Constructor (" << my_count <<") is executed\n";
	}
	Test(const Test& other) : my_count(++count)
	{
		cout << "Copy Constructor (" << my_count <<") is executed\n";
	}
	~Test()
	{
		cout << "Destructor (" << my_count <<") is executed\n";
	}
	// friend void fun(Test t); // no difference
};
int Test::count = 0;


void fun(Test t)
{
	Test();
	t.~Test();
}


int main()
{
	{	// Visible construction, copy, destruction
		Test();
		Test t;
		fun(t);
	}
	
	std::cout << std::endl;
	{	// construction/destruction happens in LIFO (stack) order
		Test t;
		Test v;
	}
	return 0;
}

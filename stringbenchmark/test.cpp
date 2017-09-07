#include <iostream>
#include <string>
using namespace std;

int main()
{
	string temp = "asdfasdf\0asdfasd";
	cout <<temp<<endl;
	cout<<temp.c_str()<<endl;
	return 0;
}

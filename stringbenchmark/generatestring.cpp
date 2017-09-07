#include <iostream>
#include <string>
#include <cstdlib>
using namespace std;

int main()
{
	char str[] = "abcdefghijklmnopqrstuvwxyz";
	char temp[256];
	register int i ,j;
	temp[0] = '"';
	temp[254] = '"';
	temp[255] = '\0';
	
	cout<<"#include <string>"<<endl;
	cout<<"std::string hugestring="<<endl;
	
	unsigned int last = 0;
	for (i=0; i<1000000; i++){
		for (j=1; j<=253; j++)
			// temp[j] = str[(char)(rand()%26)];
			temp[j] = str[last = ((j*7 + last) % 26)];
		cout<<temp<<endl;
	}
	cout<<";"<<endl;
	return 0;
}

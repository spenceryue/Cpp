#ifndef REDIRECTOR_H
#define REDIRECTOR_H

#include <iostream>			// std::ostream, std::streambuf

class Redirector
{
public:
	Redirector (std::ostream& src, std::ostream& dest)
	: src(src),
	src_buf(
		src.rdbuf(dest.rdbuf())
		),
	restored(false)
	{}
	
	~Redirector ()
	{
		if (!restored)
			restore();
	}

	void restore()
	{
		if (!restored) {
			src.rdbuf(src_buf);
			restored = true;
		}
	}

private:
	std::ostream& src;
	std::streambuf *src_buf;
	bool restored;
};
#endif /* REDIRECTOR_H */


/* Test Redirector */
#if __INCLUDE_LEVEL__ == 0 && defined __INCLUDE_LEVEL__
	#include <iostream>			// std::cout, std::endl
	#include <fstream>			// std::ofstream

using namespace std;

int main(int argc, char**argv) {

	ofstream output("out/test_redirector.txt");

	cout << "before redirect..." << endl;

	Redirector redirect(cout, output);

	cout << "after redirect: hello world" << endl;

	redirect.restore();

	cout << "source stream restored" << endl;

	return 0;
}
#endif
/* Test Redirector */
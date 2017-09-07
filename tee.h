#ifndef TEE_H
#define TEE_H

#include <iostream>			// std::streambuf, std::streamsize, std::ostream

class TeeBuffer: public std::streambuf
{
public:
	TeeBuffer (std::streambuf *sb1, std::streambuf *sb2) :
	sb1(sb1),
	sb2(sb2)
	{}

private:
	virtual int sync()
	{
		if (sb1->pubsync() || sb2->pubsync())
			return -1;
		else
			return 0;
	}
	virtual std::streamsize xsputn (const char* s, std::streamsize n)
	{
		std::streamsize written1 = sb1->sputn(s, n);
		std::streamsize written2 = sb2->sputn(s, n);

		return std::min(written1, written2);
	}
	virtual int overflow (int c = EOF)
	{
		if (c != EOF) {
			int result1 = sb1->sputc(c);
			int result2 = sb2->sputc(c);
			if (result1 == result2)
				return result1;
		}

		return EOF;
	}

private:
	std::streambuf *sb1;
	std::streambuf *sb2;
};


class Tee: public std::ostream
{
public:
	Tee (std::ostream &o1, std::ostream &o2) :
	std::ostream(&tbuf),
	tbuf(o1.rdbuf(), o2.rdbuf())
	{}

private:
	TeeBuffer tbuf;

};

#endif /* TEE_H */


/* Test Tee */
#if __INCLUDE_LEVEL__ == 0 && defined __INCLUDE_LEVEL__
	#include <fstream>		// std::ofstream
	#include <iostream>		// std::cout, std::endl;

using namespace std;

int main()
{
    ofstream log("out/test_tee.txt");
    Tee tee(cout, log);
    tee << "Hello, world!\n";
    return 0;
}
#endif
/* Test Tee */
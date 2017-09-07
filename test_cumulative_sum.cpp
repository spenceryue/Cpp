#ifndef CUMULATIVE_SUM_H
#define CUMULATIVE_SUM_H
#include <iostream> 			// std::cout, std::endl
#include <string> 				// std::string
#include <algorithm>			// std::transform
#include <type_traits>			// std::enable_if_t, std::is_unsigned_v
#include "range.h"

unsigned long long cusum(unsigned long long N)
{
	return (N/2.0) * (N + 1);
}

template <class T = unsigned int, class = std::enable_if_t<std::is_unsigned_v<T>>>
unsigned int ilog2(T n)
{
	unsigned int i=sizeof(T) * 8;
	while((n & (T(1)<<--i)) == 0);

	return i;
}

template <class T = unsigned int, class = std::enable_if_t<std::is_unsigned_v<T>>>
std::string to_binary(T N)
{
	using namespace std;
	auto bits = ilog2(N) + 1;
	auto r = range(bits);
	string result(bits, '0');
	transform(r.begin(), r.end(), rbegin(result), [i=0,N] (char c) mutable
	{
		return bool(N & (T(1) << i++)) + '0';
	});

	return "0b" + result;
}

template <int repeat = 1>
void test_it(unsigned long long N)
{
	using namespace std;
	cout << "Input: " << (void*)N << " (" << to_binary(N) << ") " << endl;
	for (int i=0; i<repeat; i++) {
		N = cusum(N);
		cout << "Output: " << (void*)N << " (" << N << ") " << endl;
	}
	cout << endl;
}
#endif /* CUMULATIVE_SUM_H */



/* Test cumulative_sum */
#if __INCLUDE_LEVEL__ == 0 && defined __INCLUDE_LEVEL__
	#include <iostream> 		// std::cout, std::endl
	#include <string> 			// std::string
	#include "faces.h"

using namespace std;
int main(int argc, char* argv[])
{
	cout << pikachu << "\n" << endl;
	// cout << ilog2(16) << endl;
	test_it(31145914794);
	test_it(31);
	test_it(15);
	test_it(14);
	test_it(7);
	test_it(3);
	return 0;
}
#endif
/* Test cumulative_sum */
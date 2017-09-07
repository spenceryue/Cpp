/*input
5
*/

#include <iostream>			// std::cout, std::cerr, std::endl, std::ostream
#include <fstream>			// std::ofstream;
#include "tee.h"			// Tee
#include <string>			// std::string
#include <vector>			// std::vector
#include <iterator>			// std::ostream_iterator
#include <cmath>			// pow

using std::cout;
using std::cerr;
using std::endl;
using std::ofstream;
using std::ostream;
using std::vector;
using std::string;
using std::ostream_iterator;


vector<unsigned int> make_plan(ostream&, string, unsigned int, unsigned int);
template<unsigned int precision = 4>
double nth_rt(int, double);

int main() {
	ofstream output("out/planning_sketch_out.txt");
	Tee tee(cout, output);

	unsigned int N=5*11*32*32;
	// cout << "Enter a test value: ";
	// cin >> N;
	tee << "N = " << N << endl;

	tee << endl << endl;
	vector<unsigned int> min_plan = make_plan(tee, "Min_plan",N,1024);
	vector<unsigned int> max_plan = make_plan(tee, "Max_plan",N,32);

	vector<unsigned int> roots;
	vector<vector<unsigned int>> plans;
	for (unsigned int i=min_plan.size(); i<max_plan.size(); i++) {
		double value = nth_rt(i,N);
		unsigned int rounded = static_cast<unsigned int>(ceil(value));
		roots.push_back(rounded);
		tee << i << "-th root(" << N << ") = " << value << " (" << rounded << " rounded up)" << endl;
		
		plans.push_back(make_plan(tee, "Test_plan_" + std::to_string(i-min_plan.size()), N, rounded));
	}

	for (int i=1; i<=4; i++) {
		unsigned int factor = pow(2,i) * 32;
		tee << "2^" << i << " = " << factor << endl;
		make_plan(tee, "2^" + std::to_string(i) + " plan", N, factor);
	}
}

vector<unsigned int> make_plan(ostream& out, string name, unsigned int N, unsigned int factor) {
	vector<unsigned int> plan;
	unsigned int n = (N + factor - 1) / factor;
	out << "\t" << "(N..." << N << ")" << endl;
	while (n > 1) {
		out << "\t" << "n..." << n << endl;
		plan.push_back(n);
		n = (n + factor - 1)/factor;
	}
	plan.push_back(1);
	out << "\t" << "n..." << n << endl;

	out << name << " = [ ";
	copy(plan.begin(), plan.end(), ostream_iterator<unsigned int>(out, " "));
	out << "]" << endl;

	unsigned int min_len = plan.size();
	out << "length = " << min_len << endl;
	out << endl << endl;

	return plan;
}

template<unsigned int precision = 4>
double nth_rt(int n, double x) {
	if (x < 0 && n%2 == 0) {
		cerr << "x cannot be negative when n is even." << endl;
		return -1;
	}
	if (n == 0) {
		cerr << "n must be nonzero." << endl;
		return -1;
	}
	if (n == 1)
		return x;
	if (n == 2)
		return sqrt(x);
	if (n == 3)
		return cbrt(x);

	bool reciprocate = false;
	if (n < 0) {
		reciprocate = true;
		n *= -1;
	}

	if (n == -1)
		return 1/x;
	if (n == -2)
		return 1/sqrt(x);
	if (n == -3)
		return 1/cbrt(x);

	double x1 = x,
	x2 = x / n;

	double eps = pow(10, -precision);
	while (abs(x1 - x2) > eps) {
		x1 = x2;
		x2 = ((n-1) * x2 + x / pow(x2, n-1)) / n;
	}

	if (reciprocate)
		return 1/x2;

	return x2;
}
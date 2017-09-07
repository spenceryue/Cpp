/* Brainstorming list comprehension possilibities with C++ */

int main() {

	

	vector<int> squares = for_range(10, [] (index_sequence<auto ...i>) {return {(i*i, ...)};} (make_index_sequence<10>{});
	vector<int> squares = list_comp([] (auto i) {return i*i;}, range(10));
	squares = vect{ [](auto& i) {return i*i;}, range(10) };
	squares = vect{ [](auto& i) {return i*i;}, range(10), [](auto& i) {return i%2;} };
	squares = vect{ [](auto& i) {return vect{ [](auto& j) {return {i,j};}, range(i)};}, range(10) };
	squares = vect{ [](auto& i) {return vect{ [](auto& j) {return {i,j};}, range(i)};}, range(10) };
	
	// python:
	squares = [i*i for i in range(10)];
	squares = [i*i for i in range(10) if i%2]
	squares = [(i,j) for i in range(10) for j in range(5)]
	
	// possible c++?:
	auto squares l1st{[](auto i) {return i*i;}, range(10)};
	auto squares l1st{[](auto i) {return i*i;}, range(10), [](auto i){return i%2;}};
	auto squares l1st{[](auto i, auto j) {return tuple{i,j};}, range(10), range(5)};
	vector<int> squares = for_each_n(range(10), 10, [] (auto i) {return i*i;});
	auto squares = list_{[](auto& i, auto& j) {return tuple_{i,j};}, range(10), range(5)};
	
}
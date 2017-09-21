#ifndef ZIP_H
#define ZIP_H
#include <iterator> 			// std::input_iterator_tag, begin, end
#include <type_traits> 			// std::enable_if_t, remove_reference_t, is_rvalue_reference_v
#include <tuple>				// std::tuple, apply
#include <utility>				// std::forward, index_sequence, index_sequence_for
#include "report_errors.h"
#include "wrap_references.h"

#include "type_stuff.h"

namespace zip_ns {
	constexpr bool VERBOSE = 1;
}

template <int HAS_TEMPS = 0, class LOOOOOOK = void, class ...Iterables>
class zip
{
	template <class ...BeginIterators>
	struct Begin {
		using Pack = std::tuple<BeginIterators...>;
		using Values = std::tuple<typename BeginIterators::value_type...>;
	};
	using B = Begin<std::remove_reference_t<decltype(std::begin(std::declval<Iterables&>()))>...>;
	using BeginPack = typename B::Pack;
	using Values = typename B::Values;

	template <class ...EndIterators>
	struct End {
		using Pack = std::tuple<EndIterators...>;
	};
	using E = End<std::remove_reference_t<decltype(std::end(std::declval<Iterables&>()))>...>;
	using EndPack = typename E::Pack;

	static constexpr std::index_sequence_for<Iterables...> indices{};
	static constexpr bool has_temporaries = (std::is_rvalue_reference_v<Iterables&&> || ...);

	std::conditional_t<has_temporaries, std::tuple<std::remove_reference_t<decltype(wrap_references(std::declval<Iterables>()))>...>, bool> persist;
	// std::conditional_t<has_temporaries, std::tuple<Iterables...>, bool> persist;
	BeginPack current;
	const EndPack stop;

	template <class ...I, size_t ...Indices>
	zip (std::index_sequence<Indices...>, I&&... i)
	try : persist{wrap_references(std::forward<I>(i))...},
		  current{std::begin(unwrap_references(std::get<Indices>(persist)))...},
		  stop{std::end(unwrap_references(std::get<Indices>(persist)))...} {
	/*try : persist{std::forward<I>(i)...},
		  current{std::begin(std::get<Indices>(persist))...},
		  stop{std::end(std::get<Indices>(persist))...} {*/
		if constexpr(zip_ns::VERBOSE) {
			std::cout << std::endl << blank_face " I see we have ourselves some temporaries." << std::endl;
			std::cout << *this << "\t\tline: " << __LINE__ << std::endl;
			std::cout << "current type:\n\t" << type_name<decltype(current)>() << std::endl << std::endl;
			std::cout << "persist type:\n\t" << type_name<decltype(persist)>() << std::endl << std::endl;
			std::cout << "Temporaries:\n\t" << persist << std::endl;
		}
	}
	catch (...)
	{
		throw_err("Make sure to give zip() arguments that are iterable (through calls to std::begin(), std::end()).");
	}

	template <size_t... i>
	bool proceed(std::index_sequence<i...>)
	const {
		return (... && (std::get<i>(current) != std::get<i>(stop)));
	}

public:
	using iterator_category = std::input_iterator_tag;
	using value_type = Values;
	using difference = void;
	using pointer = Values*;
	using reference = Values&;

	Values operator* () const {
		return std::apply([](auto&&... i) {return std::tuple{*i...};}, current);
	}

	auto& operator++ () {
		std::apply([](auto&&... i) {(++i, ...);}, current);
		return *this;
	}

	auto operator++ (int) {
		zip copy(*this);
		std::apply([](auto&&... i) {(++i, ...);}, current);
		return copy;
	}

	bool operator!= (const zip&) const {
		return proceed(indices);
	}

	bool operator== (const zip&) const {
		return !proceed(indices);
	}

	template <class ...I, std::enable_if_t<(std::is_rvalue_reference_v<I&&> || ...), int> =0>
	explicit zip (I&&... i) :		// Explicit prevents candidate interpretation as copy constructor during class template resolution
	zip(std::index_sequence_for<I...>{}, std::forward<I>(i)...) {}

	template <class ...I, std::enable_if_t<(!std::is_rvalue_reference_v<I&&> && ...), int> =0>
	explicit zip (I&&... i)		// Explicit prevents candidate interpretation as copy constructor during class template resolution
	try : persist{0}, current{std::begin(i)...}, stop{std::end(i)...} {
		if constexpr(zip_ns::VERBOSE) {
			std::cout << std::endl << stare " No temporaries here! Yay" << std::endl;
			std::cout << *this << "\t\tline: " << __LINE__ << std::endl;
			std::cout << "current type:\n\t" << type_name<decltype(current)>() << std::endl << std::endl;
			std::cout << "persist type:\n\t" << type_name<decltype(persist)>() << std::endl << std::endl;
		}
	}
	catch (...)
	{
		throw_err("Make sure to give zip() arguments that are iterable (through calls to std::begin(), std::end()).");
	}

	auto&& begin () {
		return std::move(*this);
	}

	auto& end () {
		return *this;
	}

	zip (const zip& copy) :
	persist(copy.persist), current(copy.current), stop(copy.stop) {
		if constexpr(zip_ns::VERBOSE)
			std::cout << "zip COPIED!" << *this << std::endl << std::endl;
	}

	zip (zip&& copy) :
	persist(std::move(copy.persist)), current(std::move(copy.current)), stop(copy.stop) {
		if constexpr(zip_ns::VERBOSE)
			std::cout << "zip MOVED!" << *this << std::endl << std::endl;
	}

	/* Pretty print */
	friend std::ostream& operator<< (std::ostream& output, const zip&) {
		return output << type_name<zip>();
	}
};

/* Deduction guide */
template <int HAS_TEMPS = 2, class LOOOOOOK = void, class ...Iterables, class ...I, std::enable_if_t<(std::is_rvalue_reference_v<I&&> || ...), int> =0>
explicit zip (I&&... i) -> zip<HAS_TEMPS, std::tuple<std::remove_reference_t<decltype(wrap_references(std::declval<I>()))>...>, I...>;

template <int HAS_TEMPS = 0, class LOOOOOOK = void, class ...Iterables, class ...I, std::enable_if_t<(!std::is_rvalue_reference_v<I&&> && ...), int> =0>
explicit zip (I&&... i) -> zip<HAS_TEMPS, std::tuple<std::remove_reference_t<decltype(wrap_references(std::declval<I>()))>...>, I...>;

#endif /* ZIP_H */



/* Test zip */
#if __INCLUDE_LEVEL__ == 0 && defined __INCLUDE_LEVEL__
	#include <iostream> 		// std::cout, std::endl
	#include <string> 			// std::string
	#include <tuple>			// std::tuple
	#include "faces.h"
	#include "type_stuff.h"
	#include "range.h"
	#include "basename.h"
	#include "print_tuple.h"
	#include "get_random.h"

	#include "enumerate.h"
using namespace std;
using namespace string_literals;
int main(int argc, char* argv[]) try
{
	startup_msg(argv[0]);
	cout << pikachu << endl;
	cout << "\n" << endl;

	// (void)zip(range(0),zip(range(1),range(2)));
	// (void)zip(zip("hello"s,"world"s, to_string(__LINE__)));
	// auto a = zip("hi"s);
	auto a = zip(range(0));
	/*
	auto b = string("hiii");
	auto c = range(1);
	cout << type_name<std::remove_reference_t<decltype(std::begin(std::declval<decltype(a)&>()))>>() << endl;
	cout << type_name<std::remove_reference_t<decltype(std::begin(a))>>() << endl;
	cout << type_name<std::remove_reference_t<decltype(zip(c))>>() << endl;*/
	// (void)zip(zip("hello"s,"world"s, a, to_string(__LINE__)));
	// (void) zip(a, to_string(__LINE__));
	// (void) zip(a);
	/*
	auto range10 = range(10);
	// (void) zip(range10, zip(range10, range10));
	// (void)zip(range10,zip("hello"s,"world"s, to_string(__LINE__)));
	// (void)zip(range(10),zip("hello"s,"world"s, to_string(__LINE__)));
	cout << type_name<decltype(zip(zip("hello"s,"world"s, to_string(__LINE__))))>() << endl;


	string a = "hello", b = "world", ab = a+b, c;

	auto R = range<true,int>(10);
	auto RR = range(5);
	[[maybe_unused]] auto z2 = zip{R, RR};
	[[maybe_unused]] auto z3 = zip{range<true,int>(10), RR};
	[[maybe_unused]] auto z31 = zip{range<true,int>(10), range(5)};
	[[maybe_unused]] auto z32 = zip{range<true,int>(10)};
	[[maybe_unused]] auto z33 = zip{RR};
	(void) zip{range<true,int>(10), RR};
	cout << endl << endl << endl;
	[[maybe_unused]] auto z34 = zip(range<true,int>(10), RR);
	(void) zip(range<true,int>(10), RR);
	[[maybe_unused]] auto z4 = zip{ab, c=to_string(__LINE__)};
	[[maybe_unused]] auto z333 = zip(a+b, c=to_string(__LINE__));
	[[maybe_unused]] auto [i0, i1] = *zip(a+b, c=to_string(__LINE__));
	[[maybe_unused]] auto zz = zip{a, b, a, b, a, c=to_string(__LINE__)};

	cout << i0 << '\t' << type_name<decltype(i0)>() << endl;
	cout << i1 << '\t' << type_name<decltype(i1)>() << endl;
	cout << "\n" << endl;

	cout << type_name<decltype(zip{"hello"s,"world"s, to_string(__LINE__)})>() << endl;
	cout << type_name<decltype(zip("hello"s,"world"s, to_string(__LINE__)))>() << endl;
	cout << type_name<decltype(++zip("hello"s,"world"s, to_string(__LINE__)))>() << endl;
	cout << type_name<decltype(zip("hello"s,"world"s, to_string(__LINE__)).begin())>() << endl;
	cout << type_name<decltype(*zip{"hello"s,"world"s, to_string(__LINE__)})>() << endl;
	cout << type_name<decltype(*zip("hello"s,"world"s, to_string(__LINE__)))>() << endl;
	cout << type_name<decltype("hello"s)>() << endl;
	cout << type_name<decltype(zip(zip("hello"s,"world"s, to_string(__LINE__))))>() << endl;
	// cout << type_name<decltype(enumerate(zip("hello"s,"world"s, to_string(__LINE__))))>() << endl; // requires move/copy constructor
	cout << "\n" << endl;

	cout << type_name<decltype(range('a','a' + 10u))>() << endl;
	cout << "\n" << endl;

	for (auto [i, j, ignore] : zip{range(0,24,2), range('a','a' + 11u), to_string(__LINE__) + "                "s}) {
		(void)ignore;
		cout << "type of i: " << type_name<decltype(i)>() << endl;
		cout << "type of j: " << type_name<decltype(j)>() << endl;
		cout << i << ", " << (char) j << endl;
	}
	cout << "\n" << endl;

	for (tuple<int,char,char>&& t : zip{range(0,24,2), range('a','a' + 11u), to_string(__LINE__) + "                  "s})
		cout << t << endl;*/

	return 0;
} catch (...) {cout << blank_face << endl;return 1;}
#endif
/* Test zip */
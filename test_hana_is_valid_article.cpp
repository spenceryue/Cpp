#include <iostream>
#include <type_traits>

struct A {};

std::string to_string(const A&)
{
    return "I am a A!";
}

// Type B with a serialize method.
struct B
{
    std::string serialize() const
    {
        return "I am a B!";
    }
};

// Type C with a "wrong" serialize member (not a method) and a to_string overload.
struct C
{
    std::string serialize;
};

std::string to_string(const C&)
{
    return "I am a C!";
}

struct D : A
{
    std::string serialize() const
    {
        return "I am a D!";
    }
};

struct E
{
    struct Functor
    {
        std::string operator()() {
            return "I am a E!";
        }
    };

    Functor serialize;
};



template <typename UnnamedType> struct container
{
// Let's put the test in private.
private:
    // We use std::declval to 'recreate' an object of 'UnnamedType'.
    // We use std::declval to also 'recreate' an object of type 'Param'.
    // We can use both of these recreated objects to test the validity!
    template <typename Param> constexpr auto test_validity(int /* unused */)
    -> decltype(std::declval<UnnamedType>()(std::declval<Param>()), std::true_type())
    {
        // If substitution didn't fail, we can return a true_type.
        return std::true_type();
    }

    template <typename Param> constexpr std::false_type test_validity(...)
    {
        // Our sink-hole returns a false_type.
        return std::false_type();
    }

public:
    // A public operator() that accept the argument we wish to test onto the UnnamedType.
    // Notice that the return type is automatic!
    template <typename Param> constexpr auto operator()(const Param& p)
    {
        // The argument is forwarded to one of the two overloads.
        // The SFINAE on the 'true_type' will come into play to dispatch.
        return test_validity<Param>(int());
    }
};

template <typename UnnamedType> constexpr auto is_valid(const UnnamedType& t)
{
    return container<UnnamedType>();
}

// Check if a type has a serialize method.
// auto hasSerialize = is_valid([](auto&& x) -> decltype(x.serialize()) { });
struct lamby
{
	template <class T>
	auto operator()(T&& x) -> decltype(x.serialize());
} my_lamby;
auto hasSerialize = is_valid(my_lamby);

template <class T> auto serialize(T& obj) -> typename std::enable_if<decltype(hasSerialize(obj))::value, std::string>::type
{
    return obj.serialize();
}

template <class T> auto serialize(T& obj) -> typename std::enable_if<!decltype(hasSerialize(obj))::value, std::string>::type
{
    return to_string(obj);
}

int main() {
    A a;
    B b;
    C c;
    D d;
    E e;

    std::cout << serialize(a) << std::endl;
    std::cout << serialize(b) << std::endl;
    std::cout << serialize(c) << std::endl;
    std::cout << serialize(d) << std::endl;
    std::cout << serialize(e) << std::endl;

    return 0;
}
//  (C) Copyright John Maddock 2001.
//  (C) Copyright Jens Maurer 2001.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//  See http://www.boost.org for most recent version.

//  config for libstdc++ v3
//  not much to go in here:

#ifdef __GLIBCXX__
#define BOOST_STDLIB "GNU libstdc++ version " BOOST_STRINGIZE(__GLIBCXX__)
#else
#define BOOST_STDLIB "GNU libstdc++ version " BOOST_STRINGIZE(__GLIBCPP__)
#endif

#if !defined(_GLIBCPP_USE_WCHAR_T) && !defined(_GLIBCXX_USE_WCHAR_T)
#define BOOST_NO_CWCHAR
#define BOOST_NO_CWCTYPE
#define BOOST_NO_STD_WSTRING
#define BOOST_NO_STD_WSTREAMBUF
#endif

#if defined(__osf__) && !defined(_REENTRANT) \
    && (defined(_GLIBCXX_HAVE_GTHR_DEFAULT) || defined(_GLIBCPP_HAVE_GTHR_DEFAULT))
// GCC 3 on Tru64 forces the definition of _REENTRANT when any std lib header
// file is included, therefore for consistency we define it here as well.
#define _REENTRANT
#endif

#ifdef __GLIBCXX__ // gcc 3.4 and greater:
#if defined(_GLIBCXX_HAVE_GTHR_DEFAULT) || defined(_GLIBCXX__PTHREADS)
//
// If the std lib has thread support turned on, then turn it on in Boost
// as well.  We do this because some gcc-3.4 std lib headers define _REENTANT
// while others do not...
//
#define BOOST_HAS_THREADS
#else
#define BOOST_DISABLE_THREADS
#endif
#elif defined(__GLIBCPP__) && !defined(_GLIBCPP_HAVE_GTHR_DEFAULT) && !defined(_GLIBCPP__PTHREADS)
// disable thread support if the std lib was built single threaded:
#define BOOST_DISABLE_THREADS
#endif

#if (defined(linux) || defined(__linux) || defined(__linux__)) && defined(__arm__) \
    && defined(_GLIBCPP_HAVE_GTHR_DEFAULT)
// linux on arm apparently doesn't define _REENTRANT
// so just turn on threading support whenever the std lib is thread safe:
#define BOOST_HAS_THREADS
#endif


#if !defined(_GLIBCPP_USE_LONG_LONG) && !defined(_GLIBCXX_USE_LONG_LONG) && defined(BOOST_HAS_LONG_LONG)
// May have been set by compiler/*.hpp, but "long long" without library
// support is useless.
#undef BOOST_HAS_LONG_LONG
#endif

#if defined(__GLIBCXX__) || (defined(__GLIBCPP__) && __GLIBCPP__ >= 20020514) // GCC >= 3.1.0
#define BOOST_STD_EXTENSION_NAMESPACE __gnu_cxx
#define BOOST_HAS_SLIST
#define BOOST_HAS_HASH
#define BOOST_SLIST_HEADER <ext/slist>
#if !defined(__GNUC__) || __GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 3)
#define BOOST_HASH_SET_HEADER <ext/hash_set>
#define BOOST_HASH_MAP_HEADER <ext/hash_map>
#else
#define BOOST_HASH_SET_HEADER <backward/hash_set>
#define BOOST_HASH_MAP_HEADER <backward/hash_map>
#endif
#endif

//  stdlibc++ C++0x support is detected via __GNUC__, __GNUC_MINOR__, and possibly
//  __GNUC_PATCHLEVEL__ at the suggestion of Jonathan Wakely, one of the stdlibc++
//  developers. He also commented:
//
//       "I'm not sure how useful __GLIBCXX__ is for your purposes, for instance in
//       GCC 4.2.4 it is set to 20080519 but in GCC 4.3.0 it is set to 20080305.
//       Although 4.3.0 was released earlier than 4.2.4, it has better C++0x support
//       than any release in the 4.2 series."
//
//  Another resource for understanding stdlibc++ features is:
//  http://gcc.gnu.org/onlinedocs/libstdc++/manual/status.html#manual.intro.status.standard.200x

//  C++0x headers in GCC 4.3.0 and later
//
#if __GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 3) || !defined(__GXX_EXPERIMENTAL_CXX0X__)
#define BOOST_NO_0X_HDR_ARRAY
#define BOOST_NO_0X_HDR_RANDOM
#define BOOST_NO_0X_HDR_REGEX
#define BOOST_NO_0X_HDR_TUPLE
#define BOOST_NO_0X_HDR_TYPE_TRAITS
#define BOOST_NO_STD_UNORDERED // deprecated; see following
#define BOOST_NO_0X_HDR_UNORDERED_MAP
#define BOOST_NO_0X_HDR_UNORDERED_SET
#endif

//  C++0x headers in GCC 4.4.0 and later
//
#if __GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 4) || !defined(__GXX_EXPERIMENTAL_CXX0X__)
#define BOOST_NO_0X_HDR_CHRONO
#define BOOST_NO_0X_HDR_CONDITION_VARIABLE
#define BOOST_NO_0X_HDR_FORWARD_LIST
#define BOOST_NO_0X_HDR_INITIALIZER_LIST
#define BOOST_NO_0X_HDR_MUTEX
#define BOOST_NO_0X_HDR_RATIO
#define BOOST_NO_0X_HDR_SYSTEM_ERROR
#define BOOST_NO_0X_HDR_THREAD
#endif

//  C++0x headers not yet implemented
//
#define BOOST_NO_0X_HDR_CODECVT
#define BOOST_NO_0X_HDR_CONCEPTS
#define BOOST_NO_0X_HDR_CONTAINER_CONCEPTS
#define BOOST_NO_0X_HDR_FUTURE
#define BOOST_NO_0X_HDR_ITERATOR_CONCEPTS
#define BOOST_NO_0X_HDR_MEMORY_CONCEPTS

//  --- end ---

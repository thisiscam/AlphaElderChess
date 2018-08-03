#ifndef THREADING_HPP
#define THREADING_HPP

namespace threading {

class SpinLock {
    std::atomic_flag locked = ATOMIC_FLAG_INIT ;
public:
    inline void lock() {
        while (locked.test_and_set(std::memory_order_acquire)) { ; }
    }
    inline void unlock() {
        locked.clear(std::memory_order_release);
    }
};

template<class T>
void atomic_add(std::atomic<T> &f, T d) {
    T old = f.load();
    while (!f.compare_exchange_weak(old, old + d));
}

}

#include "ThreadPool.h"

#endif
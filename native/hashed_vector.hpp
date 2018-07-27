#include <vector>
#include <unordered_map>
#include <stdlib.h>
#include <cassert>

template <typename T> 
class hashed_vector {
    std::vector<T> A;
    std::unordered_map<T, int> H;
    std::unordered_map<T, int> C;

public:
    
    inline operator std::vector<T> () const {
        return A;
    }

    inline void insert(const T& v) {
        assert(A.size() == H.size() && A.size() == C.size());
        if(contains(v)) {
            return;
        }
        H.insert({v, A.size()});
        C.insert({v, 1});
        A.push_back(v);
    }

    inline void dup_insert(const T& v) {
        assert(A.size() == H.size() && A.size() == C.size());
        if(contains(v)) {
            C[v] += 1;
            return;
        }
        H.insert({v, A.size()});
        C.insert({v, 1});
        A.push_back(v);
    }

    inline int erase(const T& v) {
        assert(A.size() == H.size());
        auto it = H.find(v);
        if(it == H.end()) {
            return 0;
        } else if (C[v] > 1) {
            C[v] -= 1;
            return 1;
        } else {
            const T& last = A.back();
            int i = it->second;
            A[i] = last;
            H[last] = i;
            A.pop_back();
            H.erase(it);
            C.erase(v);
            assert(A.size() == H.size() && A.size() == C.size());
            return 1;
        }
    }

    inline int size() const {
        assert(A.size() == H.size() && A.size() == C.size());
        return A.size();
    }

    inline bool contains(const T& v) const {
        return H.find(v) != H.end();
    }

    template<typename W, typename RandonEngine>
    inline const T& getRandom(W&& wf, RandonEngine&& engine) const {
        assert(A.size() > 0);
        assert(A.size() == H.size());
        int sum_of_weight = 0;
        std::vector<int> weights;
        for(int i=0; i < A.size(); i++) {
            int wi = wf(A[i]) * C.at(A[i]);
            weights.push_back(wi);
            sum_of_weight += wi;
        }
        std::uniform_int_distribution<int> rnds(0, sum_of_weight - 1);
        int rnd = rnds(engine);
        for(int i=0; i < A.size(); i++) {
            if(rnd < weights[i]) {
                return A[i];
            }
            rnd -= weights[i];
        }
        assert(!"should never get here");
    }

    inline const T& operator[](const size_t idx) const {
        assert(A.size() == H.size());
        return A[idx];
    }
};
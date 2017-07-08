#pragma once

#include <algorithm>
#include <cassert>
#include <iterator>
#include <utility>
#include <vector>

template <typename T, typename Comparator = std::less<T>>
class PriorityQueue {
public:
    PriorityQueue(const Comparator &compare = Comparator()) : m_compare(compare) {}

    const T &top() const { return m_heap.front(); }
    bool     empty() const { return m_heap.empty(); }
    auto     size() const { return m_heap.size(); }

    void push(const T &value) {
        m_heap.push_back(value);
        std::push_heap(m_heap.begin(), m_heap.end(), m_compare);
    }

    template <typename... Args>
    void emplace(Args &&... args) {
        m_heap.emplace_back(std::forward<Args>(args)...);
        std::push_heap(m_heap.begin(), m_heap.end(), m_compare);
    }

    void pop() {
        std::pop_heap(m_heap.begin(), m_heap.end(), m_compare);
        m_heap.pop_back();
    }

    // --- This member functions don't exist in std::priority_queue. ----------

    // Access heap value by index.
    const T &operator[](int index) {
        assert(index >= 0 && index < (int) m_heap.size());
        return m_heap[index];
    }

    // Return index of a value for which predicate p returns true.
    template <typename UnaryPredicate>
    int find(UnaryPredicate p) {
        // This a basicly a breadth-first search, although not quite optimal. We could cancel the
        // search earlier if we'd give up a bit of generality here.
        auto it = std::find_if(m_heap.begin(), m_heap.end(), p);

        return std::distance(m_heap.begin(), it);
    }

    void update(int index, T newValue) {
        assert(index >= 0 && index < (int) m_heap.size());

        // Only moving up the heap is supported!
        // assert(???)

        m_heap[index] = std::move(newValue);

        // Move new value up the heap
        while (index > 0) {
            const int parent = (index - 1) / 2;

            if (m_compare(m_heap[parent], m_heap[index])) {
                std::swap(m_heap[parent], m_heap[index]);
                index = parent;
            } else
                break;
        }

        // done!
        assert(std::is_heap(m_heap.begin(), m_heap.end(), m_compare));
    }

private:
    std::vector<T> m_heap;
    Comparator     m_compare;
};
#pragma once

#include <catboost/libs/data/pool.h>

template <typename E> class ILeafValuesProvider {
public:
    virtual const TVector<E> GetValuesForTreeIdxAndIterationNum(size_t treeIdx, size_t it) const = 0;
};

template <typename E> class TVector3Wrapper : public ILeafValuesProvider<E> {
public:
    const TVector<TVector<TVector<E>>> Vector;

    explicit TVector3Wrapper(const TVector<TVector<TVector<E>>>&& vectorRref)
            : Vector(std::move(vectorRref))
    {

    }

    const TVector<E> GetValuesForTreeIdxAndIterationNum(size_t treeIdx, size_t it) const override {
        return Vector[treeIdx][it];
    }
};

class ILeafIdxsProvider {
public:
    virtual const TVector<ui32> GetLeafIdxsForTreeIdx(size_t treeIdx) const = 0;
};
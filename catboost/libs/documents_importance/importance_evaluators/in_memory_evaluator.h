//
// Created by Boris Sharchilev on 10/07/2018.
//
#pragma once

#include "importance_evaluator.h"
#include "importance_evaluator_factory.h"

class TInMemoryDocumentImportancesEvaluator : public IDocumentImportancesEvaluator {
public:
    static std::unique_ptr<TInMemoryDocumentImportancesEvaluator> Construct(
            const TFullModel& model,
            const TPool& trainPool,
            const TInfluenceTarget& influenceTarget,
            const TUpdateMethod& updateMethod,
            const TTreesLimits& differentiatedTreesLimits,
            int threadCount);

protected:
    const TTreeStatisticsVectorWrapper TreeStatisticsVector;

    TInMemoryDocumentImportancesEvaluator(
            const TFullModel& model,
            const TTreeStatisticsVectorWrapper treeStatisticsVector,
            const TUpdateMethod& updateMethod,
            const TTreesLimits& differentiatedTreesLimits,
            TInfluenceTarget influenceTarget,
            const TLeafFormulaParams& leafFormulaParams,
            size_t treeCount,
            size_t docCount,
            int threadCount)
            : IDocumentImportancesEvaluator(
                    model,
                    updateMethod,
                    differentiatedTreesLimits,
                    std::move(influenceTarget),
                    leafFormulaParams,
                    treeCount,
                    docCount,
                    threadCount),
              TreeStatisticsVector(std::move(treeStatisticsVector))
    {

    }

    const ILeafValuesProvider<double>& GetLeafValuesProvider() const override {
        return TreeStatisticsVector;
    }

    const ILeafIdxsProvider& GetLeafIdxsProvider() const override {
        return TreeStatisticsVector;
    }

    TVector3Wrapper<double> GetLeavesGradientsWrtWeight(size_t thisDocIdx) const override;
    TVector<double> GetLeavesGradientsWrtWeightForTreeAndIteration(
            size_t thisDocIdx,
            size_t treeId,
            size_t leavesEstimationIteration,
            const TVector<ui32> &leafIdxsToUpdate,
            const TVector<double> &jacobian
    ) const;
};
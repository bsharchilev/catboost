#pragma once

#include <catboost/libs/model/model.h>
#include <catboost/libs/data/pool.h>
#include <catboost/libs/documents_importance/docs_importance.h>
#include <catboost/libs/documents_importance/ders_helpers.h>
#include <catboost/libs/documents_importance/docs_importance_helpers.h>
#include <catboost/libs/documents_importance/tree_statistics.h>

class IDocumentImportancesEvaluator {
public:
    virtual ~IDocumentImportancesEvaluator() = default;

    TVector<TVector<double>> GetDocumentImportances(const TPool& testPool) {
        NPar::TLocalExecutor localExecutor;
        localExecutor.RunAdditionalThreads(ThreadCount - 1);

        TVector<TVector<ui32>> leafIdxs = GetLeafIdxsForPool(testPool, &localExecutor);
        TVector<double> lossGradientsWrtPredictions;
        if (!InfluenceTarget.IsPredictionInfluenceTarget) {
            // TODO(bshar): this place can be rewritten to support pairwise test losses
            const TVector<double>& predictions = ApplyModel(leafIdxs, GetLeafValuesProvider());
            lossGradientsWrtPredictions.resize(testPool.Docs.GetDocCount());
            EvaluateDerivatives(
                    InfluenceTarget.LossDescription,
                    predictions,
                    testPool,
                    &lossGradientsWrtPredictions,
                    nullptr,
                    nullptr
            );
        }

        TVector<TVector<double>> result;
        result.resize(DocCount);
        localExecutor.ExecRange(
                [&] (size_t docIdx) {
                    result[docIdx] = GetDocumentImportancesForOneTrainDoc(
                            GetLeavesGradientsWrtWeight(docIdx),
                            leafIdxs,
                            lossGradientsWrtPredictions
                    );
                },
                NPar::TLocalExecutor::TExecRangeParams(0, DocCount),
                NPar::TLocalExecutor::WAIT_COMPLETE
        );
        return result;
    }

protected:
    const TFullModel& Model;
    const TUpdateMethod UpdateMethod;
    const TTreesLimits DifferentiatedTreesLimits;
    const TInfluenceTarget InfluenceTarget;
    const TLeafFormulaParams LeafFormulaParams;
    const size_t TreeCount;
    const size_t DocCount;
    const int ThreadCount;

    IDocumentImportancesEvaluator(
            const TFullModel& model,
            const TUpdateMethod& updateMethod,
            const TTreesLimits& differentiatedTreesLimits,
            TInfluenceTarget influenceTarget,
            const TLeafFormulaParams& leafFormulaParams,
            size_t treeCount,
            size_t docCount,
            int threadCount)
            : Model(model),
              UpdateMethod(updateMethod),
              DifferentiatedTreesLimits(differentiatedTreesLimits),
              InfluenceTarget(std::move(influenceTarget)),
              LeafFormulaParams(leafFormulaParams),
              TreeCount(treeCount),
              DocCount(docCount),
              ThreadCount(threadCount)
    {

    }

    virtual const ILeafValuesProvider<double>& GetLeafValuesProvider() const = 0;
    virtual const ILeafIdxsProvider& GetLeafIdxsProvider() const = 0;
    virtual TVector3Wrapper<double> GetLeavesGradientsWrtWeight(size_t thisDocIdx) const = 0;

    TVector<double> ApplyModel(const TVector<TVector<ui32>>& leafIdxs, const ILeafValuesProvider<double>& leafValuesProvider) const;

    TVector<TVector<ui32>> GetLeafIdxsForPool(const TPool& testPool, NPar::TLocalExecutor* localExecutor) const; // [treeIdx][docIdx]
    TVector<ui32> GetLeafIdxsToUpdate(size_t treeId, const TVector<double>& jacobian) const;

    TVector<double> GetDocumentImportancesForOneTrainDoc(
            const TVector3Wrapper<double>& leafGradients,
            const TVector<TVector<ui32>>& leafIndices,
            const TVector<double>& lossGradientsWrtPredictions
    ) const;
};

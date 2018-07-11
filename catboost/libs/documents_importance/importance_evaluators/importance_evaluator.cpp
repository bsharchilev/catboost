#include <contrib/libs/cxxsupp/libcxx/include/iostream>
#include "importance_evaluator.h"

TVector<TVector<ui32>> IDocumentImportancesEvaluator::GetLeafIdxsForPool(
        const TPool& testPool,
        NPar::TLocalExecutor* localExecutor) const
{
    TVector<TVector<ui32>> leafIdxs(TreeCount);
    const TVector<ui8> binarizedFeatures = BinarizeFeatures(Model, testPool);
    localExecutor->ExecRange(
            [&] (int treeIdx) {
                leafIdxs[treeIdx] = BuildIndicesForBinTree(Model, binarizedFeatures, static_cast<size_t>(treeIdx));
                },
            NPar::TLocalExecutor::TExecRangeParams(0, TreeCount),
            NPar::TLocalExecutor::WAIT_COMPLETE
    );
    return leafIdxs;
}

TVector<double> IDocumentImportancesEvaluator::ApplyModel(
        const TVector<TVector<ui32>>& leafIdxs,
        const ILeafValuesProvider<double>& leafValuesProvider) const
{
    TVector<double> result(leafIdxs[0].size());
    for (size_t treeIdx = 0; treeIdx < TreeCount; ++treeIdx) {
        auto& leafIdxsForTree = leafIdxs[treeIdx];
        for (size_t it = 0; it < LeafFormulaParams.LeafEstimationIterations; ++it) {
            auto& leafValues = leafValuesProvider.GetValuesForTreeIdxAndIterationNum(treeIdx, it);
            for (ui32 docId = 0; docId < leafIdxsForTree.size(); ++docId) {
                result[docId] += leafValues[leafIdxsForTree[docId]];
            }
        }
    }
    return result;
}

TVector<ui32> IDocumentImportancesEvaluator::GetLeafIdxsToUpdate(size_t treeId, const TVector<double> &jacobian) const {
    TVector<ui32> leafIdToUpdate;
    const size_t leafCount = 1U << static_cast<size_t>(Model.ObliviousTrees.TreeSizes[treeId]);

    if (UpdateMethod.UpdateType == EUpdateType::AllPoints) {
        leafIdToUpdate.resize(leafCount);
        std::iota(leafIdToUpdate.begin(), leafIdToUpdate.end(), 0);
    } else if (UpdateMethod.UpdateType == EUpdateType::TopKLeaves) {
        const TVector<ui32>& leafIndices = GetLeafIdxsProvider().GetLeafIdxsForTreeIdx(treeId);
        TVector<double> leafJacobians(leafCount);
        for (ui32 docId = 0; docId < DocCount; ++docId) {
            leafJacobians[leafIndices[docId]] += Abs(jacobian[docId]);
        }

        TVector<size_t> orderedLeafIndices(leafCount);
        std::iota(orderedLeafIndices.begin(), orderedLeafIndices.end(), 0);
        Sort(orderedLeafIndices.begin(), orderedLeafIndices.end(), [&](size_t firstDocId, ui32 secondDocId) {
            return leafJacobians[firstDocId] > leafJacobians[secondDocId];
        });

        leafIdToUpdate = TVector<ui32>(
            orderedLeafIndices.begin(),
            orderedLeafIndices.begin() + Min<int>(UpdateMethod.NumLeavesToUpdate, leafCount)
        );
    }

    return leafIdToUpdate;
}

TVector<double> IDocumentImportancesEvaluator::GetDocumentImportancesForOneTrainDoc(
        const TVector3Wrapper<double>& leafGradients,
        const TVector<TVector<ui32>>& leafIndices,
        const TVector<double>& lossGradientsWrtPredictions) const
{
    TVector<double> result;
    TVector<double> predictionGradientsWrtWeights = ApplyModel(leafIndices, leafGradients);
    if (!lossGradientsWrtPredictions.empty()) {
        result.resize(predictionGradientsWrtWeights.size());
        for (ui32 docIdx = 0; docIdx < predictionGradientsWrtWeights.size(); ++docIdx) {
            result[docIdx] = lossGradientsWrtPredictions[docIdx] * predictionGradientsWrtWeights[docIdx];
        }
    } else {
        result = std::move(predictionGradientsWrtWeights);
    }
    return result;
}
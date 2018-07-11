//
// Created by Boris Sharchilev on 10/07/2018.
//

#include "in_memory_evaluator.h"

std::unique_ptr<TInMemoryDocumentImportancesEvaluator> TInMemoryDocumentImportancesEvaluator::Construct(
        const TFullModel& model,
        const TPool& trainPool,
        const TInfluenceTarget& influenceTarget,
        const TUpdateMethod& updateMethod,
        const TTreesLimits& differentiatedTreesLimits,
        int threadCount)
{
    auto leafFormulaParams = TLeafFormulaParams::GetFromModel(model);
    TTreeStatisticsEvaluator treeStatisticsEvaluator = TTreeStatisticsEvaluator::Construct(
            model,
            trainPool,
            leafFormulaParams,
            differentiatedTreesLimits
    );
    TInMemoryDocumentImportancesEvaluator* importancesEvaluatorPtr = new TInMemoryDocumentImportancesEvaluator(
            model,
            treeStatisticsEvaluator.Evaluate(),
            updateMethod,
            differentiatedTreesLimits,
            influenceTarget,
            leafFormulaParams,
            model.ObliviousTrees.GetTreeCount(),
            trainPool.Docs.GetDocCount(),
            threadCount);
    return std::unique_ptr<TInMemoryDocumentImportancesEvaluator>(importancesEvaluatorPtr);
}

TVector3Wrapper<double> TInMemoryDocumentImportancesEvaluator::GetLeavesGradientsWrtWeight(size_t thisDocIdx) const {
    TVector<double> jacobian(DocCount);
    TVector<TVector<TVector<double>>> leavesGradients;
    leavesGradients.reserve(TreeCount);

    for (size_t treeIdx = 0; treeIdx < TreeCount; ++treeIdx) {
        TVector<TVector<double>> leavesGradientsForTree;
        leavesGradientsForTree.reserve(LeafFormulaParams.LeafEstimationIterations);

        // Check whether we need derivatives for this tree
        bool differentiateThisTree = DifferentiatedTreesLimits.fitsLimits(treeIdx);

        const TTreeStatistics& treeStatistics = TreeStatisticsVector.TreeStatisticsVector[treeIdx];
        for (size_t it = 0; it < LeafFormulaParams.LeafEstimationIterations; ++it) {
            TVector<double> leavesGradientsForTreeAndIteration;
            if (differentiateThisTree) {
                const TVector<ui32> leafIdxsToUpdate = GetLeafIdxsToUpdate(treeIdx, jacobian);
                leavesGradientsForTreeAndIteration = GetLeavesGradientsWrtWeightForTreeAndIteration(
                        thisDocIdx,
                        treeIdx,
                        it,
                        leafIdxsToUpdate,
                        jacobian
                );

                // Updating Jacobian
                bool isRemovedDocUpdated = false;
                for (size_t leafIdx : leafIdxsToUpdate) {
                    for (size_t docId : treeStatistics.LeavesDocsIdxs[leafIdx]) {
                        jacobian[docId] += leavesGradientsForTreeAndIteration[leafIdx];
                    }
                    isRemovedDocUpdated |= (treeStatistics.LeafIdxs[thisDocIdx] == leafIdx);
                }
                if (!isRemovedDocUpdated) {
                    size_t removedDocLeafId = treeStatistics.LeafIdxs[thisDocIdx];
                    jacobian[thisDocIdx] += leavesGradientsForTreeAndIteration[removedDocLeafId];
                }
            } else {
                int leafCount = (1U << Model.ObliviousTrees.TreeSizes[treeIdx]);
                leavesGradientsForTreeAndIteration = TVector<double>(leafCount, 0);
            }

            leavesGradientsForTree.push_back(std::move(leavesGradientsForTreeAndIteration));
        }
        leavesGradients.push_back(std::move(leavesGradientsForTree));
    }

    return TVector3Wrapper<double>(std::move(leavesGradients));
}

TVector<double> TInMemoryDocumentImportancesEvaluator::GetLeavesGradientsWrtWeightForTreeAndIteration(
        size_t thisDocIdx,
        size_t treeId,
        size_t leavesEstimationIteration,
        const TVector<ui32> &leafIdxsToUpdate,
        const TVector<double> &jacobian) const
{
    const TTreeStatistics& treeStatistics = TreeStatisticsVector.TreeStatisticsVector[treeId];
    const size_t removedDocLeafId = treeStatistics.LeafIdxs[thisDocIdx];
    const TVector<double>& formulaNumeratorJacobianMultiplier
            = treeStatistics.DerivativeFormulaParts.FormulaNumeratorJacobianMultiplier[leavesEstimationIteration];
    const TVector<double>& formulaNumeratorAddendum
            = treeStatistics.DerivativeFormulaParts.FormulaNumeratorAddendum[leavesEstimationIteration];
    const TVector<double>& formulaDenominators
            = treeStatistics.DerivativeFormulaParts.FormulaDenominators[leavesEstimationIteration];

    TVector<double> result(treeStatistics.LeafCount, 0);
    bool isThisDocIdxUpdated = false;
    for (size_t leafIdx : leafIdxsToUpdate) {
        for (size_t docIdx : treeStatistics.LeavesDocsIdxs[leafIdx]) {
            result[leafIdx] += formulaNumeratorJacobianMultiplier[docIdx] * jacobian[docIdx];
        }
        if (leafIdx == removedDocLeafId) {
            result[leafIdx] += formulaNumeratorAddendum[thisDocIdx];
        }
        result[leafIdx] *= -LeafFormulaParams.LearningRate / formulaDenominators[leafIdx];
        isThisDocIdxUpdated |= (leafIdx == removedDocLeafId);
    }
    if (!isThisDocIdxUpdated) {
        result[removedDocLeafId] += jacobian[thisDocIdx] * formulaNumeratorJacobianMultiplier[thisDocIdx];
        result[removedDocLeafId] += formulaNumeratorAddendum[thisDocIdx];
        result[removedDocLeafId] *= -LeafFormulaParams.LearningRate / formulaDenominators[removedDocLeafId];
    }
    return result;
}

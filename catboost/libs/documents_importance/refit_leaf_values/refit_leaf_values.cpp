//
// Created by Boris Sharchilev on 01/07/2018.
//

#include "refit_leaf_values.h"

void RefitLeafValues(TFullModel* model, const TPool& pool, int threadCount) {
    auto leafFormulaParams = TLeafFormulaParams::GetFromModel(*model);
    TTreeStatisticsEvaluator treeStatisticsEvaluator = TTreeStatisticsEvaluator::Construct(
            *model,
            pool,
            leafFormulaParams,
            TTreesLimits(model->GetTreeCount(), model->GetTreeCount())
    );
    TTreeStatisticsVectorWrapper treeStatisticsVector = treeStatisticsEvaluator.Evaluate();

    NPar::TLocalExecutor localExecutor;
    localExecutor.RunAdditionalThreads(threadCount - 1);

    const TVector<size_t>& treeOffsets = model->ObliviousTrees.GetFirstLeafOffsets();
    localExecutor.ExecRange(
            [&] (size_t treeIdx) {
                const TVector<double>& refittedLeafValues = treeStatisticsVector.TreeStatisticsVector[treeIdx].LeafValues.back();
                size_t leafCount = 1uLL << model->ObliviousTrees.TreeSizes[treeIdx];
                size_t treeOffset = treeOffsets[treeIdx];
                for (size_t leafIdx = 0; leafIdx < leafCount; ++leafIdx) {
                    model->ObliviousTrees.LeafValues[treeOffset + leafIdx] = refittedLeafValues[leafIdx];
                }
            },
            NPar::TLocalExecutor::TExecRangeParams(0, static_cast<int>(model->GetTreeCount())),
            NPar::TLocalExecutor::WAIT_COMPLETE
    );
}

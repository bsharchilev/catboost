//
// Created by Boris Sharchilev on 10/07/2018.
//
#pragma once

#include "importance_evaluator.h"

class TDocumentImportancesEvaluatorFactory {
public:
    static TDocumentImportancesEvaluatorFactory ForEvaluationMode(EInfluenceEvaluationMode influenceEvaluationMode) {
        return TDocumentImportancesEvaluatorFactory(influenceEvaluationMode);
    }

    std::unique_ptr<IDocumentImportancesEvaluator> Construct(
            const TFullModel& model,
            const TPool& trainPool,
            const TInfluenceTarget& influenceTarget,
            const TUpdateMethod& updateMethod,
            const TTreesLimits& differentiatedTreesLimits,
            int threadCount) const;

private:
    EInfluenceEvaluationMode InfluenceEvaluationMode;

    TDocumentImportancesEvaluatorFactory(EInfluenceEvaluationMode influenceEvaluationMode)
            : InfluenceEvaluationMode(influenceEvaluationMode)
    {

    }
};
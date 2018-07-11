//
// Created by Boris Sharchilev on 10/07/2018.
//

#include "importance_evaluator_factory.h"
#include "in_memory_evaluator.h"

std::unique_ptr<IDocumentImportancesEvaluator> TDocumentImportancesEvaluatorFactory::Construct(
        const TFullModel& model,
        const TPool& trainPool,
        const TInfluenceTarget& influenceTarget,
        const TUpdateMethod& updateMethod,
        const TTreesLimits& differentiatedTreesLimits,
        int threadCount) const
{
    switch (InfluenceEvaluationMode) {
        case EInfluenceEvaluationMode::InMemory:
            return TInMemoryDocumentImportancesEvaluator::Construct(
                    model,
                    trainPool,
                    influenceTarget,
                    updateMethod,
                    differentiatedTreesLimits,
                    threadCount
            );
            break;
        case EInfluenceEvaluationMode::OffMemory:
            CB_ENSURE(false, "Not implemented yet");
            break;
    }
}
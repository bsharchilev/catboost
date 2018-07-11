#pragma once

#include "importance_evaluators/importance_evaluator.h"
#include "influence_params.h"

struct TDStrResult {
    TDStrResult() = default;
    explicit TDStrResult(size_t testDocCount)
        : Indices(testDocCount)
        , Scores(testDocCount)
    {
    }

    TVector<TVector<ui32>> Indices; // [TestDocDount][Min(TopSize, TrainDocCount)]
    TVector<TVector<double>> Scores; // [TestDocDount][Min(TopSize, TrainDocCount)]
};

TDStrResult GetDocumentImportances(
    const TFullModel& model,
    const TPool& trainPool,
    const TPool& testPool,
    const TInfluenceRawParams& influenceStrParams
);


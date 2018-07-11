//
// Created by Boris Sharchilev on 20/06/2018.
//
#pragma once

#include "enums.h"

#include <getopt.h>
#include <util/generic/string.h>
#include <catboost/libs/data/pool.h>
#include <catboost/libs/options/loss_description.h>
#include <catboost/libs/model/model.h>
#include <utility>

struct TInfluenceRawParams {
    const TString InfluenceEvaluationMode;
    const TString InfluenceTarget;
    const TString DstrType;
    const int TopSize;
    const TString UpdateMethod;
    const size_t FirstDifferentiatedTreeIdx;
    const size_t LastDifferentiatedTreeIdx;
    const TString ImportanceValuesSign;
    const int ThreadCount;

    TInfluenceRawParams(
            TString influenceEvaluaionMode,
            TString influenceTarget,
            TString dstrType,
            int topSize,
            TString updateMethod,
            size_t firstDifferentiatedTreeIdx,
            size_t lastDifferentiatedTreeIdx,
            TString importanceValuesSign,
            int threadCount)
    : InfluenceEvaluationMode(std::move(influenceEvaluaionMode)),
      InfluenceTarget(std::move(influenceTarget)),
      DstrType(std::move(dstrType)),
      TopSize(topSize),
      UpdateMethod(std::move(updateMethod)),
      FirstDifferentiatedTreeIdx(firstDifferentiatedTreeIdx),
      LastDifferentiatedTreeIdx(lastDifferentiatedTreeIdx),
      ImportanceValuesSign(std::move(importanceValuesSign)),
      ThreadCount(threadCount)
    {

    }
};

class TInfluenceTarget {
public:
    const NCatboostOptions::TLossDescription LossDescription;
    const bool IsPredictionInfluenceTarget;

    static TInfluenceTarget Parse(const TString& influenceTargetStr, const TFullModel& model);

private:
    TInfluenceTarget(NCatboostOptions::TLossDescription&& lossDescription, bool isPredictionTarget)
            : LossDescription(std::move(lossDescription)),
              IsPredictionInfluenceTarget(isPredictionTarget)
    {

    }
};

class TUpdateMethod {
public:
    const EUpdateType UpdateType;
    const size_t NumLeavesToUpdate;

    static TUpdateMethod Parse(const TString &updateMethod);

private:
    TUpdateMethod(EUpdateType updateType, size_t numLeavesToUpdate)
            : UpdateType(updateType),
              NumLeavesToUpdate(numLeavesToUpdate)
    {

    }
};

class TTreesLimits {
public:
    const size_t FirstTreeIdx;
    const size_t LastTreeIdx;

    TTreesLimits(size_t firstTreeIdx, size_t lastTreeIdx):
            FirstTreeIdx(firstTreeIdx),
            LastTreeIdx(lastTreeIdx)
    {

    }

    bool fitsLimits(size_t treeIdx) const {
        return (treeIdx >= FirstTreeIdx) && (treeIdx <= LastTreeIdx);
    }
};

class TInfluenceParams {
public:
    const EInfluenceEvaluationMode InfluenceEvaluationMode;
    const TInfluenceTarget InfluenceTarget;
    const EDocumentStrengthType DstrType;
    const size_t TopSize;
    const TUpdateMethod UpdateMethod;
    const TTreesLimits DifferentiatedTreesLimits;
    const EImportanceValuesSign ImportanceValuesSign;
    const int ThreadCount;

    static TInfluenceParams Parse(
            const TInfluenceRawParams& rawParams,
            const TPool& trainingPool,
            const TFullModel& model);

private:
    TInfluenceParams(
            EInfluenceEvaluationMode influenceEvaluationMode,
            TInfluenceTarget influenceTarget,
            EDocumentStrengthType dstrType,
            size_t topSize,
            const TUpdateMethod& updateMethod,
            const TTreesLimits& differentiatedTreesLimits,
            EImportanceValuesSign importanceValuesSign,
            int threadCount)
            : InfluenceEvaluationMode(influenceEvaluationMode),
              InfluenceTarget(std::move(influenceTarget)),
              DstrType(dstrType),
              TopSize(topSize),
              UpdateMethod(updateMethod),
              DifferentiatedTreesLimits(differentiatedTreesLimits),
              ImportanceValuesSign(importanceValuesSign),
              ThreadCount(threadCount)
    {

    }
};

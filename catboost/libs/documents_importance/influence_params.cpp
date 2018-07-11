//
// Created by Boris Sharchilev on 20/06/2018.
//
#include "docs_importance.h"
#include "influence_params.h"

TInfluenceTarget TInfluenceTarget::Parse(const TString& influenceTargetStr, const TFullModel& model) {
    if (influenceTargetStr == "Prediction") {
        return {NCatboostOptions::TLossDescription(), true};
    }

    NJson::TJsonValue lossDescriptionJson = (influenceTargetStr == "TrainingLoss")
            ? ReadTJsonValue(model.ModelInfo.at("params"))["loss_function"]
            : LossDescriptionToJson(influenceTargetStr);
    NCatboostOptions::TLossDescription lossDescription;
    lossDescription.Load(lossDescriptionJson);
    return {std::move(lossDescription), false};
}

TUpdateMethod TUpdateMethod::Parse(const TString &updateMethod) {
    TString errorMessage = "Incorrect update-method param value. Should be one of: SinglePoint, \
        TopKLeaves, AllPoints or TopKLeaves:top=2 to set the top size in TopKLeaves method.";
    TVector<TString> tokens = StringSplitter(updateMethod).SplitLimited(':', 2).ToList<TString>();
    CB_ENSURE(tokens.size() <= 2, errorMessage);
    EUpdateType updateType;
    CB_ENSURE(TryFromString<EUpdateType>(tokens[0], updateType), tokens[0] + " update method is not supported");
    CB_ENSURE(tokens.size() == 1 || (tokens.size() == 2 && updateType == EUpdateType::TopKLeaves), errorMessage);
    size_t topSize = 0;
    if (tokens.size() == 2) {
        TVector<TString> keyValue = StringSplitter(tokens[1]).SplitLimited('=', 2).ToList<TString>();
        CB_ENSURE(keyValue[0] == "top", errorMessage);
        CB_ENSURE(TryFromString<size_t>(keyValue[1], topSize), "Top size should be nonnegative integer, got: " + keyValue[1]);
    }
    return {updateType, topSize};
}

TInfluenceParams TInfluenceParams::Parse(
        const TInfluenceRawParams &rawParams,
        const TPool &trainingPool,
        const TFullModel &model)
{
    size_t actualTopSize;
    if (rawParams.TopSize >= 0) {
        actualTopSize = static_cast<size_t>(rawParams.TopSize);
    } else {
        CB_ENSURE(rawParams.TopSize == -1, "Top size should be nonnegative integer or -1 (for unlimited top size).");
        actualTopSize = trainingPool.Docs.GetDocCount();
    }
    CB_ENSURE(
            (rawParams.FirstDifferentiatedTreeIdx >= 0) && (rawParams.LastDifferentiatedTreeIdx >= 0),
            "Differentiated trees limits have to be non-negative."
    );

    return {
        FromString<EInfluenceEvaluationMode>(rawParams.InfluenceEvaluationMode),
        TInfluenceTarget::Parse(rawParams.InfluenceTarget, model),
        FromString<EDocumentStrengthType>(rawParams.DstrType),
        actualTopSize,
        TUpdateMethod::Parse(rawParams.UpdateMethod),
        TTreesLimits(
                static_cast<size_t>(rawParams.FirstDifferentiatedTreeIdx),
                static_cast<size_t>(rawParams.LastDifferentiatedTreeIdx)
        ),
        FromString<EImportanceValuesSign>(rawParams.ImportanceValuesSign),
        rawParams.ThreadCount
    };
}
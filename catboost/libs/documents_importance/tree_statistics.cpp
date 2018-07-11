#include <contrib/libs/cxxsupp/libcxx/include/iostream>
#include "tree_statistics.h"

// TLeafFormulaParams
TLeafFormulaParams TLeafFormulaParams::GetFromModel(const TFullModel& model) {
    NJson::TJsonValue modelParamsJson = ReadTJsonValue(model.ModelInfo.at("params"));
    return {
        FromString<ELeavesEstimation>(modelParamsJson["tree_learner_options"]["leaf_estimation_method"].GetString()),
        modelParamsJson["tree_learner_options"]["leaf_estimation_iterations"].GetUInteger(),
        modelParamsJson["boosting_options"]["learning_rate"].GetDouble(),
        modelParamsJson["tree_learner_options"]["l2_leaf_reg"].GetDouble()
    };
}

// TDerivativeFormulaParts
void TTreeStatistics::TDerivativeFormulaParts::reserve(size_t n) {
    FormulaDenominators.reserve(n);
    FormulaNumeratorAddendum.reserve(n);
    FormulaNumeratorJacobianMultiplier.reserve(n);
}

// IDerivativeFormulaPartsEvaluator
std::unique_ptr<IDerivativeFormulaPartsEvaluator> IDerivativeFormulaPartsEvaluator::Construct(
        size_t docCount,
        size_t leafCount,
        const TVector<ui32>& leafIdxs,
        const TPool& trainPool,
        const TLeafFormulaParams& leafFormulaParams,
        const TLossDescription& lossDescription)
{
    if (leafFormulaParams.LeafEstimationMethod == ELeavesEstimation::Gradient) {
        return std::make_unique<TGradientDerivativeFormulaPartsEvaluator>(
                docCount, leafCount, leafIdxs, trainPool, leafFormulaParams, lossDescription
        );
    }
    Y_ASSERT(leafFormulaParams.LeafEstimationMethod == ELeavesEstimation::Newton);
    return std::make_unique<TNewtonDerivativeFormulaPartsEvaluator>(
            docCount, leafCount, leafIdxs, trainPool, leafFormulaParams, lossDescription
    );
}

void IDerivativeFormulaPartsEvaluator::AddLeafValuesAndDerivativeFormulaParts(
        const TVector<double>& approxes,
        TVector<TVector<double>>* leafValues,
        TTreeStatistics::TDerivativeFormulaParts* derivativeFormulaParts,
        bool computeDerivativeFormulaParts)
{
    UpdateDerivatives(approxes);

    TVector<double> leafNumerators = ComputeLeafNumerators();
    TVector<double> leafDenominators = ComputeLeafDenominators();
    TVector<double> currentLeafValues = LeafValuesFromNumeratorsAndDenominators(leafNumerators, leafDenominators);

    if (computeDerivativeFormulaParts) {
        TVector<double> formulaNumeratorAddendum = ComputeFormulaNumeratorAddendum(currentLeafValues);
        TVector<double> formulaNumeratorJacobianMultiplier = ComputeFormulaNumeratorJacobianMultiplier(currentLeafValues);

        derivativeFormulaParts->FormulaDenominators.push_back(std::move(leafDenominators));
        derivativeFormulaParts->FormulaNumeratorAddendum.push_back(std::move(formulaNumeratorAddendum));
        derivativeFormulaParts->FormulaNumeratorJacobianMultiplier.push_back(std::move(formulaNumeratorJacobianMultiplier));
    }
    leafValues->push_back(std::move(currentLeafValues));
}

TVector<double> IDerivativeFormulaPartsEvaluator::LeafValuesFromNumeratorsAndDenominators(
        const TVector<double>& leafNumerators,
        const TVector<double>& leafDenominators) const
{
    TVector<double> result(LeafCount);
    for (ui32 leafId = 0; leafId < LeafCount; ++leafId) {
        result[leafId] = -leafNumerators[leafId] / leafDenominators[leafId] * LeafFormulaParams.LearningRate;
    }
    return result;
}

// TGradientDerivativeFormulaPartsEvaluator
void TGradientDerivativeFormulaPartsEvaluator::UpdateDerivatives(const TVector<double>& approxes) {
    EvaluateDerivatives(
            LossDescription,
            approxes,
            TrainPool,
            &FirstDerivatives,
            &SecondDerivatives,
            nullptr
    );
}

TVector<double> TGradientDerivativeFormulaPartsEvaluator::ComputeLeafNumerators() const {
    TVector<double> leafNumerators(LeafCount);
    if (TrainPool.Docs.Weight.empty()) {
        for (ui32 docId = 0; docId < DocCount; ++docId) {
            leafNumerators[LeafIdxs[docId]] += FirstDerivatives[docId];
        }
    } else {
        for (ui32 docId = 0; docId < DocCount; ++docId) {
            leafNumerators[LeafIdxs[docId]] += TrainPool.Docs.Weight[docId] * FirstDerivatives[docId];
        }
    }
    return leafNumerators;
}

TVector<double> TGradientDerivativeFormulaPartsEvaluator::ComputeLeafDenominators() const {
    TVector<double> leafDenominators(LeafCount);
    if (TrainPool.Docs.Weight.empty()) {
        for (ui32 docId = 0; docId < DocCount; ++docId) {
            leafDenominators[LeafIdxs[docId]] += 1;
        }
    } else {
        for (ui32 docId = 0; docId < DocCount; ++docId) {
            leafDenominators[LeafIdxs[docId]] += TrainPool.Docs.Weight[docId];
        }
    }
    for (ui32 leafId = 0; leafId < LeafCount; ++leafId) {
        leafDenominators[leafId] += LeafFormulaParams.L2LeafReg;
    }
    return leafDenominators;
}

TVector<double> TGradientDerivativeFormulaPartsEvaluator::ComputeFormulaNumeratorAddendum(
        const TVector<double>& leafValues) const
{
    TVector<double> formulaNumeratorAdding(DocCount);
    for (ui32 docId = 0; docId < DocCount; ++docId) {
        formulaNumeratorAdding[docId] = leafValues[LeafIdxs[docId]] / LeafFormulaParams.LearningRate + FirstDerivatives[docId];
    }
    return formulaNumeratorAdding;
}

TVector<double> TGradientDerivativeFormulaPartsEvaluator::ComputeFormulaNumeratorJacobianMultiplier(
        const TVector<double>&/*leafValues*/) const
{
    TVector<double> formulaNumeratorMultiplier(DocCount);
    if (TrainPool.Docs.Weight.empty()) {
        formulaNumeratorMultiplier = SecondDerivatives;
    } else {
        for (ui32 docId = 0; docId < DocCount; ++docId) {
            formulaNumeratorMultiplier[docId] = TrainPool.Docs.Weight[docId] * SecondDerivatives[docId];
        }
    }
    return formulaNumeratorMultiplier;
}

// TNewtonDerivativeFormulaPartsEvaluator
void TNewtonDerivativeFormulaPartsEvaluator::UpdateDerivatives(const TVector<double>& approxes) {
    EvaluateDerivatives(
            LossDescription,
            approxes,
            TrainPool,
            &FirstDerivatives,
            &SecondDerivatives,
            &ThirdDerivatives
    );
}

TVector<double> TNewtonDerivativeFormulaPartsEvaluator::ComputeLeafNumerators() const {
    TVector<double> leafNumerators(LeafCount);
    if (TrainPool.Docs.Weight.empty()) {
        for (ui32 docId = 0; docId < DocCount; ++docId) {
            leafNumerators[LeafIdxs[docId]] += FirstDerivatives[docId];
        }
    } else {
        for (ui32 docId = 0; docId < DocCount; ++docId) {
            leafNumerators[LeafIdxs[docId]] += TrainPool.Docs.Weight[docId] * FirstDerivatives[docId];
        }
    }
    return leafNumerators;
}

TVector<double> TNewtonDerivativeFormulaPartsEvaluator::ComputeLeafDenominators() const {
    TVector<double> leafDenominators(LeafCount);
    if (TrainPool.Docs.Weight.empty()) {
        for (ui32 docId = 0; docId < DocCount; ++docId) {
            leafDenominators[LeafIdxs[docId]] += SecondDerivatives[docId];
        }
    } else {
        for (ui32 docId = 0; docId < DocCount; ++docId) {
            leafDenominators[LeafIdxs[docId]] += TrainPool.Docs.Weight[docId] * SecondDerivatives[docId];
        }
    }
    for (ui32 leafId = 0; leafId < LeafCount; ++leafId) {
        leafDenominators[leafId] += LeafFormulaParams.L2LeafReg;
    }
    return leafDenominators;
}

TVector<double> TNewtonDerivativeFormulaPartsEvaluator::ComputeFormulaNumeratorAddendum(
        const TVector<double>& leafValues) const
{
    TVector<double> formulaNumeratorAdding(DocCount);
    for (ui32 docId = 0; docId < DocCount; ++docId) {
        formulaNumeratorAdding[docId] = leafValues[LeafIdxs[docId]] * SecondDerivatives[docId]
                / LeafFormulaParams.LearningRate + FirstDerivatives[docId];
    }
    return formulaNumeratorAdding;
}

TVector<double> TNewtonDerivativeFormulaPartsEvaluator::ComputeFormulaNumeratorJacobianMultiplier(
        const TVector<double>& leafValues) const
{
    TVector<double> formulaNumeratorMultiplier(DocCount);
    if (TrainPool.Docs.Weight.empty()) {
        for (ui32 docId = 0; docId < DocCount; ++docId) {
            formulaNumeratorMultiplier[docId] = leafValues[LeafIdxs[docId]] * ThirdDerivatives[docId]
                    / LeafFormulaParams.LearningRate + SecondDerivatives[docId];
        }
    } else {
        for (ui32 docId = 0; docId < DocCount; ++docId) {
            formulaNumeratorMultiplier[docId] = TrainPool.Docs.Weight[docId] * (leafValues[LeafIdxs[docId]]
                    * ThirdDerivatives[docId] / LeafFormulaParams.LearningRate + SecondDerivatives[docId]);
        }
    }
    return formulaNumeratorMultiplier;
}

// TTreeStatisticsEvaluator
TTreeStatisticsEvaluator TTreeStatisticsEvaluator::Construct(
        const TFullModel& model,
        const TPool& trainPool,
        const TLeafFormulaParams& leafFormulaParams,
        const TTreesLimits& differentiatedTreesLimits)
{
    TVector<ui8> binarizedFeatures = BinarizeFeatures(model, trainPool);
    TLossDescription trainLossDescription;
    NJson::TJsonValue modelTrainLossJson = ReadTJsonValue(model.ModelInfo.at("params"))["loss_function"];
    trainLossDescription.Load(modelTrainLossJson);
    return {
        model,
        trainPool,
        leafFormulaParams,
        differentiatedTreesLimits,
        std::move(binarizedFeatures),
        std::move(trainLossDescription)
    };
}

TTreeStatistics TTreeStatisticsEvaluator::EvaluateForOneTree(size_t treeIdx, TVector<double>* approxes) const {
    auto leafCount = static_cast<size_t>(1U << Model.ObliviousTrees.TreeSizes[treeIdx]);
    TVector<ui32> leafIdxs = BuildIndicesForBinTree(Model, BinarizedFeatures, treeIdx);
    TVector<TVector<ui32>> leavesDocId = PartitionDocsByLeaves(leafCount, leafIdxs);

    // Check whether we need derivatives for this tree
    bool differentiateThisTree = DifferentiatedTreesLimits.fitsLimits(treeIdx);

    TVector<TVector<double>> leafValues;
    leafValues.reserve(LeafFormulaParams.LeafEstimationIterations);
    TTreeStatistics::TDerivativeFormulaParts derivativeFormulaParts;
    if (differentiateThisTree) {
        derivativeFormulaParts.reserve(LeafFormulaParams.LeafEstimationIterations);
    }

    std::unique_ptr<IDerivativeFormulaPartsEvaluator> derivativeFormulaPartsEvaluator
            = IDerivativeFormulaPartsEvaluator::Construct(
                    TrainPool.Docs.GetDocCount(),
                    leafCount,
                    leafIdxs,
                    TrainPool,
                    LeafFormulaParams,
                    TrainLossDescription
            );
    for (size_t iteration = 0; iteration < LeafFormulaParams.LeafEstimationIterations; ++iteration) {
        derivativeFormulaPartsEvaluator->AddLeafValuesAndDerivativeFormulaParts(
                *approxes,
                &leafValues,
                &derivativeFormulaParts,
                differentiateThisTree
        );
        UpdateApproxesWithLeafValues(leafValues.back(), leafIdxs, approxes);
    }

    return TTreeStatistics(
            leafCount,
            std::move(leafIdxs),
            std::move(leavesDocId),
            std::move(leafValues),
            std::move(derivativeFormulaParts)
    );
}

TVector<TVector<ui32>> TTreeStatisticsEvaluator::PartitionDocsByLeaves(
        size_t leafCount,
        const TVector<ui32>& leafIndices) const
{
    TVector<TVector<ui32>> result(leafCount);
    for (ui32 docId : leafIndices) {
        result[leafIndices[docId]].push_back(docId);
    }
    return result;
}

void TTreeStatisticsEvaluator::UpdateApproxesWithLeafValues(
        const TVector<double>& leafValues,
        const TVector<ui32>& leafIdxs,
        TVector<double>* approxes) const
{
    for (size_t docIdx = 0; docIdx < approxes->size(); ++docIdx) {
        (*approxes)[docIdx] += leafValues[leafIdxs[docIdx]];
    }
}
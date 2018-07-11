#pragma once

#include "enums.h"
#include "influence_params.h"
#include "ders_helpers.h"
#include "docs_importance_helpers.h"

#include <catboost/libs/model/model.h>
#include <catboost/libs/data/pool.h>
#include <catboost/libs/options/catboost_options.h>
#include <utility>
#include <catboost/libs/algo/index_calcer.h>

using NCatboostOptions::TLossDescription;

class TLeafFormulaParams {
public:
    const ELeavesEstimation LeafEstimationMethod;
    const size_t LeafEstimationIterations;
    const double LearningRate;
    const double L2LeafReg;

    static TLeafFormulaParams GetFromModel(const TFullModel& model);

private:
    TLeafFormulaParams(
            ELeavesEstimation leafEstimationMethod,
            size_t leafEstimationIterations,
            double learningRate,
            double l2LeafReg)
            : LeafEstimationMethod(leafEstimationMethod),
              LeafEstimationIterations(leafEstimationIterations),
              LearningRate(learningRate),
              L2LeafReg(l2LeafReg)
    {

    }
};

class TTreeStatistics {
public:
    class TDerivativeFormulaParts {
    public:
        TVector<TVector<double>> FormulaDenominators; // [LeavesEstimationIterationsCount][leafCount] // Denominator from equation (6).
        TVector<TVector<double>> FormulaNumeratorAddendum; // [LeavesEstimationIterationsCount][docCount] // The first term from equation (6).
        TVector<TVector<double>> FormulaNumeratorJacobianMultiplier; // [LeavesEstimationIterationsCount][docCount] // The jacobian multiplier from equation (6).

        void reserve(size_t n);
    };

    const size_t LeafCount;
    const TVector<ui32> LeafIdxs; // [docCount] // leafId for every train docId.
    const TVector<TVector<ui32>> LeavesDocsIdxs; // [leafCount] // docIds for every leafId.
    const TVector<TVector<double>> LeafValues; // [LeavesEstimationIterationsCount][leafCount]
    const TDerivativeFormulaParts DerivativeFormulaParts;

    TTreeStatistics(
            size_t leafCount,
            TVector<ui32> leafIndices,
            TVector<TVector<ui32>> leavesDocId,
            TVector<TVector<double>> leafValues,
            TDerivativeFormulaParts derivativeFormulaParts)
            : LeafCount(leafCount),
              LeafIdxs(std::move(leafIndices)),
              LeavesDocsIdxs(std::move(leavesDocId)),
              LeafValues(std::move(leafValues)),
              DerivativeFormulaParts(std::move(derivativeFormulaParts))
    {

    }
};

class TTreeStatisticsVectorWrapper : public ILeafValuesProvider<double>, public ILeafIdxsProvider {
public:
    const TVector<TTreeStatistics> TreeStatisticsVector;

    explicit TTreeStatisticsVectorWrapper(TVector<TTreeStatistics> treeStatisticsVector)
            : TreeStatisticsVector(std::move(treeStatisticsVector))
    {

    }

    const TVector<double> GetValuesForTreeIdxAndIterationNum(size_t treeIdx, size_t it) const override {
        return TreeStatisticsVector[treeIdx].LeafValues[it];
    }

    const TVector<ui32> GetLeafIdxsForTreeIdx(size_t treeIdx) const {
        return TreeStatisticsVector[treeIdx].LeafIdxs;
    }
};

class IDerivativeFormulaPartsEvaluator {
public:
    virtual ~IDerivativeFormulaPartsEvaluator() = default;

    static std::unique_ptr<IDerivativeFormulaPartsEvaluator> Construct(
            size_t docCount,
            size_t leafCount,
            const TVector<ui32>& leafIdxs,
            const TPool& trainPool,
            const TLeafFormulaParams& leafFormulaParams,
            const TLossDescription& lossDescription);

    void AddLeafValuesAndDerivativeFormulaParts(
            const TVector<double>& approxes,
            TVector<TVector<double>>* leafValues,
            TTreeStatistics::TDerivativeFormulaParts* derivativeFormulaParts,
            bool computeDerivativeFormulaParts);

protected:
    virtual void UpdateDerivatives(const TVector<double>& approxes) = 0;

    virtual TVector<double> ComputeLeafNumerators() const = 0;
    virtual TVector<double> ComputeLeafDenominators() const = 0;
    virtual TVector<double> ComputeFormulaNumeratorAddendum(const TVector<double>& leafValues) const = 0;
    virtual TVector<double> ComputeFormulaNumeratorJacobianMultiplier(const TVector<double>& leafValues) const = 0;

protected:
    const size_t DocCount;
    const size_t LeafCount;
    const TVector<ui32>& LeafIdxs;
    const TPool& TrainPool;
    const TLeafFormulaParams& LeafFormulaParams;
    const TLossDescription& LossDescription;

    TVector<double> LeafValuesFromNumeratorsAndDenominators(
            const TVector<double>& leafNumerators,
            const TVector<double>& leafDenominators) const;

    IDerivativeFormulaPartsEvaluator(
            size_t docCount,
            size_t leafCount,
            const TVector<ui32>& leafIndices,
            const TPool& trainPool,
            const TLeafFormulaParams& leafFormulaParams,
            const TLossDescription& lossDescription)
            : DocCount(docCount),
              LeafCount(leafCount),
              LeafIdxs(leafIndices),
              TrainPool(trainPool),
              LeafFormulaParams(leafFormulaParams),
              LossDescription(lossDescription)
    {

    }
};

class TGradientDerivativeFormulaPartsEvaluator : public IDerivativeFormulaPartsEvaluator {
public:
    ~TGradientDerivativeFormulaPartsEvaluator() override {
        ~FirstDerivatives;
        ~SecondDerivatives;
    }

    void UpdateDerivatives(const TVector<double>& approxes) override;

    TVector<double> ComputeLeafNumerators() const override;
    TVector<double> ComputeLeafDenominators() const override;
    TVector<double> ComputeFormulaNumeratorAddendum(const TVector<double>& leafValues) const override;
    TVector<double> ComputeFormulaNumeratorJacobianMultiplier(const TVector<double>& leafValues) const override;

    TVector<double> FirstDerivatives;
    TVector<double> SecondDerivatives;

    TGradientDerivativeFormulaPartsEvaluator(
            size_t docCount,
            size_t leafCount,
            const TVector<ui32>& leafIndices,
            const TPool& trainPool,
            const TLeafFormulaParams& leafFormulaParams,
            const TLossDescription& lossDescription)
            : IDerivativeFormulaPartsEvaluator(docCount, leafCount, leafIndices, trainPool, leafFormulaParams, lossDescription),
              FirstDerivatives(docCount),
              SecondDerivatives(docCount)
    {

    }
};

class TNewtonDerivativeFormulaPartsEvaluator : public IDerivativeFormulaPartsEvaluator {
public:
    ~TNewtonDerivativeFormulaPartsEvaluator() override {
        ~FirstDerivatives;
        ~SecondDerivatives;
        ~ThirdDerivatives;
    }

    void UpdateDerivatives(const TVector<double>& approxes) override;

    TVector<double> ComputeLeafNumerators() const override;
    TVector<double> ComputeLeafDenominators() const override;
    TVector<double> ComputeFormulaNumeratorAddendum(const TVector<double>& leafValues) const override;
    TVector<double> ComputeFormulaNumeratorJacobianMultiplier(const TVector<double>& leafValues) const override;

    TVector<double> FirstDerivatives;
    TVector<double> SecondDerivatives;
    TVector<double> ThirdDerivatives;

    TNewtonDerivativeFormulaPartsEvaluator(
            size_t docCount,
            size_t leafCount,
            const TVector<ui32>& leafIndices,
            const TPool& trainPool,
            const TLeafFormulaParams& leafFormulaParams,
            const TLossDescription& lossDescription)
            : IDerivativeFormulaPartsEvaluator(docCount, leafCount, leafIndices, trainPool, leafFormulaParams, lossDescription),
              FirstDerivatives(docCount),
              SecondDerivatives(docCount),
              ThirdDerivatives(docCount)
    {

    }
};

class TTreeStatisticsEvaluator {
public:
    static TTreeStatisticsEvaluator Construct(
            const TFullModel& model,
            const TPool& trainPool,
            const TLeafFormulaParams& leafFormulaParams,
            const TTreesLimits& differentiatedTreesLimits);

    TTreeStatisticsVectorWrapper Evaluate() {
        TVector<TTreeStatistics> result;
        result.reserve(Model.GetTreeCount());

        TVector<double> approxes = (!TrainPool.Docs.Baseline.empty())
                ? TrainPool.Docs.Baseline.front()
                : TVector<double>(TrainPool.Docs.GetDocCount(), 0);

        for (size_t treeIdx = 0; treeIdx < Model.GetTreeCount(); ++treeIdx) {
            result.push_back(EvaluateForOneTree(treeIdx, &approxes));
        }

        return TTreeStatisticsVectorWrapper(std::move(result));
    }

private:
    const TFullModel& Model;
    const TPool& TrainPool;
    const TLeafFormulaParams& LeafFormulaParams;
    const TTreesLimits& DifferentiatedTreesLimits;
    const TVector<ui8> BinarizedFeatures;
    const TLossDescription TrainLossDescription;

    TTreeStatisticsEvaluator(
            const TFullModel& model,
            const TPool& trainPool,
            const TLeafFormulaParams& leafFormulaParams,
            const TTreesLimits& differentiatedTreesLimits,
            TVector<ui8> binarizedFeatures,
            TLossDescription trainLossDescription)
            : Model(model),
              TrainPool(trainPool),
              LeafFormulaParams(leafFormulaParams),
              DifferentiatedTreesLimits(differentiatedTreesLimits),
              BinarizedFeatures(std::move(binarizedFeatures)),
              TrainLossDescription(std::move(trainLossDescription))
    {

    }

    TTreeStatistics EvaluateForOneTree(size_t treeIdx, TVector<double>* approxes) const;

    TVector<TVector<ui32>> PartitionDocsByLeaves(size_t leafCount, const TVector<ui32>& leafIndices) const; // [leafIdx][docIdx]

    void UpdateApproxesWithLeafValues(
            const TVector<double>& leafValues,
            const TVector<ui32>& leafIdxs,
            TVector<double>* approxes) const;
};
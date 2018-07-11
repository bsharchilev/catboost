#include "ders_helpers.h"

#include "catboost/libs/algo/error_functions.h"
#include "influence_params.h"

template<typename TError>
void EvaluateDerivativesForError(
    const TVector<double>& approxes,
    const TPool& pool,
    const NCatboostOptions::TLossDescription& lossDescription,
    TVector<double>* firstDerivatives,
    TVector<double>* secondDerivatives,
    TVector<double>* thirdDerivatives
) {
    ELossFunction lossFunction = lossDescription.GetLossFunction();
    const bool isStoreExpApprox = IsStoreExpApprox(lossFunction);
    size_t docCount = pool.Docs.GetDocCount();

    TVector<double> expApproxes;
    if (isStoreExpApprox) {
        expApproxes.resize(docCount);
        for (ui32 docId = 0; docId < docCount; ++docId) {
            expApproxes[docId] = fast_exp(approxes[docId]);
        }
    }
    const TVector<double>& approxesRef = isStoreExpApprox ? expApproxes : approxes;

    TError error(isStoreExpApprox);
    TVector<TDers> derivatives(docCount);

    Y_ASSERT(error.GetErrorType() == EErrorType::PerObjectError);
    error.CalcDersRange(
        /*start=*/0,
        docCount,
        /*calcThirdDer=*/thirdDerivatives != nullptr,
        approxesRef.data(),
        /*approxDeltas=*/nullptr,
        pool.Docs.Target.data(),
        /*weights=*/nullptr,
        derivatives.data()
    );

    for (ui32 docId = 0; docId < docCount; ++docId) {
        if (firstDerivatives) {
            (*firstDerivatives)[docId] = -derivatives[docId].Der1;
        }
        if (secondDerivatives) {
            (*secondDerivatives)[docId] = -derivatives[docId].Der2;
        }
        if (thirdDerivatives) {
            (*thirdDerivatives)[docId] = -derivatives[docId].Der3;
        }
    }
}

using TEvaluateDerivativesFunc = std::function<void(
    const TVector<double>& approxes,
    const TPool& pool,
    const NCatboostOptions::TLossDescription& influenceTarget,
    TVector<double>* firstDerivatives,
    TVector<double>* secondDerivatives,
    TVector<double>* thirdDerivatives
)>;

static TEvaluateDerivativesFunc GetEvaluateDerivativesFunc(const NCatboostOptions::TLossDescription& lossDescription) {
    ELossFunction lossFunction = lossDescription.GetLossFunction();
    switch (lossFunction) {
        case ELossFunction::Logloss:
        case ELossFunction::CrossEntropy:
            return EvaluateDerivativesForError<TCrossEntropyError>;
            break;
        case ELossFunction::RMSE:
            return EvaluateDerivativesForError<TRMSEError>;
            break;
        case ELossFunction::MAE:
        case ELossFunction::Quantile:
            return EvaluateDerivativesForError<TQuantileError>;
            break;
        case ELossFunction::LogLinQuantile:
            return EvaluateDerivativesForError<TLogLinQuantileError>;
            break;
        case ELossFunction::MAPE:
            return EvaluateDerivativesForError<TMAPError>;
            break;
        case ELossFunction::Poisson:
            return EvaluateDerivativesForError<TPoissonError>;
            break;
        default:
            CB_ENSURE(false, "provided error function is not supported yet");
    }
}

void EvaluateDerivatives(
    const NCatboostOptions::TLossDescription& lossDescription,
    const TVector<double>& approxes,
    const TPool& pool,
    TVector<double>* firstDerivatives,
    TVector<double>* secondDerivatives,
    TVector<double>* thirdDerivatives
) {
    auto evaluateDerivativesFunc = GetEvaluateDerivativesFunc(lossDescription);
    evaluateDerivativesFunc(
        approxes,
        pool,
        lossDescription,
        firstDerivatives,
        secondDerivatives,
        thirdDerivatives
    );
}

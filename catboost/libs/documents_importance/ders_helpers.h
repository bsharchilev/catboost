#pragma once

#include <catboost/libs/algo/approx_util.h>
#include <catboost/libs/algo/approx_calcer_querywise.h>
#include <catboost/libs/data/pool.h>
#include <catboost/libs/metrics/ders_holder.h>
#include <catboost/libs/helpers/query_info_helper.h>
#include "influence_params.h"

void EvaluateDerivatives(
    const NCatboostOptions::TLossDescription& lossDescription,
    const TVector<double>& approxes,
    const TPool& pool,
    TVector<double>* firstDerivatives,
    TVector<double>* secondDerivatives,
    TVector<double>* thirdDerivatives
);

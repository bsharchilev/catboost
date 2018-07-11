//
// Created by Boris Sharchilev on 01/07/2018.
//
#pragma once

#include <catboost/libs/documents_importance/tree_statistics.h>

void RefitLeafValues(TFullModel* model, const TPool& pool, int threadCount);
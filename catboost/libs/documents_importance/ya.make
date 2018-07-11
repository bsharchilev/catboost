LIBRARY()

OWNER(
    g:matrixnet
)

SRCS(
    importance_evaluators/importance_evaluator.cpp
    importance_evaluators/importance_evaluator_factory.cpp
    importance_evaluators/in_memory_evaluator.cpp
    docs_importance.cpp
    docs_importance_helpers.cpp
    tree_statistics.cpp
    ders_helpers.cpp
    influence_params.cpp
)

PEERDIR(
    catboost/libs/algo
    catboost/libs/data
    catboost/libs/model
    catboost/libs/options
    catboost/libs/metrics
    catboost/libs/helpers
)

GENERATE_ENUM_SERIALIZATION(
    enums.h
)

END()

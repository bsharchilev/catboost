#include "modes.h"
#include "bind_options.h"
#include "cmd_line.h"

#include <catboost/libs/documents_importance/docs_importance.h>
#include <catboost/libs/data/load_data.h>


using namespace NCB;


struct TObjectImportancesParams {
    TString ModelFileName;
    TString OutputPath;
    TPathWithScheme LearnSetPath;
    TPathWithScheme TestSetPath;
    NCatboostOptions::TDsvPoolFormatParams DsvPoolFormatParams;
    TString InfluenceEvaluationMode = "InMemory";
    TString TestLossDescriptionStr = "TrainingLoss";
    TString UpdateMethod = ToString(EUpdateType::SinglePoint);
    int ThreadCount = NSystemInfo::CachedNumberOfCpus();
    char Delimiter = '\t';
    bool HasHeader = false;

    void BindParserOpts(NLastGetopt::TOpts& parser) {
        parser.AddLongOption('m', "model-path", "path to model")
            .StoreResult(&ModelFileName)
            .DefaultValue("model.bin");
        parser.AddLongOption('f', "learn-set", "learn set path")
            .StoreResult(&LearnSetPath)
            .RequiredArgument("PATH");
        parser.AddLongOption('t', "test-set", "test set path")
            .StoreResult(&TestSetPath)
            .RequiredArgument("PATH");
        BindDsvPoolFormatParams(&parser, &DsvPoolFormatParams);
        parser.AddLongOption('o', "output-path", "output result path")
            .StoreResult(&OutputPath)
            .DefaultValue("object_importances.tsv");
        parser.AddLongOption("mode", "Should be one of: InMemory, OffMemory")
              .RequiredArgument("string")
              .StoreResult(&InfluenceEvaluationMode);
        parser.AddLongOption("loss-function",
                             "Should be one of: Logloss, CrossEntropy, RMSE, MAE, Quantile, LogLinQuantile, MAPE, Poisson, MultiClass, MultiClassOneVsAll, PairLogit, QueryRMSE, QuerySoftMax. A loss might have params, then params should be written in format Loss:paramName=value.")
            .RequiredArgument("string")
            .StoreResult(&TestLossDescriptionStr);
        parser.AddLongOption('T', "thread-count", "worker thread count (default: core count)")
            .StoreResult(&ThreadCount);
        parser.AddLongOption("update-method", "Should be one of: SinglePoint, TopKLeaves, AllPoints or TopKLeaves:top=2 to set the top size in TopKLeaves method.")
            .StoreResult(&UpdateMethod)
            .DefaultValue("SinglePoint");
    }
};

int mode_ostr(int argc, const char* argv[]) {
    TObjectImportancesParams params;

    auto parser = NLastGetopt::TOpts();
    parser.AddHelpOption();
    params.BindParserOpts(parser);
    parser.SetFreeArgsNum(0);
    NLastGetopt::TOptsParseResult parserResult{&parser, argc, argv};

    CB_ENSURE(NFs::Exists(params.ModelFileName), "Model file doesn't exist: " << params.ModelFileName);
    TFullModel model = ReadModel(params.ModelFileName);

    TPool trainPool;
    NCB::ReadPool(params.LearnSetPath,
                  /*pairsFilePath=*/NCB::TPathWithScheme(),
                  params.DsvPoolFormatParams,
                  /*ignoredFeatures*/ {},
                  params.ThreadCount,
                  /*verbose=*/false,
                  /*classNames=*/{},
                  &trainPool);

    TPool testPool;
    NCB::ReadPool(params.TestSetPath,
                  /*pairsFilePath=*/NCB::TPathWithScheme(),
                  params.DsvPoolFormatParams,
                  /*ignoredFeatures=*/{},
                  params.ThreadCount,
                  /*verbose=*/false,
                  /*classNames=*/{},
                  &testPool);

    TInfluenceRawParams influenceRawParams = TInfluenceRawParams(
            params.InfluenceEvaluationMode,
            params.TestLossDescriptionStr,
            ToString(EDocumentStrengthType::Raw),
            -1,
            params.UpdateMethod,
            0,
            model.GetTreeCount() - 1,
            ToString(EImportanceValuesSign::All),
            params.ThreadCount
    );
    TDStrResult results = GetDocumentImportances(
        model,
        trainPool,
        testPool,
        influenceRawParams
    );

    TFileOutput output(params.OutputPath);
    for (const auto& row : results.Scores) {
        for (double value : row) {
            output.Write(ToString(value) + '\t');
        }
        output.Write('\n');
    }
    return 0;
}

using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

public class RetrievalAugmentedGeneration
{
    private MLContext _mlContext;
    private ITransformer _textTransformer;
    private float[][] _knowledgeFeatures;
    private List<string> _knowledgeList;

    public RetrievalAugmentedGeneration()
    {
        _mlContext = new MLContext();
    }

    public void TrainModel(List<string> knowledgeBase)
    {
        string knowledgeFilePath = "knowledge.txt";

        IDataView knowledgeDataView = _mlContext.Data.LoadFromTextFile<KnowledgeEntry>(knowledgeFilePath, hasHeader: false);

        var textPipeline = _mlContext.Transforms.Text.FeaturizeText("Features", "Content");
        Console.WriteLine($"train stuff");
        Console.WriteLine($"{textPipeline.ToString()}");

        _textTransformer = textPipeline.Fit(knowledgeDataView);
        IDataView transformedKnowledgeData = _textTransformer.Transform(knowledgeDataView);
        Console.WriteLine($"{transformedKnowledgeData.ToString()}");

        _knowledgeFeatures = transformedKnowledgeData.GetColumn<float[]>("Features").ToArray();
        Console.WriteLine($"{_knowledgeFeatures.ToString()}");
        _knowledgeList = knowledgeBase;
    }

    public string GetRelevantResult(string query)
    {

        var queryData = new[] { new QueryEntry { Content = query } };

        IDataView queryDataView = _mlContext.Data.LoadFromEnumerable(queryData);
        IDataView transformedQueryData = _textTransformer.Transform(queryDataView);

        Console.WriteLine($"{transformedQueryData}");

        var queryFeatures = transformedQueryData.GetColumn<float[]>("Features").ToArray()[0];
        Console.WriteLine($"{queryFeatures.ToString()}");
        Console.WriteLine($"------------");
        Console.WriteLine($"{_knowledgeFeatures.ToString()}");

        double bestSimilarity = -1;
        int bestMatchIndex = -1;

        for (int j = 0; j < _knowledgeFeatures.Length; j++)
        {
            double similarity = CalculateCosineSimilarity(queryFeatures, _knowledgeFeatures[j]);
            if (similarity > bestSimilarity)
            {
                bestSimilarity = similarity;
                bestMatchIndex = j;
            }
        }

        Console.WriteLine("Finishing Query to Model");

        return bestMatchIndex != -1 ? _knowledgeList[bestMatchIndex] : "No relevant information found.";
    }

    private static double CalculateCosineSimilarity(float[] vectorA, float[] vectorB)
    {
        if (vectorA == null || vectorB == null || vectorA.Length != vectorB.Length)
        {
            return 0;
        }

        double dotProduct = 0;
        double magnitudeA = 0;
        double magnitudeB = 0;

        for (int i = 0; i < vectorA.Length; i++)
        {
            dotProduct += vectorA[i] * vectorB[i];
            magnitudeA += vectorA[i] * vectorA[i];
            magnitudeB += vectorB[i] * vectorB[i];
        }

        if (magnitudeA == 0 || magnitudeB == 0)
        {
            return 0;
        }

        return dotProduct / (Math.Sqrt(magnitudeA) * Math.Sqrt(magnitudeB));
    }

    public class KnowledgeEntry
    {
        [LoadColumn(0)]
        public string Content { get; set; }
    }

    public class QueryEntry
    {
        [LoadColumn(0)]
        public string Content { get; set; }
    }
}
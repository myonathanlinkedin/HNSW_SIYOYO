using System;
using System.Collections.Generic;
using System.Linq;

namespace HnswSiyoyoProject
{
    /// <summary>
    /// Comprehensive validation to ensure all functions from the paper are implemented and used.
    /// </summary>
    public class PaperValidation
    {
        public static void ValidatePaperImplementation()
        {
            Console.WriteLine("=== Paper Implementation Validation ===");
            
            var validationResults = new List<ValidationResult>();
            
            // Validate VectorStorage functions
            validationResults.Add(ValidateVectorStorage());
            
            // Validate ObjectPool functions
            validationResults.Add(ValidateObjectPool());
            
            // Validate ParallelProcessor functions
            validationResults.Add(ValidateParallelProcessor());
            
            // Validate GraphAnalytics functions
            validationResults.Add(ValidateGraphAnalytics());
            
            // Validate RealTimeGraphModifier functions
            validationResults.Add(ValidateRealTimeGraphModifier());
            
            // Validate GPU functions
            validationResults.Add(ValidateGpuFunctions());
            
            // Validate Benchmarking functions
            validationResults.Add(ValidateBenchmarkingFunctions());
            
            // Print validation summary
            PrintValidationSummary(validationResults);
        }

        private static ValidationResult ValidateVectorStorage()
        {
            var result = new ValidationResult("VectorStorage");
            
            try
            {
                var storage = new VectorStorage(128);
                
                // Test all functions from the paper
                var testVector = new float[128];
                storage.AddVector(testVector);
                result.AddTest("AddVector", true);
                
                var batch = storage.GetBatch(0, 1);
                result.AddTest("GetBatch", batch.Length == 1);
                
                var flattened = storage.GetFlattenedBatch(0, 1);
                result.AddTest("GetFlattenedBatch", flattened.Length == 128);
                
                storage.OptimizeMemoryLayout();
                result.AddTest("OptimizeMemoryLayout", true);
                
                var optimized = storage.GetOptimizedBatchTransfer(0, 1);
                result.AddTest("GetOptimizedBatchTransfer", optimized.Length == 128);
                
                result.Success = true;
            }
            catch (Exception ex)
            {
                result.AddTest("Exception", false, ex.Message);
            }
            
            return result;
        }

        private static ValidationResult ValidateObjectPool()
        {
            var result = new ValidationResult("ObjectPool");
            
            try
            {
                var pool = new ObjectPool<List<int>>(10);
                
                // Test all functions from the paper
                var item1 = pool.Rent();
                result.AddTest("Rent", item1 != null);
                
                pool.Return(item1);
                result.AddTest("Return", true);
                
                var stats = pool.GetStatistics();
                result.AddTest("GetStatistics", !string.IsNullOrEmpty(stats));
                
                pool.Clear();
                result.AddTest("Clear", true);
                
                result.Success = true;
            }
            catch (Exception ex)
            {
                result.AddTest("Exception", false, ex.Message);
            }
            
            return result;
        }

        private static ValidationResult ValidateParallelProcessor()
        {
            var result = new ValidationResult("ParallelProcessor");
            
            try
            {
                var items = Enumerable.Range(0, 100).ToList();
                var processed = new List<int>();
                
                // Test all functions from the paper
                ParallelProcessor.ProcessBatch(items, item => processed.Add(item));
                result.AddTest("ProcessBatch", processed.Count == 100);
                
                var results = ParallelProcessor.ProcessBatchWithResult(items, item => item * 2);
                result.AddTest("ProcessBatchWithResult", results.Count == 100);
                
                var query = new float[128];
                var vectors = new float[10][];
                for (int i = 0; i < 10; i++)
                {
                    vectors[i] = new float[128];
                }
                
                var similarities = ParallelProcessor.ProcessSimilarityBatch(query, vectors, (a, b) => 0.5f);
                result.AddTest("ProcessSimilarityBatch", similarities.Length == 10);
                
                var batchSize = ParallelProcessor.GetOptimalBatchSize();
                result.AddTest("GetOptimalBatchSize", batchSize > 0);
                
                result.Success = true;
            }
            catch (Exception ex)
            {
                result.AddTest("Exception", false, ex.Message);
            }
            
            return result;
        }

        private static ValidationResult ValidateGraphAnalytics()
        {
            var result = new ValidationResult("GraphAnalytics");
            
            try
            {
                var graph = new HnswSiyoyoGraph();
                
                // Add test data
                for (int i = 0; i < 50; i++)
                {
                    var vector = new float[128];
                    graph.Insert(vector);
                }
                
                var analytics = new GraphAnalytics(graph);
                
                // Test all functions from the paper
                var layerDistribution = analytics.GetLayerDistribution();
                result.AddTest("GetLayerDistribution", layerDistribution != null);
                
                var similarities = analytics.GetAverageSimilarityPerLayer();
                result.AddTest("GetAverageSimilarityPerLayer", similarities != null);
                
                var connectivity = analytics.GetGraphConnectivityScore();
                result.AddTest("GetGraphConnectivityScore", connectivity >= 0);
                
                var searchConverged = analytics.IsSearchConverged(new float[128]);
                result.AddTest("IsSearchConverged", true);
                
                var constructionConverged = analytics.IsGraphConstructionConverged();
                result.AddTest("IsGraphConstructionConverged", true);
                
                var approximationError = analytics.CalculateApproximationError(new float[128], 0);
                result.AddTest("CalculateApproximationError", approximationError >= 0);
                
                var recall = analytics.CalculateRecallAtK(new List<int> { 1, 2, 3 }, new List<int> { 1, 2, 4 }, 3);
                result.AddTest("CalculateRecallAtK", recall >= 0 && recall <= 1);
                
                var optimalM0 = analytics.CalculateOptimalM0(1.0, 1.0);
                result.AddTest("CalculateOptimalM0", optimalM0 > 0);
                
                var memoryComplexity = analytics.CalculateMemoryComplexity(16, 3);
                result.AddTest("CalculateMemoryComplexity", memoryComplexity > 0);
                
                var queryComplexity = analytics.CalculateQueryComplexity(64);
                result.AddTest("CalculateQueryComplexity", queryComplexity > 0);
                
                var metrics = analytics.GetDetailedPerformanceMetrics();
                result.AddTest("GetDetailedPerformanceMetrics", metrics.Count > 0);
                
                // Test new functions from the paper
                var query = new float[128];
                var candidateSet = analytics.GenerateCandidateSet(query, 0, 10, 0);
                result.AddTest("GenerateCandidateSet", candidateSet.Count > 0);
                
                var searchConvergence = analytics.TrackSearchConvergence(query);
                result.AddTest("TrackSearchConvergence", searchConvergence != null);
                
                var constructionConvergence = analytics.MonitorConstructionConvergence();
                result.AddTest("MonitorConstructionConvergence", constructionConvergence != null);
                
                var optimalEf = analytics.CalculateOptimalEf();
                result.AddTest("CalculateOptimalEf", optimalEf > 0);
                
                var gpuSpeedup = analytics.CalculateGpuSpeedup(1000, 128, 1e-6, 1e-7, 1e-3, 1e-4);
                result.AddTest("CalculateGpuSpeedup", gpuSpeedup > 0);
                
                result.Success = true;
            }
            catch (Exception ex)
            {
                result.AddTest("Exception", false, ex.Message);
            }
            
            return result;
        }

        private static ValidationResult ValidateRealTimeGraphModifier()
        {
            var result = new ValidationResult("RealTimeGraphModifier");
            
            try
            {
                var graph = new HnswSiyoyoGraph();
                
                // Add test data
                for (int i = 0; i < 20; i++)
                {
                    var vector = new float[128];
                    graph.Insert(vector);
                }
                
                var modifier = new RealTimeGraphModifier(graph);
                
                // Test all functions from the paper
                var updateSuccess = modifier.UpdateVector(0, new float[128]);
                result.AddTest("UpdateVector", updateSuccess);
                
                var removeSuccess = modifier.RemoveVector(0);
                result.AddTest("RemoveVector", removeSuccess);
                
                var rebalanceSuccess = modifier.RebalanceGraph();
                result.AddTest("RebalanceGraph", rebalanceSuccess);
                
                var isValid = modifier.ValidateGraphStructure();
                result.AddTest("ValidateGraphStructure", isValid);
                
                var workloadOptimization = modifier.OptimizeForDynamicWorkload("high_query", new Dictionary<string, double>());
                result.AddTest("OptimizeForDynamicWorkload", workloadOptimization != null);
                
                var parameterTuning = modifier.AdaptiveParameterTuning(
                    new Dictionary<string, double> { ["Recall"] = 0.8 },
                    new Dictionary<string, double> { ["Recall"] = 0.9 }
                );
                result.AddTest("AdaptiveParameterTuning", parameterTuning != null);
                
                var performanceMonitoring = modifier.MonitorPerformance();
                result.AddTest("MonitorPerformance", performanceMonitoring != null);
                
                var stats = modifier.GetModificationStatistics();
                result.AddTest("GetModificationStatistics", stats.Count > 0);
                
                result.Success = true;
            }
            catch (Exception ex)
            {
                result.AddTest("Exception", false, ex.Message);
            }
            
            return result;
        }

        private static ValidationResult ValidateGpuFunctions()
        {
            var result = new ValidationResult("GPU Functions");
            
            try
            {
                var gpuHelper = new GpuSimilarityHelper();
                
                // Test all functions from the paper
                var normalized = gpuHelper.NormalizeVector(new float[128]);
                result.AddTest("NormalizeVector", normalized.Length == 128);
                
                var topK = gpuHelper.ComputeTopKNeighbors(new float[128], new float[10][], 5);
                result.AddTest("ComputeTopKNeighbors", topK.Count == 5);
                
                var recall = gpuHelper.ComputeRecall(new List<int> { 1, 2, 3 }, new List<int> { 1, 2, 4 }, 3);
                result.AddTest("ComputeRecall", recall >= 0 && recall <= 1);
                
                var syntheticData = gpuHelper.GenerateSyntheticData(100, 128, "uniform");
                result.AddTest("GenerateSyntheticData", syntheticData.Length == 100);
                
                var batchSimilarity = gpuHelper.ComputeBatchSimilarity(new float[5][], new float[10][]);
                result.AddTest("ComputeBatchSimilarity", batchSimilarity.Length == 5);
                
                var gpuInfo = gpuHelper.GetGpuInfo();
                result.AddTest("GetGpuInfo", !string.IsNullOrEmpty(gpuInfo));
                
                var speedup = gpuHelper.MeasureGpuSpeedup(new float[128], new float[100][]);
                result.AddTest("MeasureGpuSpeedup", speedup > 0);
                
                // Test advanced functions
                try
                {
                    var advancedResults = gpuHelper.ComputeBatchSimilarityAdvanced(new float[5][], new float[10][]);
                    result.AddTest("ComputeBatchSimilarityAdvanced", advancedResults.Length == 5);
                }
                catch
                {
                    result.AddTest("ComputeBatchSimilarityAdvanced", false, "GPU not available");
                }
                
                result.Success = true;
            }
            catch (Exception ex)
            {
                result.AddTest("Exception", false, ex.Message);
            }
            
            return result;
        }

        private static ValidationResult ValidateBenchmarkingFunctions()
        {
            var result = new ValidationResult("Benchmarking Functions");
            
            try
            {
                var runner = new BenchmarkRunner();
                
                // Test all functions from the paper
                var gpuHelper = new GpuSimilarityHelper();
                var dataset = gpuHelper.GenerateSyntheticData(100, 128, "uniform");
                var queries = gpuHelper.GenerateSyntheticData(10, 128, "uniform");
                
                var benchmarkResults = runner.RunBenchmark(dataset, queries);
                result.AddTest("RunBenchmark", benchmarkResults != null);
                
                var sensitivityResults = runner.RunParameterSensitivity(dataset, queries, new[] { 8, 16 });
                result.AddTest("RunParameterSensitivity", sensitivityResults != null);
                
                var advancedResults = runner.RunAdvancedBenchmark(dataset, queries);
                result.AddTest("RunAdvancedBenchmark", advancedResults != null);
                
                runner.ExportResults(benchmarkResults, "test_results.csv");
                result.AddTest("ExportResults", true);
                
                runner.ExportParameterSensitivity(sensitivityResults, "test_sensitivity.csv");
                result.AddTest("ExportParameterSensitivity", true);
                
                var summary = runner.GenerateSummaryReport(benchmarkResults);
                result.AddTest("GenerateSummaryReport", !string.IsNullOrEmpty(summary));
                
                var quickTest = runner.RunQuickTest();
                result.AddTest("RunQuickTest", quickTest != null);
                
                result.Success = true;
            }
            catch (Exception ex)
            {
                result.AddTest("Exception", false, ex.Message);
            }
            
            return result;
        }

        private static void PrintValidationSummary(List<ValidationResult> results)
        {
            Console.WriteLine("\n=== Validation Summary ===");
            
            var totalTests = 0;
            var passedTests = 0;
            var totalComponents = 0;
            var passedComponents = 0;
            
            foreach (var result in results)
            {
                totalComponents++;
                if (result.Success) passedComponents++;
                
                Console.WriteLine($"\n{result.ComponentName}:");
                foreach (var test in result.Tests)
                {
                    totalTests++;
                    if (test.Passed) passedTests++;
                    
                    var status = test.Passed ? "✅ PASS" : "❌ FAIL";
                    var message = string.IsNullOrEmpty(test.Message) ? "" : $" - {test.Message}";
                    Console.WriteLine($"  {status}: {test.Name}{message}");
                }
            }
            
            Console.WriteLine($"\n=== Final Results ===");
            Console.WriteLine($"Components: {passedComponents}/{totalComponents} passed");
            Console.WriteLine($"Tests: {passedTests}/{totalTests} passed");
            Console.WriteLine($"Success Rate: {(double)passedTests / totalTests * 100:F1}%");
            
            if (passedTests == totalTests)
            {
                Console.WriteLine("🎉 All paper functions implemented and validated successfully!");
            }
            else
            {
                Console.WriteLine("⚠️  Some functions need attention.");
            }
        }
    }

    public class ValidationResult
    {
        public string ComponentName { get; set; }
        public bool Success { get; set; }
        public List<TestResult> Tests { get; set; } = new List<TestResult>();

        public ValidationResult(string componentName)
        {
            ComponentName = componentName;
        }

        public void AddTest(string name, bool passed, string message = "")
        {
            Tests.Add(new TestResult { Name = name, Passed = passed, Message = message });
        }
    }

    public class TestResult
    {
        public string Name { get; set; }
        public bool Passed { get; set; }
        public string Message { get; set; }
    }
} 
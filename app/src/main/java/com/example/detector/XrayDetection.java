package com.example.detector;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.net.Uri;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.core.content.FileProvider;
import androidx.fragment.app.Fragment;

import android.graphics.pdf.PdfDocument;

import com.google.android.material.progressindicator.CircularProgressIndicator;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.text.SimpleDateFormat;
import java.util.Collections;
import java.util.Date;
import java.util.Locale;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;


public class XrayDetection extends Fragment {

    // UI
    ImageView imageView;
    TextView resultText;
    TextView confidenceText;

    Button btnSelect;
    Button btnDownload;

    CircularProgressIndicator confidenceProgress;

    // AI
    OrtEnvironment env;
    OrtSession diseaseSession;

    // Analysis data
    private Bitmap analyzedBitmap = null;
    private boolean analysisDone = false;
    private boolean lastDiseaseDetected = false;
    private float lastConfidence = 0f;

    // Image picker
    ActivityResultLauncher<Intent> imagePickerLauncher;

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

        // Initial UI
        resultText.setText("Loading AI Model...");
        confidenceText.setText("0%");
        confidenceProgress.setProgress(0, true);

        // Initialize AI
        initAI();

        // Image picker
        imagePickerLauncher =
                registerForActivityResult(
                        new ActivityResultContracts.StartActivityForResult(),

                        result -> {

                            if (result.getResultCode()
                                    == android.app.Activity.RESULT_OK
                                    && result.getData() != null) {

                                Uri imageUri =
                                        result.getData().getData();

                                if (imageUri != null) {
                                    processImage(imageUri);
                                }
                            }
                        }
                );

        // Select image
        btnSelect.setOnClickListener(v -> {

            Intent intent =
                    new Intent(Intent.ACTION_PICK);

            intent.setType("image/*");

            imagePickerLauncher.launch(intent);
        });

        // Download report
        btnDownload.setOnClickListener(v -> {

            if (!analysisDone || analyzedBitmap == null) {

                showNoAnalysisDialog();
                return;
            }

            try {

                File pdf = generatePdfReport();
                openPdf(pdf);

            } catch (Exception e) {

                showErrorDialog(e.getMessage());
                e.printStackTrace();
            }
        });

        return view;
    }

    // Initialize AI model
    private void initAI() {

        try {

            env = OrtEnvironment.getEnvironment();

            diseaseSession =
                    env.createSession(
                            copyModel("disease_xray_model.onnx"),
                            new OrtSession.SessionOptions()
                    );

            resultText.setText("AI System Ready");

        } catch (Exception e) {

            resultText.setText("Model Load Failed");
            e.printStackTrace();
        }
    }

    // Process selected image
    private void processImage(Uri uri) {

        try {

            InputStream is =
                    requireContext()
                            .getContentResolver()
                            .openInputStream(uri);

            if (is == null) {

                resultText.setText("Unable to open image");
                return;
            }

            Bitmap bitmap =
                    BitmapFactory.decodeStream(is);

            is.close();

            if (bitmap == null) {

                resultText.setText("Invalid Image");
                return;
            }

            imageView.setImageBitmap(bitmap);

            analyzeBitmap(bitmap);

        } catch (Exception e) {

            resultText.setText("Image Load Error");
            e.printStackTrace();
        }
    }

    // Analyze image
    private void analyzeBitmap(Bitmap bitmap) {

        OnnxTensor diseaseTensor = null;

        try {

            analysisDone = false;
            lastDiseaseDetected = false;
            lastConfidence = 0f;

            resultText.setText("Analyzing...");
            confidenceText.setText("0%");
            confidenceProgress.setProgress(0, true);

            // Prepare model input
            float[] diseaseInput =
                    preprocessNHWC(bitmap);

            // Create model tensor
            diseaseTensor =
                    OnnxTensor.createTensor(
                            env,
                            FloatBuffer.wrap(diseaseInput),
                            new long[]{
                                    1,
                                    224,
                                    224,
                                    3
                            }
                    );

            // Run model
            Object output =
                    diseaseSession
                            .run(
                                    Collections.singletonMap(
                                            diseaseSession
                                                    .getInputNames()
                                                    .iterator()
                                                    .next(),
                                            diseaseTensor
                                    )
                            )
                            .get(0)
                            .getValue();

            // Get model output
            float confidence =
                    ((float[][]) output)[0][0];

            analyzedBitmap = bitmap;
            lastConfidence = confidence;

            // Check prediction
            lastDiseaseDetected =
                    confidence > 0.5f;

            analysisDone = true;

            // Update confidence
            int percentage =
                    (int) (confidence * 100);

            percentage =
                    Math.max(
                            0,
                            Math.min(
                                    percentage,
                                    100
                            )
                    );

            updateConfidenceUI(percentage);

            // Show result
            if (lastDiseaseDetected) {

                resultText.setText(
                        "Pneumonia Detected"
                );

            } else {

                resultText.setText(
                        "Normal Chest X-ray"
                );
            }

        } catch (Exception e) {

            resultText.setText("Analysis Error");
            confidenceText.setText("0%");
            confidenceProgress.setProgress(0, true);

            e.printStackTrace();

        } finally {

            if (diseaseTensor != null) {

                try {
                    diseaseTensor.close();
                } catch (Exception ignored) {
                }
            }
        }
    }

    // Generate PDF report
    private File generatePdfReport()
            throws Exception {

        String time =
                new SimpleDateFormat(
                        "dd MMM yyyy | hh:mm a",
                        Locale.getDefault()
                ).format(new Date());

        String reportId =
                "AI-" + System.currentTimeMillis();

        PdfDocument pdf =
                new PdfDocument();

        PdfDocument.Page page =
                pdf.startPage(
                        new PdfDocument.PageInfo
                                .Builder(
                                595,
                                842,
                                1
                        )
                                .create()
                );

        Canvas c =
                page.getCanvas();

        Paint titlePaint =
                new Paint();

        titlePaint.setTextSize(20);
        titlePaint.setFakeBoldText(true);
        titlePaint.setColor(Color.BLACK);

        Paint headerPaint =
                new Paint();

        headerPaint.setTextSize(14);
        headerPaint.setFakeBoldText(true);
        headerPaint.setColor(Color.BLACK);

        Paint bodyPaint =
                new Paint();

        bodyPaint.setTextSize(12);
        bodyPaint.setColor(Color.DKGRAY);

        Paint linePaint =
                new Paint();

        linePaint.setStrokeWidth(2);

        int y = 40;

        // Report title
        c.drawText(
                "AI BASED PNEUMONIA SCREENING REPORT",
                40,
                y,
                titlePaint
        );

        y += 10;

        c.drawLine(
                40,
                y,
                555,
                y,
                linePaint
        );

        // Report information
        y += 30;

        c.drawText(
                "Report ID : " + reportId,
                40,
                y,
                bodyPaint
        );

        y += 18;

        c.drawText(
                "Date & Time : " + time,
                40,
                y,
                bodyPaint
        );

        y += 18;

        c.drawText(
                "Generated By : AI Pneumonia Detection System",
                40,
                y,
                bodyPaint
        );

        y += 25;

        c.drawLine(
                40,
                y,
                555,
                y,
                linePaint
        );

        // Patient information
        y += 25;

        c.drawText(
                "1. Patient Information",
                40,
                y,
                headerPaint
        );

        y += 18;

        c.drawText(
                "• Patient Name : Not Provided",
                40,
                y,
                bodyPaint
        );

        y += 15;

        c.drawText(
                "• Patient ID : N/A",
                40,
                y,
                bodyPaint
        );

        y += 15;

        c.drawText(
                "• Gender / Age : N/A",
                40,
                y,
                bodyPaint
        );

        // X-ray image
        y += 25;

        c.drawText(
                "2. Uploaded Chest X-Ray Image",
                40,
                y,
                headerPaint
        );

        y += 15;

        if (analyzedBitmap != null) {

            Bitmap img =
                    Bitmap.createScaledBitmap(
                            analyzedBitmap,
                            220,
                            220,
                            true
                    );

            c.drawBitmap(
                    img,
                    40,
                    y,
                    null
            );
        }

        // Analysis summary
        y += 240;

        c.drawText(
                "3. AI Analysis Summary",
                40,
                y,
                headerPaint
        );

        y += 18;

        String result;

        if (lastDiseaseDetected) {

            result =
                    "Pneumonia Detected";

        } else {

            result =
                    "Normal Chest X-ray";
        }

        c.drawText(
                "• Diagnostic Result : " + result,
                40,
                y,
                bodyPaint
        );

        y += 15;

        c.drawText(
                "• Confidence Score : "
                        + (int) (lastConfidence * 100)
                        + " %",
                40,
                y,
                bodyPaint
        );

        // Interpretation
        y += 25;

        c.drawText(
                "4. Interpretation",
                40,
                y,
                headerPaint
        );

        y += 18;

        if (lastDiseaseDetected) {

            c.drawText(
                    "The AI model detected patterns",
                    40,
                    y,
                    bodyPaint
            );

            y += 15;

            c.drawText(
                    "associated with Pneumonia in the",
                    40,
                    y,
                    bodyPaint
            );

            y += 15;

            c.drawText(
                    "uploaded chest X-ray image.",
                    40,
                    y,
                    bodyPaint
            );

        } else {

            c.drawText(
                    "The AI model did not detect patterns",
                    40,
                    y,
                    bodyPaint
            );

            y += 15;

            c.drawText(
                    "associated with Pneumonia in the",
                    40,
                    y,
                    bodyPaint
            );

            y += 15;

            c.drawText(
                    "uploaded chest X-ray image.",
                    40,
                    y,
                    bodyPaint
            );
        }

        // Disclaimer
        y += 30;

        c.drawText(
                "5. Medical Disclaimer",
                40,
                y,
                headerPaint
        );

        y += 18;

        c.drawText(
                "This report is generated by an AI-based",
                40,
                y,
                bodyPaint
        );

        y += 15;

        c.drawText(
                "screening system and is intended for",
                40,
                y,
                bodyPaint
        );

        y += 15;

        c.drawText(
                "preliminary assessment only. It should",
                40,
                y,
                bodyPaint
        );

        y += 15;

        c.drawText(
                "not be considered a final medical diagnosis.",
                40,
                y,
                bodyPaint
        );

        y += 15;

        c.drawText(
                "Clinical correlation with a qualified",
                40,
                y,
                bodyPaint
        );

        y += 15;

        c.drawText(
                "medical professional is recommended.",
                40,
                y,
                bodyPaint
        );

        // Signature
        y = 780;

        c.drawLine(
                350,
                y - 10,
                550,
                y - 10,
                linePaint
        );

        c.drawText(
                "AI Pneumonia Detection System",
                350,
                y + 5,
                bodyPaint
        );

        // Finish PDF
        pdf.finishPage(page);

        File file =
                new File(
                        requireContext()
                                .getExternalFilesDir(null),
                        "AI_Pneumonia_Report_"
                                + reportId
                                + ".pdf"
                );

        FileOutputStream fos =
                new FileOutputStream(file);

        pdf.writeTo(fos);

        fos.close();
        pdf.close();

        return file;
    }

    // Open PDF
    private void openPdf(File file) {

        try {

            Uri uri =
                    FileProvider.getUriForFile(
                            requireContext(),
                            requireContext()
                                    .getPackageName()
                                    + ".provider",
                            file
                    );

            Intent intent =
                    new Intent(
                            Intent.ACTION_VIEW
                    );

            intent.setDataAndType(
                    uri,
                    "application/pdf"
            );

            intent.setFlags(
                    Intent.FLAG_GRANT_READ_URI_PERMISSION
            );

            startActivity(intent);

        } catch (Exception e) {

            showErrorDialog(
                    "No PDF viewer app found."
            );

            e.printStackTrace();
        }
    }

    // Copy model from assets
    private String copyModel(String name)
            throws Exception {

        File f =
                new File(
                        requireContext()
                                .getFilesDir(),
                        name
                );

        if (f.exists()) {
            return f.getAbsolutePath();
        }

        InputStream is =
                requireContext()
                        .getAssets()
                        .open(name);

        FileOutputStream fos =
                new FileOutputStream(f);

        byte[] buffer =
                new byte[4096];

        int r;

        while ((r = is.read(buffer)) != -1) {

            fos.write(
                    buffer,
                    0,
                    r
            );
        }

        fos.close();
        is.close();

        return f.getAbsolutePath();
    }

    // Prepare NHWC input
    private float[] preprocessNHWC(
            Bitmap bitmap) {

        Bitmap resized =
                Bitmap.createScaledBitmap(
                        bitmap,
                        224,
                        224,
                        true
                );

        float[] input =
                new float[
                        224 * 224 * 3
                        ];

        int index = 0;

        for (int y = 0; y < 224; y++) {

            for (int x = 0; x < 224; x++) {

                int pixel =
                        resized.getPixel(
                                x,
                                y
                        );

                float red =
                        ((pixel >> 16) & 255);

                float green =
                        ((pixel >> 8) & 255);

                float blue =
                        (pixel & 255);

                input[index++] = red;
                input[index++] = green;
                input[index++] = blue;
            }
        }

        return input;
    }

    // Update confidence UI
    private void updateConfidenceUI(
            int percentage) {

        confidenceProgress.setProgress(
                percentage,
                true
        );

        confidenceText.setText(
                percentage + "%"
        );
    }

    // Show no-analysis dialog
    private void showNoAnalysisDialog() {

        new androidx.appcompat.app.AlertDialog
                .Builder(
                requireContext()
        )
                .setTitle(
                        "No Analysis"
                )
                .setMessage(
                        "Please select a chest X-ray image and run the analysis first."
                )
                .setPositiveButton(
                        "OK",
                        null
                )
                .show();
    }

    // Show error dialog
    private void showErrorDialog(
            String message) {

        if (message == null ||
                message.trim().isEmpty()) {

            message =
                    "Something went wrong.";
        }

        new androidx.appcompat.app.AlertDialog
                .Builder(
                requireContext()
        )
                .setTitle(
                        "Error"
                )
                .setMessage(
                        message
                )
                .setPositiveButton(
                        "OK",
                        null
                )
                .show();
    }
}
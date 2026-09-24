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
import android.graphics.pdf.PdfDocument;
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

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;


public class XrayDetection extends Fragment {

    // ================= UI =================

    ImageView imageView;
    TextView resultText;
    TextView confidenceText;

    Button btnSelect;
    Button btnDownload;

    CircularProgressIndicator confidenceProgress;


    // AI

    OrtEnvironment env;
    OrtSession diseaseSession;


    // ANALYSIS DATA

    private Bitmap analyzedBitmap = null;

    private boolean analysisDone = false;

    private boolean lastDiseaseDetected = false;

    private float lastConfidence = 0f;


    // IMAGE PICKER

    ActivityResultLauncher<Intent> imagePickerLauncher;

    // ON CREATE VIEW

    @Nullable
    @Override
    public View onCreateView(
            @NonNull LayoutInflater inflater,
            @Nullable ViewGroup container,
            @Nullable Bundle savedInstanceState) {

        // Load fragment layout
        View view = inflater.inflate(
                R.layout.fragment_xray_detection,
                container,
                false
        );


        // FIND VIEWS

        imageView = view.findViewById(R.id.imageView);

        resultText = view.findViewById(R.id.resultText);

        confidenceText = view.findViewById(R.id.confidenceText);

        confidenceProgress =
                view.findViewById(R.id.confidenceProgress);

        btnSelect =
                view.findViewById(R.id.btnSelect);

        btnDownload =
                view.findViewById(R.id.btndownload);


        // INITIAL UI

        resultText.setText("Loading AI Model...");

        confidenceText.setText("0%");

        confidenceProgress.setProgress(0, true);


        // INITIALIZE AI

        initAI();


        // IMAGE PICKER

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


        // SELECT IMAGE

        btnSelect.setOnClickListener(v -> {

            Intent intent =
                    new Intent(Intent.ACTION_PICK);

            intent.setType("image/*");

            imagePickerLauncher.launch(intent);
        });


        // DOWNLOAD REPORT

        btnDownload.setOnClickListener(v -> {

            if (!analysisDone || analyzedBitmap == null) {

                showNoAnalysisDialog();

                return;
            }


            try {

                File pdf =
                        generatePdfReport();

                openPdf(pdf);

            } catch (Exception e) {

                showErrorDialog(
                        e.getMessage()
                );

                e.printStackTrace();
            }
        });


        return view;
    }

    // INITIALIZE AI MODEL

    private void initAI() {

        try {

            // Create ONNX environment
            env =
                    OrtEnvironment.getEnvironment();


            // Load ONLY pneumonia model
            diseaseSession =
                    env.createSession(
                            copyModel(
                                    "disease_xray_model.onnx"
                            ),
                            new OrtSession.SessionOptions()
                    );


            // Model loaded successfully
            resultText.setText(
                    "AI System Ready"
            );


        } catch (Exception e) {

            resultText.setText(
                    "Model Load Failed"
            );

            e.printStackTrace();
        }
    }

    // PROCESS IMAGE

    private void processImage(Uri uri) {

        try {

            InputStream is =
                    requireContext()
                            .getContentResolver()
                            .openInputStream(uri);


            if (is == null) {

                resultText.setText(
                        "Unable to open image"
                );

                return;
            }


            Bitmap bitmap =
                    BitmapFactory.decodeStream(is);


            is.close();


            if (bitmap == null) {

                resultText.setText(
                        "Invalid Image"
                );

                return;
            }


            // Show selected image
            imageView.setImageBitmap(bitmap);


            // Analyze image
            analyzeBitmap(bitmap);


        } catch (Exception e) {

            resultText.setText(
                    "Image Load Error"
            );

            e.printStackTrace();
        }
    }

    // ANALYZE IMAGE

    private void analyzeBitmap(Bitmap bitmap) {

        try {

            // Reset previous analysis
            analysisDone = false;

            lastDiseaseDetected = false;

            lastConfidence = 0f;


            // Show analyzing state
            resultText.setText(
                    "Analyzing..."
            );

            confidenceText.setText(
                    "0%"
            );

            confidenceProgress.setProgress(
                    0,
                    true
            );

            // PREPROCESS IMAGE

            float[] diseaseInput =
                    preprocessNCHW(bitmap);

            // CREATE ONNX TENSOR

            OnnxTensor diseaseTensor =
                    OnnxTensor.createTensor(
                            env,
                            FloatBuffer.wrap(
                                    diseaseInput
                            ),
                            new long[]{
                                    1,
                                    3,
                                    224,
                                    224
                            }
                    );

            // RUN MODEL

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

            // GET MODEL OUTPUT

            float confidence =
                    ((float[][]) output)[0][0];


            // SAVE RESULTS
            analyzedBitmap = bitmap;

            lastConfidence = confidence;


            // Pneumonia if confidence > 0.5
            lastDiseaseDetected =
                    confidence > 0.5f;


            analysisDone = true;

            // UPDATE CONFIDENCE UI

            int percentage =
                    (int) (confidence * 100);


            // Keep value between 0 and 100
            percentage =
                    Math.max(
                            0,
                            Math.min(
                                    percentage,
                                    100
                            )
                    );


            updateConfidenceUI(
                    percentage
            );

            // SHOW RESULT

            if (lastDiseaseDetected) {

                resultText.setText(
                        "Pneumonia Detected"
                );

            } else {

                resultText.setText(
                        "Normal Chest X-ray"
                );
            }


            // Close tensor
            diseaseTensor.close();


        } catch (Exception e) {

            resultText.setText(
                    "Analysis Error"
            );

            confidenceText.setText(
                    "0%"
            );

            confidenceProgress.setProgress(
                    0,
                    true
            );

            e.printStackTrace();
        }
    }

    // GENERATE PDF REPORT

    private File generatePdfReport()
            throws Exception {


        // Date and time
        String time =
                new SimpleDateFormat(
                        "dd MMM yyyy | hh:mm a",
                        Locale.getDefault()
                ).format(
                        new Date()
                );


        // Unique report ID
        String reportId =
                "AI-" +
                        System.currentTimeMillis();


        // Create PDF
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

        // PAINTS

        Paint titlePaint =
                new Paint();

        titlePaint.setTextSize(20);

        titlePaint.setFakeBoldText(true);

        titlePaint.setColor(
                Color.BLACK
        );


        Paint headerPaint =
                new Paint();

        headerPaint.setTextSize(14);

        headerPaint.setFakeBoldText(true);

        headerPaint.setColor(
                Color.BLACK
        );


        Paint bodyPaint =
                new Paint();

        bodyPaint.setTextSize(12);

        bodyPaint.setColor(
                Color.DKGRAY
        );


        Paint linePaint =
                new Paint();

        linePaint.setStrokeWidth(2);


        int y = 40;


        // REPORT TITLE

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


        // REPORT META

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


        // PATIENT INFORMATION

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

        // IMAGE

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
        // ANALYSIS

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
                        + (int)
                        (lastConfidence * 100)
                        + " %",
                40,
                y,
                bodyPaint
        );

        // INTERPRETATION

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
        // DISCLAIMER

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
        // SIGNATURE

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
        // FINISH PDF

        pdf.finishPage(page);

        // SAVE FILE

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
    // OPEN PDF

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

    // COPY MODEL FROM ASSETS

    private String copyModel(String name)
            throws Exception {


        File f =
                new File(
                        requireContext()
                                .getFilesDir(),
                        name
                );


        // Already copied
        if (f.exists()) {

            return f.getAbsolutePath();
        }


        // Open model from assets
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

    // IMAGE PREPROCESSING

    private float[] preprocessNCHW(
            Bitmap bitmap) {


        // Resize image to 224 x 224
        Bitmap resized =
                Bitmap.createScaledBitmap(
                        bitmap,
                        224,
                        224,
                        true
                );


        // 3 channels × 224 × 224
        float[] input =
                new float[
                        3 * 224 * 224
                        ];


        int index = 0;


        // Channel first
        for (int c = 0; c < 3; c++) {

            for (int y = 0; y < 224; y++) {

                for (int x = 0; x < 224; x++) {


                    int pixel =
                            resized.getPixel(
                                    x,
                                    y
                            );


                    float value;


                    if (c == 0) {

                        // Red
                        value =
                                ((pixel >> 16)
                                        & 255);

                    } else if (c == 1) {

                        // Green
                        value =
                                ((pixel >> 8)
                                        & 255);

                    } else {

                        // Blue
                        value =
                                (pixel & 255);
                    }


                    // Normalize 0-255 → 0-1
                    input[index++] =
                            value / 255f;
                }
            }
        }


        return input;
    }

    // UPDATE CONFIDENCE UI

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
    // NO ANALYSIS DIALOG

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
    // ERROR DIALOG

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
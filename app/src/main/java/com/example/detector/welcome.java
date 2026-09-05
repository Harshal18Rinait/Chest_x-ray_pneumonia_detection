package com.example.detector;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.AppCompatButton;

public class welcome extends AppCompatActivity {

    AppCompatButton startbtn;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // ✅ FIRST load layout
        setContentView(R.layout.activity_welcome);

        // ✅ THEN bind views
        startbtn = findViewById(R.id.getStarted);

        // ✅ THEN use the view
        startbtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(welcome.this, MainActivity.class);
                startActivity(intent);
            }
        });
    }
}

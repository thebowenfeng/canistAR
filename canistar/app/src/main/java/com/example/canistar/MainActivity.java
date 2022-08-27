package com.example.canistar;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.os.SystemClock;
import android.view.View;
import android.widget.Button;
import android.widget.ImageButton;

import com.google.android.material.snackbar.Snackbar;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Button btnLoginGuest = findViewById(R.id.btnLoginGuest);
        ImageButton btnLoginGoogle = findViewById(R.id.btnLoginGoogle);

        btnLoginGuest.setAlpha(0f);
        btnLoginGoogle.setAlpha(0f);

        btnLoginGuest.setEnabled(false);
        btnLoginGoogle.setEnabled(false);

        Thread t1 = new Thread(() -> {
            SystemClock.sleep(1000);
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    btnLoginGoogle.animate().alpha(1.0f).setDuration(3000).start();
                    btnLoginGuest.animate().alpha(1.0f).setDuration(3000).start();
                    btnLoginGoogle.setEnabled(true);
                    btnLoginGuest.setEnabled(true);
                }
            });
        });
        t1.start();

        btnLoginGoogle.setOnClickListener((View view) -> {
            Snackbar snack = Snackbar.make(view, "Button clicked", Snackbar.LENGTH_SHORT);
            snack.show();
        });

        btnLoginGuest.setOnClickListener((View view) -> {
            Snackbar snack = Snackbar.make(view, "Logged in as guest", Snackbar.LENGTH_SHORT);
            snack.show();
            Intent i = new Intent(this, ARPage.class);
            startActivity(i);
        });
    }
}
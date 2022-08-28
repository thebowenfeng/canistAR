package com.example.canistar;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.content.IntentSender;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageButton;

import com.google.android.gms.auth.api.Auth;
import com.google.android.gms.auth.api.identity.BeginSignInRequest;
import com.google.android.gms.auth.api.identity.BeginSignInResult;
import com.google.android.gms.auth.api.identity.Identity;
import com.google.android.gms.auth.api.identity.SignInClient;
import com.google.android.gms.auth.api.identity.SignInCredential;
import com.google.android.gms.auth.api.signin.GoogleSignIn;
import com.google.android.gms.auth.api.signin.GoogleSignInClient;
import com.google.android.gms.auth.api.signin.GoogleSignInOptions;
import com.google.android.gms.common.api.GoogleApiClient;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.material.snackbar.Snackbar;
import com.google.firebase.auth.FirebaseAuth;

public class MainActivity extends AppCompatActivity {
    private FirebaseAuth mAuth;
    private GoogleSignInOptions gso;
    private GoogleSignInClient gsc;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mAuth = FirebaseAuth.getInstance();

        gso = new GoogleSignInOptions.Builder(GoogleSignInOptions.DEFAULT_SIGN_IN)
                .requestEmail().build();

        gsc = GoogleSignIn.getClient(this, gso);

        Button btnLoginGuest = findViewById(R.id.btnLoginGuest);
        ImageButton btnLoginGoogle = findViewById(R.id.btnLoginGoogle);

        btnLoginGuest.setAlpha(0f);
        btnLoginGoogle.setAlpha(0f);

        btnLoginGuest.setEnabled(false);
        btnLoginGoogle.setEnabled(false);

        Thread t1 = new Thread(() -> {
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    btnLoginGoogle.animate().alpha(1.0f).setDuration(3000).start();
                    btnLoginGuest.animate().alpha(1.0f).setDuration(3000).start();
                }
            });
            SystemClock.sleep(3000);
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    btnLoginGoogle.setEnabled(true);
                    btnLoginGuest.setEnabled(true);
                }
            });
        });
        t1.start();

        btnLoginGoogle.setOnClickListener((View view) -> {
            Intent i = gsc.getSignInIntent();
            startActivityForResult(i, 100);
        });

        btnLoginGuest.setOnClickListener((View view) -> {
            Snackbar snack = Snackbar.make(view, "Logged in as guest", Snackbar.LENGTH_SHORT);
            snack.show();
            Intent i = new Intent(this, ARPage.class);
            startActivity(i);
        });
    }

    @Override
    public void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        Intent i = new Intent(this, ARPage.class);
        startActivity(i);
    }
}
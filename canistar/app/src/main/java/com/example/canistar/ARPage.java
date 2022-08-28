package com.example.canistar;
import android.app.Activity;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageButton;

import androidx.activity.result.ActivityResult;
import androidx.activity.result.ActivityResultCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;

import com.google.ar.sceneform.rendering.Material;
import com.google.ar.core.Anchor;
import com.google.ar.core.Config;
import com.google.ar.core.Frame;
import com.google.ar.core.HitResult;
import com.google.ar.core.InstantPlacementPoint;
import com.google.ar.sceneform.AnchorNode;
import com.google.ar.sceneform.ArSceneView;
import com.google.ar.sceneform.math.Vector3;
import com.google.ar.sceneform.rendering.ModelRenderable;
import com.google.ar.sceneform.rendering.ShapeFactory;
import com.google.ar.sceneform.rendering.Texture;
import com.google.ar.sceneform.ux.ArFragment;
import com.google.ar.sceneform.ux.TransformableNode;
import com.gorisse.thomas.sceneform.light.LightEstimationConfig;

import java.util.List;

public class ARPage extends AppCompatActivity {
    private Material unlitMaterial;
    private double prevTimestamp = 0;
    private ImageButton btnImage;
    private String currImgUri = "";

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.ar_page);

        ArFragment arfragment = (ArFragment) getSupportFragmentManager().findFragmentById(R.id.arFragment);
        assert arfragment != null;

        arfragment.setOnSessionConfigurationListener(
                (session, config) -> {
                    config.setInstantPlacementMode(Config.InstantPlacementMode.LOCAL_Y_UP);
                }
        );

        arfragment.setOnViewCreatedListener(
                (ArSceneView view) -> {
                    view.getScene().setOnTouchListener(
                            (v, event) -> {
                                Frame frame = view.getArFrame();
                                assert frame != null;
                                boolean isValidTap = false;

                                double currTimestamp = frame.getTimestamp() / Math.pow(10, 9);
                                if(prevTimestamp == 0){
                                    prevTimestamp = currTimestamp;
                                    isValidTap = true;
                                }else{
                                    if(currTimestamp - prevTimestamp > 2){
                                        isValidTap = true;
                                        prevTimestamp = currTimestamp;
                                    }
                                }

                                if(isValidTap){
                                    List<HitResult> results = frame.hitTestInstantPlacement(event.getX(), event.getY(), 1.0f);

                                    if(!results.isEmpty() && !currImgUri.equals("")){
                                        InstantPlacementPoint point = (InstantPlacementPoint) results.get(0).getTrackable();
                                        Anchor anchor = point.createAnchor(point.getPose());
                                        placeObject(arfragment, anchor);
                                    }
                                }
                                return true;
                            }
                    );
                }
        );

        ActivityResultLauncher<Intent> launcher = registerForActivityResult(
                new ActivityResultContracts.StartActivityForResult(),
                new ActivityResultCallback<ActivityResult>() {
                    @Override
                    public void onActivityResult(ActivityResult result) {
                        if(result.getResultCode() == Activity.RESULT_OK){
                            Intent data = result.getData();
                            currImgUri = data.getDataString();
                        }
                    }
                }
        );

        this.btnImage = findViewById(R.id.btnImage);
        btnImage.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v){
                Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
                intent.addCategory(Intent.CATEGORY_OPENABLE);
                intent.setType("image/*");

                launcher.launch(intent);
            }
        });
    }

    public void placeObject(ArFragment arfragment, Anchor anchor){
        arfragment.getArSceneView()._lightEstimationConfig = LightEstimationConfig.SPECTACULAR;

        Texture.builder().setSource(this, Uri.parse(currImgUri)).build().thenAccept(res1 -> {
            Material.builder()
                    .setSource(this, R.raw.unlit)
                    .build()
                    .thenAccept((material) -> {
                        ModelRenderable plane = ShapeFactory.makeCube(
                                new Vector3(0.5f, .0f, 0.5f), // size
                                new Vector3(0.0f, 0.0f, 0.0f), // center
                                material);

                        material.setTexture("imageTexture", res1);

                        plane.setShadowCaster(false);
                        plane.setShadowCaster(false);

                        AnchorNode anchorNode = new AnchorNode(anchor);
                        TransformableNode node = new TransformableNode(arfragment.getTransformationSystem());
                        node.setRenderable(plane);
                        node.setParent(anchorNode);

                        arfragment.getArSceneView().getScene().addChild(anchorNode);
                        node.select();
                    })
                    .exceptionally( throwable -> {
                        Log.d("Error", throwable.getMessage());
                        return null;
                    });
        });
    }
}

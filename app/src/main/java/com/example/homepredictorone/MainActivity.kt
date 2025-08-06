package com.example.homepredictorone

import android.media.MediaPlayer
import android.os.Bundle
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import android.widget.Button
import android.widget.EditText
import android.widget.Toast
import android.widget.TextView


import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OnnxTensor
import android.media.MediaSession2Service
import java.nio.FloatBuffer




class MainActivity : AppCompatActivity() {

    private var mediaPlayer: MediaPlayer? = null

/*
*
    override fun onCreate(savedInstanceState: Bundle?){
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets



        }
    }
 */




    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }
        val inputArea = findViewById<EditText>(R.id.inputnum1)
        val inputRooms = findViewById<EditText>(R.id.inputnum2)

        val pressToPredict = findViewById<Button>(R.id.predict_button)
        val resultText = findViewById<EditText>(R.id.outputnum1)

        pressToPredict.setOnClickListener {
            val areaData  = inputArea.text.toString().toFloatOrNull()
            val roomData = inputRooms.text.toString().toFloatOrNull()

            if (areaData != null && roomData != null){
                val ortEnvironment = OrtEnvironment.getEnvironment()
                val ortSession = createORTSession(ortEnvironment)
                val output = ortSession?.let { it1 -> executeModel(areaData, roomData, it1, ortEnvironment) }
                resultText.setText("Predicted Price is $${output}.")


            }else{
                Toast.makeText(this, "Invalid Input. Please enter valid numbers.", Toast.LENGTH_LONG).show()
            }
        }



    }

    private fun executeModel(area: Float, rooms: Float, ortSession: OrtSession, ortEnvironment: OrtEnvironment): Float {

            val getName = ortSession.inputNames?.iterator()?.next()
            val floatBufferInput = FloatBuffer.wrap(floatArrayOf(area, rooms))
            val tensorInput = OnnxTensor.createTensor(ortEnvironment, floatBufferInput, longArrayOf(1, 2))
            val result = ortSession.run(mapOf(getName to tensorInput))
            val output = result[0].value as Array<FloatArray>
            return output[0][0]


    }

    private fun createORTSession(ortEnvironment: OrtEnvironment?): OrtSession? {
        // Use openRawResource to get an InputStream for the raw resource
        val modelInputStream = resources.openRawResource(R.raw.house_price_model)
        // Read the bytes from the InputStream
        val modelFile = modelInputStream.readBytes()
        modelInputStream.close() // Don't forget to close the stream

        return ortEnvironment?.createSession(modelFile)
    }



}
